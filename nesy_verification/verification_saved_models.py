"""Provide verification bounds for the saved models"""
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch import float64
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from torchvision.datasets import MNIST

from nesy_verification.neural.model_definitions import SimpleEventCNN, SimpleEventCNNnoSoftmax
from nesy_verification.pgd import pgd
# from nesy_verification.verification_saved_models_softmax import round_tensor

BATCH_SIZE = 32
NUM_MAGNITUDE_CLASSES = 3
NUM_PARITY_CLASSES = 2
PRINT = False
NUM_SAMPLES = 20
MODEL_PATH = Path(__file__).parent.resolve() / "neural/saved_models/icl"

BOUND_PATH = Path(__file__).parent.resolve() / "neural/neural_bounds"
WITH_SOFTMAX = False
BOUND_PATH = BOUND_PATH / "with_softmax" if WITH_SOFTMAX else BOUND_PATH / "without_softmax"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pd.options.display.float_format = "{:.10f}".format

RESULTS_COLUMNS = [
    "mnist_id",
    "magnitude_0_lb",
    "magnitude_0_ub",
    "magnitude_1_lb",
    "magnitude_1_ub",
    "magnitude_2_lb",
    "magnitude_2_ub",
    "parity_0_lb",
    "parity_0_ub",
    "parity_1_lb",
    "parity_1_ub",
    "magnitude_prediction_idx",
    "magnitude_label_idx",
    "parity_prediction_idx",
    "parity_label_idx",
    "magnitude_correct",
    "parity_correct",
    "magnitude_safe",
    "parity_safe",
    "safe",
]


def approx_lte(x, y, atol=1e-5):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=float64)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=float64)

    return (x <= y).all() or (torch.isclose(x, y, atol=atol)).all()


def approx_gte(x, y, atol=1e-5):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=float64)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=float64)

    return (x >= y).all() or (torch.isclose(x, y, atol=atol)).all()


def bound_softmax(h_L, h_U, use_float64=False):
    """Given lower and upper input bounds into a softmax, calculate their concrete
    output bounds."""

    if use_float64:
        h_L = h_L.to(float64)
        h_U = h_U.to(float64)

    shift = h_U.max(dim=1, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = (exp_L / (torch.sum(exp_U, dim=1, keepdim=True) - exp_U + exp_L))  # TODO EdS: Check removed epsilon
    upper = (exp_U / (torch.sum(exp_L, dim=1, keepdim=True) - exp_L + exp_U))

    return lower, upper


def calculate_bounds(
    model: torch.nn.Module, dataloader, epsilon: float, with_softmax, method: str = "IBP", round_floats=False,
):
    """Calculate bounds for the provided model and dataset.

    Note that there is a magnitude classification task (num < 3, 3 < num < 6,
    num > 6) and a parity classification task, i.e. (even(num), odd(num))
    """

    print(f"Performing verification with an epsilon of {epsilon}")
    print(f"Using the bounding method: {method}") if PRINT else None

    df_results = pd.DataFrame(columns=RESULTS_COLUMNS)

    num_magnitude_samples_verified = 0
    num_magnitude_samples_correctly_classified = 0
    num_magnitude_samples_safe = 0
    num_parity_samples_verified = 0
    num_parity_samples_correctly_classified = 0
    num_parity_samples_safe = 0
    num_samples_safe = 0

    for dl_idx, (mnist_idx, inputs, labels) in enumerate(dataloader):
        print(f"Starting new batch") if PRINT else None

        magnitude_labels = torch.argmax(labels[:, :3], dim=1)
        parity_labels = torch.argmax(labels[:, 3:], dim=1)

        if dl_idx == 0:
            if DEVICE == "cuda":
                inputs = inputs.cuda()
                magnitude_labels = magnitude_labels.cuda()
                parity_labels = parity_labels.cuda()
                model = model.cuda()
            print("Running on", DEVICE)

            lirpa_model = BoundedModule(
                model,
                torch.empty_like(inputs),
                device=inputs.device,
                verbose=True,
            )

        ptb = PerturbationLpNorm(norm=np.inf, eps=epsilon)
        ptb_inputs = BoundedTensor(inputs, ptb)

        pred = lirpa_model(ptb_inputs)
        magnitude_preds = torch.argmax(pred[:, :3], dim=1)
        parity_preds = torch.argmax(pred[:, 3:], dim=1)

        lb, ub = lirpa_model.compute_bounds(
            x=(ptb_inputs,), method=method.split()[0]
        )
        if round_floats:
            pred = round_tensor(pred, decimal_places=5)
            lb, ub = round_tensor(lb, decimal_places=5), round_tensor(ub, decimal_places=5)

        # Sanity check that bounds are indeed higher/lower than output (both magnitude and parity)
        # TODO EdS: This is pre-softmax! Do the same check post softmax
        # TODO EdS: Might I need to put the numerical stability epsilon term back in?
        assert (lb <= pred).all().item()
        assert (ub >= pred).all().item()

        attack_results = pgd(model, epsilon, inputs, labels, final_layer=False)
        # num_successful_attacks = sum(not value for value in attacks_results)

        # Iterate over each element in batch, first handling magnitude verification
        for i in range(len(magnitude_labels)):
            if mnist_idx[i] == 2901:
                x = "break"

            new_row = {"mnist_id": mnist_idx[i].item()}

            # First check magnitude
            num_magnitude_samples_verified += 1

            print(
                f"Image {i} top-1 prediction is: {magnitude_preds[i]}, the "
                f"ground-truth is: {magnitude_labels[i]}"
            ) if PRINT else None

            magnitude_correct = (magnitude_preds[i] == magnitude_labels[i]).item()

            if magnitude_correct:
                num_magnitude_samples_correctly_classified += 1

            # Pass the bounds through a softmax bounding layer
            lb_magnitude, ub_magnitude = bound_softmax(lb[:, :3], ub[:, :3], use_float64=True)

            # Sanity check that post-softmax bounds are a valid probability, i.e. between 0 and 1
            assert (0 <= lb_magnitude).all(), f"Magnitude lower bound lower than 0"
            assert (lb_magnitude).all() <= 1, f"Magnitude lower bound greater than 1"
            assert (0 <= ub_magnitude).all(), f"Magnitude Upper bound lower than 0"
            assert (ub_magnitude).all() <= 1, f"Magnitude Upper bound greater than 1"
            softmax_magnitude_preds = torch.softmax(pred[i][:3], dim=0)
            if round_floats:
                softmax_magnitude_preds = round_tensor(softmax_magnitude_preds, decimal_places=5)
            assert approx_gte(softmax_magnitude_preds, lb_magnitude[i], 1e-5), f"{dl_idx} {i}"
            assert approx_lte(softmax_magnitude_preds, ub_magnitude[i], 1e-5), f"{dl_idx} {i}"

            truth_idx = int(magnitude_labels[i])

            # Check that the lower bound of the truth class is greater than
            # the upper bound of all other classes
            if (
                (
                    lb_magnitude[:, :3][i][truth_idx]
                    > torch.cat(
                        (
                            ub_magnitude[:, :3][i][:truth_idx],
                            ub_magnitude[:, :3][i][truth_idx + 1:],
                        )
                    )
                )
                .all()
                .item()
            ):
                magnitude_safe = True
                num_magnitude_samples_safe += 1
            else:
                magnitude_safe = False

            if attack_results[i] and magnitude_safe:
                a, adv_output, adv_input = pgd(model, epsilon, inputs, labels, final_layer=False, return_model_output=True)
                print(f"For the input {np.array(inputs[i,0,...])}")
                print(f"We have the classification {labels[i,...][:3]}")
                print(f"We have the adv input {np.array(adv_input[i,0, ...])}")
                print(f"We have the adv output: {adv_output[i,...]}")
                raise Exception("Attack was successful but the bounds were not safe!")

            for j in range(NUM_MAGNITUDE_CLASSES):
                indicator = "(ground-truth)" if j == magnitude_labels[i] else ""
                pred_indicator = (
                    "(prediction)" if j == magnitude_preds[i] else ""
                )
                safe_indicator = (
                    "(safe)" if j == magnitude_labels[i] and magnitude_safe else ""
                )
                print(
                    "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                        j=j,
                        l=lb_magnitude[i][j].item(),
                        u=ub_magnitude[i][j].item(),
                        ind=indicator,
                        pred=pred_indicator,
                        safe=safe_indicator,
                    )
                ) if PRINT else None
                new_row[f"magnitude_{j}_lb"] = lb_magnitude[i][j].item()
                new_row[f"magnitude_{j}_ub"] = ub_magnitude[i][j].item()
                new_row["magnitude_correct"] = magnitude_correct
                new_row["magnitude_prediction_idx"] = magnitude_preds[i].item()
                new_row["magnitude_label_idx"] = magnitude_labels[i].item()
                new_row["magnitude_safe"] = magnitude_safe

            # Second check parity
            num_parity_samples_verified += 1

            print(
                f"Image {i} top-1 parity prediction is: {parity_preds[i]}, "
                f"the ground-truth is: {parity_labels[i]}"
            ) if PRINT else None

            parity_correct = (parity_preds[i] == parity_labels[i]).item()

            if parity_correct:
                num_parity_samples_correctly_classified += 1

            # Pass the bounds through a softmax bounding layer
            # TODO EdS: A sanity check is required to check why I was getting
            #  different results when using a softmax layer in auto_LiRPA
            lb_parity, ub_parity = bound_softmax(lb[:, 3:], ub[:, 3:], use_float64=True)

            # Sanity check that post-softmax bounds are a valid probability, i.e. between 0 and 1
            assert (0 <= lb_parity).all(), f"Parity Lower bound lower than 0"
            assert (lb_parity <= 1).all(), f"Parity Lower bound greater than 1"
            assert (0 <= ub_parity).all(), f"Parity Upper bound lower than 0"
            assert (ub_parity <= 1).all(), f"Parity Upper bound greater than 1"
            softmax_parity_preds = torch.softmax(pred[i][3:], dim=0)
            if round_floats:
                softmax_parity_preds = round_tensor(softmax_parity_preds, decimal_places=5)
            assert approx_gte(softmax_parity_preds, lb_parity[i], 1e-5), f"{dl_idx} {i}"
            assert approx_lte(softmax_parity_preds, ub_parity[i], 1e-5), f"{dl_idx} {i}"

            truth_idx = int(parity_labels[i])

            # Check that the lower bound of the truth class is greater than
            # the upper bound of all other classes
            if (
                    (
                            lb_parity[:, :3][i][truth_idx]
                            > torch.cat(
                        (
                                ub_parity[:, :3][i][:truth_idx],
                                ub_parity[:, :3][i][truth_idx + 1:],
                        )
                    )
                    )
                    .all()
                    .item()
            ):
                parity_safe = True
                num_parity_samples_safe += 1
            else:
                parity_safe = False

            for j in range(NUM_PARITY_CLASSES):
                indicator = "(ground-truth)" if j == parity_labels[i] else ""
                pred_indicator = (
                    "(prediction)" if j == parity_preds[i] else ""
                )
                safe_indicator = (
                    "(safe)" if j == parity_labels[i] and parity_safe else ""
                )
                print(
                    "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind} {pred} {safe}".format(
                        j=j,
                        l=lb_parity[i][j].item(),
                        u=ub_parity[i][j].item(),
                        ind=indicator,
                        pred=pred_indicator,
                        safe=safe_indicator,
                    )
                ) if PRINT else None

                # Sanity check Vasilis' point
                lb_parity_digit_even = lb_parity[i][0].item()
                lb_parity_digit_odd = lb_parity[i][1].item()
                ub_parity_digit_even = ub_parity[i][0].item()
                ub_parity_digit_odd = ub_parity[i][1].item()

                assert ((lb_parity_digit_even + ub_parity_digit_odd) - 1.0 < 1e-12), f"{(lb_parity_digit_even + ub_parity_digit_odd) - 1.0}"
                assert ((lb_parity_digit_odd + ub_parity_digit_even) - 1.0 < 1e-12), f"{(lb_parity_digit_odd + ub_parity_digit_even)}"

                assert ((float(format(lb_parity_digit_even)) + float(format(ub_parity_digit_odd))) - 1.0 < 1e-12), f"{(lb_parity_digit_even + ub_parity_digit_odd) - 1.0}"
                assert ((float(format(lb_parity_digit_odd)) + float(format(ub_parity_digit_even))) - 1.0 < 1e-12), f"{(lb_parity_digit_odd + ub_parity_digit_even)}"

                new_row[f"parity_{j}_lb"] = lb_parity[i][j].item()
                new_row[f"parity_{j}_ub"] = ub_parity[i][j].item()
                new_row["parity_correct"] = parity_correct
                new_row["parity_prediction_idx"] = parity_preds[i].item()
                new_row["parity_label_idx"] = parity_labels[i].item()
                new_row["parity_safe"] = parity_safe
                new_row["epsilon"] = epsilon

            safe = True if magnitude_safe and parity_safe else False
            new_row["safe"] = safe
            if safe:
                num_samples_safe += 1

            new_row_df = pd.json_normalize(new_row)
            df_results = pd.concat([df_results, new_row_df], ignore_index=True)

    # Quick sanity check
    assert num_parity_samples_verified == num_magnitude_samples_verified

    print(f"----\nSUMMARY\n----")
    print(f"For the method: {method}")
    print(f"Num magnitude samples verified: {num_magnitude_samples_verified}")
    print(f"Num magnitude samples correctly classified: {num_magnitude_samples_correctly_classified}")
    print(f"Num magnitude samples safe: {num_magnitude_samples_safe}")
    print(f"Num parity samples verified: {num_parity_samples_verified}")
    print(f"Num parity samples correctly classified: {num_parity_samples_correctly_classified}")
    print(f"Num parity samples safe: {num_parity_samples_safe}")
    print()

    results_summary = {
        "method": f"{method}",
        "num_magnitude_samples_verified": f"{num_magnitude_samples_verified}",
        "num_magnitude_samples_correctly_classified": f"{num_magnitude_samples_correctly_classified}",
        "num_magnitude_samples_safe": f"{num_magnitude_samples_safe}",
        "num_parity_samples_verified": f"{num_parity_samples_verified}",
        "num_parity_samples_correctly_classified": f"{num_parity_samples_correctly_classified}",
        "num_parity_samples_safe": f"{num_parity_samples_safe}",
        "epsilon": f"{epsilon}",
    }

    filename = f"results_{epsilon}_no_softmax"
    filename += f"_rounded" if round_floats else ""
    save_results_to_csv(df_results, results_summary, filename)


def save_results_to_csv(results: pd.DataFrame, summary: dict, filename: str):
    results.to_csv(
        BOUND_PATH / f"{filename}.csv",
        index=False,
    )

    with open(
        BOUND_PATH / f"{filename}.json",
        "w",
    ) as file:
        json.dump(summary, file, indent=4)

    print(f"Saved to file { BOUND_PATH / f'{filename}.csv'}")


def load_model(model_filename: str, num_classes: int, with_softmax=False, log_softmax=False):
    if with_softmax:
        model = SimpleEventCNN(num_classes=num_classes, log_softmax=log_softmax)
    else:
        model = SimpleEventCNNnoSoftmax(num_classes=num_classes)

    model.load_state_dict(torch.load(MODEL_PATH / model_filename))
    return model


def load_datasets() -> Tuple:
    dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)

    train_indices = torch.load(MODEL_PATH / "train_indices.pt")
    test_indices = torch.load(MODEL_PATH / "test_indices.pt")
    train_subset = CustomSubset(dataset, train_indices)
    test_subset = CustomSubset(dataset, test_indices)

    train_dl = DataLoader([d for d in train_subset], batch_size=BATCH_SIZE)
    test_dl = DataLoader([d for d in test_subset], batch_size=BATCH_SIZE)
    return train_dl, test_dl


class CustomSubset(Subset):
    """Created a custom subset to have access to MNIST indices when creating the CSV."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.transform = transforms.PILToTensor()
        self.normalise = transforms.Compose(
            [
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def classify_label(self, x):
        """Given the MNIST digit, return the class label (for magnitude classification
        then parity classification)."""
        return [
            x < 3,
            3 <= x <= 6,
            x > 6,
            x % 2 == 0,
            x % 2 != 0,
        ]

    def __getitem__(self, idx):
        # one_hot_tensor = torch.nn.functional.one_hot(torch.tensor(self.dataset[self.indices[idx]][1]), NUM_MAGNITUDE_CLASSES)
        return (
            self.indices[idx],
            self.normalise(self.transform(self.dataset[self.indices[idx]][0]).float().double()),
            torch.tensor(
                self.classify_label(self.dataset[self.indices[idx]][1])
            ).float(),
        )


class CustomDataset(Dataset):
    """Created a custom subset to have access to MNIST indices when creating the CSV."""

    def __init__(self, data, indices):
        super().__init__(data, indices)

    def __getitem__(self, idx):
        return self.indices[idx], self.dataset[self.indices[idx]]


if __name__ == "__main__":
    train_dl, test_dl = load_datasets()

    if WITH_SOFTMAX:
        print("Verifying CNN with Softmax")
        BOUNDS_PATH = BOUND_PATH / "with_softmax"
        cnn = load_model("cnn_with_softmax.pt", 5, with_softmax=True)
    else:
        print("Verifying CNN without Softmax")
        BOUNDS_PATH = BOUND_PATH / "without_softmax"
        cnn = load_model("cnn_no_softmax.pt", 5, with_softmax=False)

    cnn.double()
    cnn.eval()

    for epsilon in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        calculate_bounds(cnn, test_dl, epsilon=epsilon, with_softmax=WITH_SOFTMAX)