import os
import torch
import pandas as pd
from pathlib import Path
from typing import Tuple


from torch import float64
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
import torchvision.transforms as transforms

from neural.model_definitions import SimpleEventCNN, SimpleEventCNNnoSoftmax


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

    train_dl = DataLoader([d for d in train_subset], batch_size=32)
    test_dl = DataLoader([d for d in test_subset], batch_size=32)
    return train_dl, test_dl



MODEL_PATH = Path(__file__).parent.resolve() / "saved_models/icl"

bounds_files = [
    MODEL_PATH / "results_0.1_no_softmax.csv",
    MODEL_PATH / "results_0.01_no_softmax.csv",
    MODEL_PATH / "results_0.001_no_softmax.csv",
    MODEL_PATH / "results_0.0001_no_softmax.csv",
    MODEL_PATH / "results_1e-05_no_softmax.csv",
]

train_dl, test_dl = load_datasets()

cnn_no_softmax = load_model("cnn_no_softmax.pt", 5, with_softmax=False)
cnn_no_softmax.double()
cnn_no_softmax.eval()

# cnn_softmax = load_model("cnn_softmax.pt", 5, with_softmax=True)
# cnn_softmax.double()
# cnn_softmax.eval()

for bounds_file in bounds_files:

    neural_bounds = pd.read_csv(str(bounds_file))

    for test_idx, idx_sequence in enumerate(test_dl):
        for i in range(idx_sequence[0].shape[0]):
            image_idx, img, label = idx_sequence[0][i].item(), idx_sequence[1][i, ...], idx_sequence[2][i, ...]

            # get all bounds for this image
            df_row = neural_bounds[neural_bounds["mnist_id"] == image_idx]
            loaded_bounds = {
                "less_than_3": [df_row.iloc[0, 1], df_row.iloc[0, 2]],  # smaller_than_3
                "between_3_and_6": [df_row.iloc[0, 3], df_row.iloc[0, 4]],  # between_3_and_6
                "greater_than_6": [df_row.iloc[0, 5], df_row.iloc[0, 6]],  # larger_than_6
                "even": [df_row.iloc[0, 7], df_row.iloc[0, 8]],  # even
            }

            # get the CNN prediction, unsqueeze because we don't have
            # batches here so we need an extra dimension in front
            input_img = torch.unsqueeze(img, dim=0)
            cnn_outputs = cnn_no_softmax(input_img)
            # cnn_outputs = round_tensor(cnn_outputs)

            # pass the predictions through a softmax
            softmax_mag = torch.softmax(cnn_outputs[:, :3], dim=1)
            softmax_par = torch.softmax(cnn_outputs[:, 3:], dim=1)
            # softmax_mag = cnn_outputs[:, :3]
            # softmax_par = cnn_outputs[:, 3:]

            (
                smaller_than_3_pred,
                between_3_and_6_pred,
                larger_than_6_pred,
            ) = softmax_mag[0].tolist()
            even_pred, _ = softmax_par[0].tolist()

            # compare with bounds
            assert approx_gte(smaller_than_3_pred, loaded_bounds["less_than_3"][0], 1e-12), f"a {test_idx} {image_idx} {i} {loaded_bounds['less_than_3'][0] - smaller_than_3_pred}"
            assert approx_lte(smaller_than_3_pred, loaded_bounds["less_than_3"][1], 1e-12), f"b {test_idx} {image_idx} {i} {loaded_bounds['less_than_3'][1] - smaller_than_3_pred,}"
            assert approx_gte(between_3_and_6_pred, loaded_bounds["between_3_and_6"][0], 1e-12), f"c {test_idx} {image_idx} {i} {loaded_bounds['between_3_and_6'][0] - between_3_and_6_pred}"
            assert approx_lte(between_3_and_6_pred, loaded_bounds["between_3_and_6"][1], 1e-12), f"d {test_idx} {image_idx} {i} {loaded_bounds['between_3_and_6'][1] - between_3_and_6_pred}"
            assert approx_gte(larger_than_6_pred, loaded_bounds["greater_than_6"][0], 1e-12), f"e {test_idx} {image_idx} {i} {loaded_bounds['greater_than_6'][0] - larger_than_6_pred}"
            assert approx_lte(larger_than_6_pred, loaded_bounds["greater_than_6"][1], 1e-12), f"f {test_idx} {image_idx} {i} {loaded_bounds['greater_than_6'][1] - larger_than_6_pred}"
            assert approx_gte(even_pred, loaded_bounds["even"][0], 1e-12), f"g {test_idx} {image_idx} {i} {loaded_bounds['even'][0] - even_pred}"
            assert approx_lte(even_pred, loaded_bounds["even"][1], 1e-12), f"h{test_idx} {image_idx} {i} {loaded_bounds['even'][1] - even_pred}"
