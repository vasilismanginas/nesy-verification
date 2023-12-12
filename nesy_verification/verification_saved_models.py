"""Provide verification bounds for the saved models"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.utils.data import DataLoader, random_split

from data_utils import MNISTSimpleEvents
from models import SimpleEventCNN, SimpleEventCNNnoSoftmax


def calculate_bounds(model: torch.nn.Module, is_magnitude_classification=True):
    """Calculate bounds for the provided model.

    Note that there is a magnitude classification task (num < 3, 3 < num < 6,
    num > 6) and a parity classification task, i.e. (even(num), odd(num))

    Args:
        is_magnitude_classification
    """

    for test_inputs, test_labels in test_dl:

        test_labels = torch.argmax(test_labels[:, :3], dim=1) if is_magnitude_classification else torch.argmax(test_labels[:, 3:], dim=1)
        n_classes = 3 if is_magnitude_classification else 2

        # image = image.to(torch.float32) / 255.0
        if torch.cuda.is_available():
            test_inputs = test_inputs.cuda()
            test_labels = test_labels.cuda()
            model = model.cuda()

        # wrap model with auto_LiRPA
        lirpa_model = BoundedModule(model, torch.empty_like(test_inputs), device=test_inputs.device, verbose=True)
        print("Running on", test_inputs.device)

        # compute bounds for the final output
        eps = 1e-2
        ptb = PerturbationLpNorm(norm=np.inf, eps=eps)
        ptb_test_inputs = BoundedTensor(test_inputs, ptb)

        # Get model prediction
        pred = lirpa_model(ptb_test_inputs)
        labels = torch.argmax(pred[:, :3], dim=1) if is_magnitude_classification else torch.argmax(pred[:, 3:], dim=1)

        print("1. Bound computation and comparisons of different methods.\n")

        for method in [
            "IBP",
            # "IBP+backward (CROWN-IBP)",  # TODO EdS: NB. Haven't implement backward LogSoftmax
        ]:
            print(f"Using the bounding method: {method}")

            lb, ub = lirpa_model.compute_bounds(x=(ptb_test_inputs,), method=method.split()[0])

            for i in range(len(test_labels)):
                print(
                    f"Image {i} top-1 prediction is: {labels[i]}, the ground-truth is: {test_labels[i]}"
                )

                # TODO EdS: Add check for correct classification

                for j in range(n_classes):
                    indicator = "(ground-truth)" if j == test_labels[i] else ""
                    print(
                        "f_{j}(x_0): {l:8.3f} <= f_{j}(x_0+delta) <= {u:8.3f} {ind}".format(
                            j=j, l=lb[i][j].item(), u=ub[i][j].item(), ind=indicator
                        )
                    )
            print()

        # print("2. Obtaining linear coefficients of the lower and upper bounds.\n")
        #
        # # There are many bound coefficients during CROWN bound calculation; here we are interested in the linear bounds
        # # of the output layer, with respect to the input layer (the image).
        # required_A = defaultdict(set)
        # required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
        #
        # for method in [
        #     "IBP+backward (CROWN-IBP)",
        #     # 'backward (CROWN)',
        #     # 'CROWN',
        #     # 'CROWN-Optimized (alpha-CROWN)'
        # ]:
        #     print("Bounding method:", method)
        #     if "Optimized" in method:
        #         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        #         lirpa_model.set_bound_opts(
        #             {"optimize_bound_args": {"iteration": 20, "lr_alpha": 0.1}}
        #         )
        #     lb, ub, A_dict = lirpa_model.compute_bounds(
        #         x=(image,),
        #         method=method.split()[0],
        #         return_A=True,
        #         needed_A_dict=required_A,
        #     )
        #     lower_A, lower_bias = (
        #         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]["lA"],
        #         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]["lbias"],
        #     )
        #     upper_A, upper_bias = (
        #         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]["uA"],
        #         A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]["ubias"],
        #     )
        #     print(
        #         f"lower bound linear coefficients size (batch, output_dim, *input_dims): {list(lower_A.size())}"
        #     )
        #     print(
        #         f"lower bound linear coefficients norm (smaller is better): {lower_A.norm()}"
        #     )
        #     print(
        #         f"lower bound bias term size (batch, output_dim): {list(lower_bias.size())}"
        #     )
        #     print(f"lower bound bias term sum (larger is better): {lower_bias.sum()}")
        #     print(
        #         f"upper bound linear coefficients size (batch, output_dim, *input_dims): {list(upper_A.size())}"
        #     )
        #     print(
        #         f"upper bound linear coefficients norm (smaller is better): {upper_A.norm()}"
        #     )
        #     print(
        #         f"upper bound bias term size (batch, output_dim): {list(upper_bias.size())}"
        #     )
        #     print(f"upper bound bias term sum (smaller is better): {upper_bias.sum()}")
        #     print(
        #         f"These linear lower and upper bounds are valid everywhere within the perturbation radii.\n"
        #     )
        #
        # ## An example for computing margin bounds.
        # # In compute_bounds() function you can pass in a specification matrix C, which is a final linear matrix applied to the last layer NN output.
        # # For example, if you are interested in the margin between the groundtruth class and another class, you can use C to specify the margin.
        # # This generally yields tighter bounds.
        # # Here we compute the margin between groundtruth class and groundtruth class + 1.
        # # If you have more than 1 specifications per batch element, you can expand the second dimension of C (it is 1 here for demonstration).
        # lirpa_model = BoundedModule(model, torch.empty_like(image), device=image.device)
        # C = torch.zeros(
        #     size=(len(true_label), 1, n_classes), device=image.device
        # )  # 3 1 10
        # groundtruth = (
        #     true_label.to(dtype=torch.int64, device=image.device)
        #     .unsqueeze(1)
        #     .unsqueeze(1)
        # )  # 3 1 1
        # target_label = (groundtruth + 1) % n_classes  # 3 1 1
        # C.scatter_(dim=2, index=groundtruth, value=1.0)
        # C.scatter_(dim=2, index=target_label, value=-1.0)
        # print("3. Computing bounds with a specification matrix.\n")
        # print("Specification matrix:\n", C)
        #
        # for method in [
        #     "IBP",
        #     "IBP+backward (CROWN-IBP)",
        #     #    'backward (CROWN)',
        #     #    'CROWN-Optimized (alpha-CROWN)'
        # ]:
        #     print("Bounding method:", method)
        #     if "Optimized" in method:
        #         # For optimized bound, you can change the number of iterations, learning rate, etc here. Also you can increase verbosity to see per-iteration loss values.
        #         lirpa_model.set_bound_opts(
        #             {
        #                 "optimize_bound_args": {
        #                     "iteration": 20,
        #                     "lr_alpha": 0.1,
        #                 }
        #             }
        #         )
        #     lb, ub = lirpa_model.compute_bounds(
        #         x=(image,), method=method.split()[0], C=C
        #     )
        #     for i in range(len(true_label)):
        #         print(
        #             "Image {} top-1 prediction {} ground-truth {}".format(
        #                 i, label[i], true_label[i]
        #             )
        #         )
        #         print(
        #             "margin bounds: {l:8.3f} <= f_{j}(x_0+delta) - f_{target}(x_0+delta) <= {u:8.3f}".format(
        #                 j=true_label[i],
        #                 target=(true_label[i] + 1) % n_classes,
        #                 l=lb[i][0].item(),
        #                 u=ub[i][0].item(),
        #             )
        #         )
        #     print()


if __name__ == "__main__":
    # saved_models_path = os.path.join(
    #     Path(__file__).parent.resolve(), "/saved_models"
    # )
    saved_models_path = "/Users/edward/github/imperial/nesy-verification/nesy_verification/saved_models"

    # Load the Softmax model
    cnn_with_softmax = SimpleEventCNN(num_classes=5)
    cnn_with_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_no_softmax.pt"))
    )

    # Load the non-Softmax model
    cnn_non_softmax = SimpleEventCNNnoSoftmax(num_classes=5)
    cnn_non_softmax.load_state_dict(
        torch.load(os.path.join(saved_models_path, "cnn_no_softmax.pt"))
    )

    # Getting test data
    dataset = MNISTSimpleEvents()
    N = len(dataset)
    N_test = int(N * 0.2)
    _, test_dataset = random_split(dataset, [N - N_test, N_test])

    test_dl = DataLoader(test_dataset, batch_size=1)

    cnn_with_softmax.eval()
    cnn_non_softmax.eval()

    print("Verifying CNN Softmax for Magnitude Classification ")
    calculate_bounds(cnn_with_softmax, is_magnitude_classification=True)
    print("--------------------------------------------------------")

    # print("Verifying CNN Softmax for Parity Classification ")
    # calculate_bounds(cnn_with_softmax, is_magnitude_classification=False)
    # print("--------------------------------------------------------")
