import os
import torch
import pandas as pd
from data.MNIST_data_utils import get_mnist_sequences
from neural.model_definitions import SimpleEventCNN
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def bound_softmax(h_L, h_U, use_float64=False):
    """Given lower and upper input bounds into a softmax, calculate their concrete
    output bounds."""

    shift = h_U.max(dim=1, keepdim=True).values
    exp_L, exp_U = torch.exp(h_L - shift), torch.exp(h_U - shift)
    lower = exp_L / (
        torch.sum(exp_U, dim=1, keepdim=True) - exp_U + exp_L
    )  # TODO EdS: Check removed epsilon
    upper = exp_U / (torch.sum(exp_L, dim=1, keepdim=True) - exp_L + exp_U)

    return lower, upper


models_dir = os.path.join(os.getcwd(), "nesy_verification/neural/saved_models/icl")
neural_bounds = pd.read_csv(
    os.path.join(os.getcwd(), "nesy_verification/neural_bounds/results_0.01_no_NaN.csv")
)

_, idx_sequences, _, _ = get_mnist_sequences(num_sequences=100, only_test=True)

cnn_no_softmax = SimpleEventCNN(num_classes=5, log_softmax=True)
cnn_no_softmax.load_state_dict(
    torch.load(os.path.join(models_dir, "cnn_no_softmax.pt"))
)
cnn_no_softmax.eval()

dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)
transform = transforms.PILToTensor()
normalise = transforms.Compose(
    [
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

for idx_sequence in idx_sequences:
    for image_idx in idx_sequence:
        # get all bounds for this image
        df_row = neural_bounds[neural_bounds["mnist_id"] == image_idx]
        variable_bounds = {
            1: [df_row.iloc[0, 1], df_row.iloc[0, 2]],  # smaller_than_3
            2: [df_row.iloc[0, 3], df_row.iloc[0, 4]],  # between_3_and_6
            3: [df_row.iloc[0, 5], df_row.iloc[0, 6]],  # larger_than_6
            4: [df_row.iloc[0, 7], df_row.iloc[0, 8]],  # even
        }

        # get the CNN prediction, unsqueeze because we don't have
        # batches here so we need an extra dimension in front
        input_img = torch.unsqueeze(
            normalise(transform(dataset[image_idx][0]).float()), 0
        )
        cnn_outputs = cnn_no_softmax(input_img)

        # # pass the predictions through a softmax
        # softmax_mag, _ = bound_softmax(
        #     cnn_outputs[:, :3],
        #     cnn_outputs[:, :3],
        #     use_float64=True,
        # )
        # softmax_par, _ = bound_softmax(
        #     cnn_outputs[:, 3:],
        #     cnn_outputs[:, 3:],
        #     use_float64=True,
        # )

        # (
        #     smaller_than_3_pred,
        #     between_3_and_6_pred,
        #     larger_than_6_pred,
        # ) = softmax_mag[0].tolist()
        # even_pred, _ = softmax_par[0].tolist()

        smaller_than_3_pred, between_3_and_6_pred, larger_than_6_pred = torch.exp(cnn_outputs[:, :3])[0].tolist()
        even_pred, _ = torch.exp(cnn_outputs[:, 3:])[0].tolist()


        # compare with bounds
        print(smaller_than_3_pred >= variable_bounds[1][0])
        print(smaller_than_3_pred <= variable_bounds[1][1])
        print(between_3_and_6_pred >= variable_bounds[2][0])
        print(between_3_and_6_pred <= variable_bounds[2][1])
        print(larger_than_6_pred >= variable_bounds[3][0])
        print(larger_than_6_pred <= variable_bounds[3][1])
        print(even_pred >= variable_bounds[4][0])
        print(even_pred <= variable_bounds[4][1])

        break
    break