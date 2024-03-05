import os
from pathlib import Path
import torch
import pandas as pd
from data.MNIST_data_utils import get_mnist_sequences
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from nesy_verification.verification_saved_models import (
    load_model,
    approx_gte,
    approx_lte,
)

BOUND_PATH = Path(__file__).parent.resolve() / "neural/neural_bounds/without_softmax"
bounds_files = [
    BOUND_PATH / f"results_0.1_no_softmax.csv",
    BOUND_PATH / f"results_0.01_no_softmax.csv",
    BOUND_PATH / f"results_0.001_no_softmax.csv",
    BOUND_PATH / f"results_0.0001_no_softmax.csv",
    BOUND_PATH / f"results_1e-05_no_softmax.csv",
]

cnn = load_model("cnn_no_softmax.pt", 5, with_softmax=False)
cnn.double()
cnn.eval()


_, idx_sequences, _, _ = get_mnist_sequences(num_sequences=100, only_test=True)

dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)
transform = transforms.PILToTensor()
normalise = transforms.Normalize((0.1307,), (0.3081,))

for bounds_file in bounds_files:
    neural_bounds = pd.read_csv(str(bounds_file))

    for test_idx, idx_sequence in enumerate(idx_sequences):
        for i, image_idx in enumerate(idx_sequence):
            # get all bounds for this image
            df_row = neural_bounds[neural_bounds["mnist_id"] == image_idx]

            loaded_bounds = {
                "less_than_3": [
                    df_row.iloc[0, 1],
                    df_row.iloc[0, 2],
                ],
                "between_3_and_6": [
                    df_row.iloc[0, 3],
                    df_row.iloc[0, 4],
                ],
                "greater_than_6": [
                    df_row.iloc[0, 5],
                    df_row.iloc[0, 6],
                ],
                "even": [
                    df_row.iloc[0, 7],
                    df_row.iloc[0, 8],
                ],
            }

            # get the CNN prediction, unsqueeze because we don't have
            # batches here so we need an extra dimension in front
            input_img = torch.unsqueeze(
                normalise(transform(dataset[image_idx][0]).double()),
                0
            )
            cnn_outputs = cnn(input_img)

            # pass the predictions through a softmax
            softmax_mag = torch.softmax(cnn_outputs[:, :3], dim=1)
            softmax_par = torch.softmax(cnn_outputs[:, 3:], dim=1)

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