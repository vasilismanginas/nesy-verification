import torch
import pandas as pd
from pathlib import Path
from verification_saved_models import load_datasets, load_model, approx_gte, approx_lte


with_softmax = False

if with_softmax:
    file_ending = "with_softmax"
    cnn = load_model("cnn_with_softmax.pt", 5, with_softmax=True)
    BOUND_PATH = Path(__file__).parent.resolve() / "neural/neural_bounds/with_softmax"
else:
    file_ending = "no_softmax"
    cnn = load_model("cnn_no_softmax.pt", 5, with_softmax=False)
    BOUND_PATH = Path(__file__).parent.resolve() / "neural/neural_bounds/without_softmax"

bounds_files = [
    BOUND_PATH / f"results_0.1_{file_ending}.csv",
    BOUND_PATH / f"results_0.01_{file_ending}.csv",
    BOUND_PATH / f"results_0.001_{file_ending}.csv",
    BOUND_PATH / f"results_0.0001_{file_ending}.csv",
    BOUND_PATH / f"results_1e-05_{file_ending}.csv",
]

train_dl, test_dl = load_datasets()
cnn.double()
cnn.eval()


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
            cnn_outputs = cnn(input_img)
            # cnn_outputs = round_tensor(cnn_outputs)

            if with_softmax:
                # slice the cnn outputs for parity/magnitude
                softmax_mag = cnn_outputs[:, :3]
                softmax_par = cnn_outputs[:, 3:]
            else:
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

    print(f"All assertions passed for {bounds_file}")
