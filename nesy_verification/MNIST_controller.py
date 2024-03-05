import os
import torch
import pandas as pd
from pathlib import Path
from data.MNIST_data_utils import get_mnist_sequences
from nesy_verification.verification_saved_models import load_model
from arithmetic_circuit_bound_propagation import get_expression_min_max
from automaton_bound_propagation import get_state_min_max
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# TODO: inference is currently done stupidly and should be rewritten the way it is meant to be
# i.e. with transition matrices, circuits, etc. probably steal from Nick


def get_consistent_bounds(bounds_per_label, print_stuff=False):
    """
    To get consistent bounds for non-binary classes we do the following:

        1. Find the class with the loosest bounds. This is the class in
           which the difference upper bound - lower bound is maximized.

        2. Generate new bounds for this class
           The upper bound for this class is 1 - the lower bounds of the other classes
           The lower bound for this class is 1 - the upper bounds of the other classes
    """

    # find the class with the loosest bounds
    loosest_label = max(
        bounds_per_label.keys(),
        key=lambda label: bounds_per_label[label][1] - bounds_per_label[label][0],
    )

    # generate new bounds for this class
    new_bounds = [
        1
        - sum(
            [
                bounds_per_label[label][1]
                for label in bounds_per_label.keys()
                if label is not loosest_label
            ]
        ),
        1
        - sum(
            [
                bounds_per_label[label][0]
                for label in bounds_per_label.keys()
                if label is not loosest_label
            ]
        ),
    ]

    # update the bounds with the new bounds for the loosest class
    variable_bounds = {
        label: (bounds if label is not loosest_label else new_bounds)
        for label, bounds in bounds_per_label.items()
    }

    if print_stuff:
        print(
            "\n",
            f"Old bounds: \t\t{bounds_per_label} \n",
            f"Differences: \t\t{[bounds_per_label[label][1] - bounds_per_label[label][0] for label in bounds_per_label.keys()]} \n",
            f"Loosest label: \t{loosest_label} \n",
            f"New bounds: \t\t{variable_bounds} \n",
            f"Bound changes: \t{[(variable_bounds[label][0] - bounds_per_label[label][0], variable_bounds[label][1] - bounds_per_label[label][1]) for label in bounds_per_label.keys()]} \n",
        )

    return variable_bounds


def print_current_timestep(
    t,
    image_label,
    variable_bounds,
    transition_bounds,
    previous_state_probs,
    current_state_probs,
):
    var_names = {
        1: "smaller_than_3 \t\t\t\t",
        2: "between_3_and_6 \t\t\t",
        3: "larger_than_6 \t\t\t\t",
        4: "even \t\t\t\t\t",
    }

    expressions = {
        "t1": "even and larger_than_6 \t\t",
        "t2": "(~even) and (~larger_than_6) \t",
        "t3": "smaller_than_3 \t\t\t",
    }

    state_expressions = {
        "s0": "s0 * (1 - t1) \t\t",
        "s1": "s0 * t1 + s1 * (1 - t2) \t",
        "s2": "s1 * t2 + s2 * (1 - t3) \t",
        "s3": "s2 * t3 + s3 * 1 \t\t",
    }

    print("t =", t, ", image label =", image_label)
    print("-" * 70)

    for var, bounds in variable_bounds.items():
        print(f"{var_names[var]}: {[round(n, 4) for n in bounds]}")

    print()
    for tran, bounds in transition_bounds.items():
        print(f"{tran} = {expressions[tran]}: {[round(n, 4) for n in bounds]}")

    print()
    print("\t\t\t\t  Previous \t  Current")
    for state in previous_state_probs.keys():
        print(
            f"{state} = {state_expressions[state]}: {[round(n, 4) for n in previous_state_probs[state]]} \t {[round(n, 4) for n in current_state_probs[state]]}"
        )

    print("-" * 70)
    print()


def get_state_bounds_from_sequence(
    idx_sequence,
    neural_bounds,
    initial_state_prob_range,
    state_expressions,
    print_timestep_info=False,
):
    state_prob_range = initial_state_prob_range
    dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)

    for image_idx in idx_sequence:
        # get all bounds for this image
        df_row = neural_bounds[neural_bounds["mnist_id"] == image_idx]

        # get lower/upper bounds for the non-binary classes
        variable_bounds = {
            1: [df_row.iloc[0, 1], df_row.iloc[0, 2]],  # smaller_than_3
            2: [df_row.iloc[0, 3], df_row.iloc[0, 4]],  # between_3_and_6
            3: [df_row.iloc[0, 5], df_row.iloc[0, 6]],  # larger_than_6
        }

        # # this ensures that the bounds for non-binary classes are consistent
        # # i.e. choosing a value for all variables but one will always lead
        # # to a choice for the final variable which is within its range
        # variable_bounds = get_consistent_bounds(variable_bounds)

        # add lower and upper bounds for even
        # odd is not needed as this is a binary class
        variable_bounds[4] = [df_row.iloc[0, 7], df_row.iloc[0, 8]]

        # calculate the min/max for the boolean expressions of all transitions
        t1_min, t1_max = get_expression_min_max("t1", variable_bounds)  # type: ignore
        t2_min, t2_max = get_expression_min_max("t2", variable_bounds)  # type: ignore
        t3_min, t3_max = get_expression_min_max("t3", variable_bounds)  # type: ignore

        # bounds for the transition expressions for this timestep
        transition_bounds = {
            "t1": [t1_min, t1_max],
            "t2": [t2_min, t2_max],
            "t3": [t3_min, t3_max],
        }

        new_state_prob = {}
        for state in state_expressions.keys():
            new_state_prob[state] = get_state_min_max(
                state_prob_range,
                transition_bounds,
                state_expressions[state],
            )

        if print_timestep_info:
            print_current_timestep(
                idx_sequence.index(image_idx),
                dataset[image_idx][1],
                variable_bounds,
                transition_bounds,
                previous_state_probs=state_prob_range,
                current_state_probs=new_state_prob,
            )

        state_prob_range = new_state_prob

    return state_prob_range


def super_naive_inference(
    cnn,
    idx_sequence,
    initial_state_prob_range,
    state_expressions,
    print_timestep_info=False,
):
    state_prob_range = initial_state_prob_range

    dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)
    transform = transforms.PILToTensor()
    normalise = transforms.Normalize((0.1307,), (0.3081,))


    for image_idx in idx_sequence:
        input_img = torch.unsqueeze(
            normalise(transform(dataset[image_idx][0]).double()), 0
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

        # since we don't care about bounds now and we simply want to
        # do inference we use the same value as lower and upper bound
        variable_bounds = {
            1: [smaller_than_3_pred, smaller_than_3_pred],  # smaller_than_3
            2: [between_3_and_6_pred, between_3_and_6_pred],  # between_3_and_6
            3: [larger_than_6_pred, larger_than_6_pred],  # larger_than_6
            4: [even_pred, even_pred],
        }

        # calculate the min/max for the boolean expressions of all transitions
        t1_min, t1_max = get_expression_min_max("t1", variable_bounds)  # type: ignore
        t2_min, t2_max = get_expression_min_max("t2", variable_bounds)  # type: ignore
        t3_min, t3_max = get_expression_min_max("t3", variable_bounds)  # type: ignore

        # bounds for the transition expressions for this timestep
        transition_bounds = {
            "t1": [t1_min, t1_max],
            "t2": [t2_min, t2_max],
            "t3": [t3_min, t3_max],
        }

        new_state_prob = {}
        for state in state_expressions.keys():
            new_state_prob[state] = get_state_min_max(
                state_prob_range,
                transition_bounds,
                state_expressions[state],
            )

        if print_timestep_info:
            print_current_timestep(
                idx_sequence.index(image_idx),
                dataset[image_idx][1],
                variable_bounds,
                transition_bounds,
                previous_state_probs=state_prob_range,
                current_state_probs=new_state_prob,
            )

        state_prob_range = new_state_prob

    return state_prob_range


if __name__ == "__main__":
    # define the files containing the bounds for the simple events
    # in this case (MNIST) these are even(x)/odd(x), x<3, 3<x<6, 6<x
    BOUND_PATH = (
        Path(__file__).parent.resolve() / "neural/neural_bounds/without_softmax"
    )
    bounds_files = [
        BOUND_PATH / f"results_0.1_no_softmax.csv",
        BOUND_PATH / f"results_0.01_no_softmax.csv",
        BOUND_PATH / f"results_0.001_no_softmax.csv",
        BOUND_PATH / f"results_0.0001_no_softmax.csv",
        BOUND_PATH / f"results_1e-05_no_softmax.csv",
    ]

    # load the trained CNN
    # this is needed to carry out normal inference without bounds
    # we will measure the performance with the bounds against this
    cnn = load_model("cnn_no_softmax.pt", 5, with_softmax=False)
    cnn.double()
    cnn.eval()

    # define the automaton
    # e.g. the second row says that the automaton can enter
    # state 1 if it is in state 0 and it sees t1 in the input,
    # or if it is in state 1 and it sees (1 - t2) in the input
    state_expressions = {
        "s0": "s0 * (1 - t1)",
        "s1": "s0 * t1 + s1 * (1 - t2)",
        "s2": "s1 * t2 + s2 * (1 - t3)",
        "s3": "s2 * t3 + s3 * 1",
    }
    starting_state = "s0"

    # initial probability bounds for each state: the automaton
    # is deterministically in the starting state so that state
    # gets 1 for both the lower/upper bound while the rest get 0
    initial_state_prob_range = {
        state: [
            1.0 if state == starting_state else 0.0,
            1.0 if state == starting_state else 0.0,
        ]
        for state in state_expressions.keys()
    }

    for bounds_file in bounds_files:
        # get the bounds of the CNN on simple events for this epsilon
        neural_bounds = pd.read_csv(str(bounds_file))

        # get the test sequences for the MNIST task
        _, idx_sequences, sequence_labels, _ = get_mnist_sequences(
            num_sequences=100, only_test=True
        )

        true_positives = 0
        true_negatives = 0
        robust_true_positives = 0
        robust_true_negatives = 0
        for i, idx_sequence in enumerate(idx_sequences):
            # perform normal inference for this sequence to obtain the
            # probability distribution over states at the end of the sequence
            final_state_distribution = super_naive_inference(
                cnn,
                idx_sequence,
                initial_state_prob_range,
                state_expressions,
                print_timestep_info=False,
            )

            # compute bounds only if this sequence is a true positive or true negative
            # verification is not meaningful in false positives/negatives
            sequence_label = sequence_labels[i]
            prob_accept = final_state_distribution["s3"][0]
            true_positive = prob_accept > 0.5 and sequence_label == 1
            true_negative = prob_accept < 0.5 and sequence_label == 0
            true_positives += true_positive
            true_negatives += true_negative

            if true_positive or true_negative:

                # get the probability bounds for all states at the end of the sequence
                # for every timestep given bounds on the simple events this computes
                # bounds of boolean expressions over these simple events, and then
                # propagates these through the run of the automaton (i.e. through time)
                final_state_bounds = get_state_bounds_from_sequence(
                    idx_sequence,
                    neural_bounds,
                    initial_state_prob_range,
                    state_expressions,
                    print_timestep_info=False,
                )

                # check if our decision was robust given these bounds
                accept_state_lower_bound = final_state_bounds["s3"][0]

                if sequence_label == 0 and accept_state_lower_bound < 0.5:
                    robust_true_negatives += 1
                
                if sequence_label == 1 and accept_state_lower_bound > 0.5:
                    robust_true_positives += 1

        epsilon = str(bounds_file).split('/')[-1].split('_')[1]
        acc = (true_positives + true_negatives) / len(idx_sequences)
        veri_acc = (robust_true_positives + robust_true_negatives) / len(idx_sequences)
        robustness = (robust_true_positives + robust_true_negatives) / (true_positives + true_negatives)
        print("Epsilon:", epsilon, end="\t - \t")
        print(f"Accuracy: {acc}", end=", ")
        print(f"Verifiable Accuracy: {veri_acc}", end=", ")
        print(f"Robustness: {robustness}")