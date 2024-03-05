import os
import torch
import random
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from nesy_verification.verification_saved_models import MODEL_PATH


class MNISTSimpleEvents(Dataset):
    def __init__(self):
        sequences, indices, _, simple_events = get_mnist_sequences(num_sequences=2000)

        self.images, self.indices = [], []
        for sequence, index in zip(sequences, indices):
            self.images.extend(sequence)
            self.indices.extend(index)

        self.simple_event_labels = []
        for labels in simple_events:
            self.simple_event_labels.extend(labels)

        self.transform = transforms.PILToTensor()
        self.normalise = transforms.Compose(
            [
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            return (
                self.normalise(self.transform(self.images[idx]).double()),
                torch.tensor([float(label) for label in self.simple_event_labels[idx]]),
            )
        except Exception:
            x = 1


def get_mnist_sequences(num_sequences, only_test=False):
    dataset = MNIST(os.path.expanduser("~/.cache/mnist"), train=True, download=True)

    label2id = defaultdict(list)

    if not only_test:  # make sequences both for train and test
        for i, (_, label) in enumerate(dataset):  # type: ignore (type checkers don't work in this language)
            label2id[label].append(i)
    else:
        test_idxs = torch.load(MODEL_PATH / "test_indices.pt")

        for index in test_idxs:
            _, label = dataset[index]
            label2id[label].append(index)

    max_sequence_length, num_datapoints, positive_sequences, negative_sequences = (
        10,
        num_sequences,
        [],
        [],
    )

    even_larger_than_6, odd_smaller_than_6, smaller_than_3 = [8], [1, 3, 5], [0, 1, 2]

    while len(positive_sequences) < num_datapoints // 2:
        sequence_length = random.randint(3, max_sequence_length)
        transition_1_index = random.randint(0, sequence_length - 3)
        transition_2_index = random.randint(transition_1_index + 1, sequence_length - 2)
        transition_3_index = random.randint(transition_2_index + 1, sequence_length - 1)

        digit_1, digit_2, digit_3 = (
            random.choice(even_larger_than_6),
            random.choice(odd_smaller_than_6),
            random.choice(smaller_than_3),
        )

        sequence = (
            random.choices(
                list(set(range(10)) - set(even_larger_than_6)), k=transition_1_index
            )
            + [digit_1]
            + random.choices(
                list(set(range(10)) - set(odd_smaller_than_6)),
                k=transition_2_index - transition_1_index - 1,
            )
            + [digit_2]
            + random.choices(
                list(set(range(10)) - set(smaller_than_3)),
                k=transition_3_index - transition_2_index - 1,
            )
            + [digit_3]
            + random.choices(
                list(set(range(10))),
                k=sequence_length - transition_3_index - 1,
            )
        )

        positive_sequences.append(sequence)

    # There must be a better way to do this than random sampling but don't know
    while len(negative_sequences) < num_datapoints // 2:
        sequence_length = random.randint(3, max_sequence_length)
        candidate = [random.choice(range(10)) for _ in range(sequence_length)]

        found_first = -1
        for i in range(sequence_length):
            if candidate[i] in even_larger_than_6:
                found_first = i
                break

        if found_first == -1:
            negative_sequences.append(candidate)
            continue

        found_second = -1
        for i in range(found_first + 1, sequence_length):
            if candidate[i] in odd_smaller_than_6:
                found_second = i
                break

        if found_second == -1:
            negative_sequences.append(candidate)
            continue

        found_third = -1
        for i in range(found_second + 1, sequence_length):
            if candidate[i] in smaller_than_3:
                found_third = i
                break

        if found_third == -1:
            negative_sequences.append(candidate)

    sequences, idx_sequences, labels, simple_events = [], [], [], []

    for sequence in positive_sequences:
        image_seq = []
        idx_seq = []
        simple_event_seq = []
        for number in sequence:
            choice_idx = random.choice(label2id[number])
            idx_seq.append(choice_idx)
            image_seq.append(dataset[choice_idx][0])
            simple_event_seq.append(
                [
                    number < 3,
                    3 <= number <= 6,
                    number > 6,
                    number % 2 == 0,
                    number % 2 != 0,
                ]
            )
        sequences.append(image_seq)
        idx_sequences.append(idx_seq)
        simple_events.append(simple_event_seq)
        labels.append(1)

    for sequence in negative_sequences:
        image_seq = []
        idx_seq = []
        simple_event_seq = []
        for number in sequence:
            choice_idx = random.choice(label2id[number])
            idx_seq.append(choice_idx)
            image_seq.append(dataset[choice_idx][0])
            simple_event_seq.append(
                [
                    number < 3,
                    3 <= number <= 6,
                    number > 6,
                    number % 2 == 0,
                    number % 2 != 0,
                ]
            )
        sequences.append(image_seq)
        idx_sequences.append(idx_seq)
        simple_events.append(simple_event_seq)
        labels.append(0)

    return sequences, idx_sequences, labels, simple_events
