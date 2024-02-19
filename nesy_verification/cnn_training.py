import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from models import SimpleEventCNN, SimpleEventCNNnoSoftmax
from data_utils import MNISTSimpleEvents
from nesy_verification.verification_saved_models import load_datasets

dataset = MNISTSimpleEvents()

N = len(dataset)
N_test = int(N * 0.2)
train_dataset, test_dataset = random_split(dataset, [N - N_test, N_test])

# TODO EdS: Instead of the below, import the same indices as used in cnn_training_no_softmax.py
train_indices = train_dataset.indices
test_indices = test_dataset.indices
dummy_indices = test_dataset.indices[0]

train_dl, test_dl = load_datasets()

num_epochs = 50
# log_cnn = SimpleEventCNN(num_classes=5, log_softmax=True)
cnn = SimpleEventCNN(num_classes=5, log_softmax=False)
cnn.double()

optimizer = optim.Adam(cnn.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    train_losses = []
    for mnist_indices, train_inputs, train_labels in train_dl:
        train_outputs = cnn(train_inputs)

        # calculate the loss for the magnitude task (num < 3, 3 < num < 6, num > 6)
        magnitude_loss = loss_function(train_outputs[:, :3], torch.argmax(train_labels[:, :3], dim=1))

        # calculate the loss for the parity task (even(num), odd(num))
        parity_loss = loss_function(train_outputs[:, 3:], torch.argmax(train_labels[:, 3:], dim=1))

        # sum them to get the total loss
        loss = magnitude_loss + parity_loss
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        cnn.eval()
        test_losses = []
        all_magnitude_labels = []
        all_magnitude_outputs = []
        all_parity_labels = []
        all_parity_outputs = []

        for mnist_index, test_inputs, test_labels in test_dl:
            test_outputs = cnn(test_inputs)

            test_magnitude_loss = loss_function(test_outputs[:, :3], torch.argmax(test_labels[:, :3], dim=1))
            test_parity_loss = loss_function(test_outputs[:, 3:], torch.argmax(test_labels[:, 3:], dim=1))
            test_losses.append((test_magnitude_loss + test_parity_loss).item())

            all_magnitude_labels.extend(torch.argmax(test_labels[:, :3], dim=1))
            all_magnitude_outputs.extend(torch.argmax(test_outputs[:, :3], dim=1))
            all_parity_labels.extend(torch.argmax(test_labels[:, 3:], dim=1))
            all_parity_outputs.extend(torch.argmax(test_outputs[:, 3:], dim=1))

    print(
        "Epoch {}/{} \t---\t loss (train): {:.4f}\t loss (test): {:.4f}\t f1_magnitude (test): {:.4f}\t f1_parity (test): {:.4f}".format(
            epoch + 1,
            num_epochs,
            sum(train_losses) / len(train_losses),
            sum(test_losses) / len(test_losses),
            f1_score(
                np.asarray(all_magnitude_labels),
                np.asarray(all_magnitude_outputs),
                average="macro",
            ),
            f1_score(
                np.asarray(all_parity_labels),
                np.asarray(all_parity_outputs),
            ),
        )
    )

    torch.save(cnn.state_dict(), os.path.join(os.getcwd(), f'saved_models/icl/cnn_softmax.pt'))
