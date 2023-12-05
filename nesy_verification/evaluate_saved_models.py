import os
import torch
from data_utils import MNISTSimpleEvents
from torch.utils.data import DataLoader, random_split
from models import SimpleEventCNN, SimpleEventCNNnoSoftmax
from sklearn.metrics import classification_report

saved_models_path = os.path.join(os.getcwd(), "nesy_verification/saved_models")

cnn_with_softmax = SimpleEventCNN(num_classes=5)
cnn_with_softmax.load_state_dict(
    torch.load(os.path.join(saved_models_path, "cnn_no_softmax.pt"))
)

cnn_no_softmax = SimpleEventCNNnoSoftmax(num_classes=5)
cnn_no_softmax.load_state_dict(
    torch.load(os.path.join(saved_models_path, "cnn_no_softmax.pt"))
)


dataset = MNISTSimpleEvents()
_, test_dataset = random_split(dataset, [0.8, 0.2])
test_dl = DataLoader(test_dataset, batch_size=len(test_dataset))
cnn_with_softmax.eval()
cnn_no_softmax.eval()


for test_inputs, test_labels in test_dl:
    test_outputs_softmax = cnn_with_softmax(test_inputs)
    test_outputs_no_softmax = cnn_no_softmax(test_inputs)

    magnitude_classification_softmax = classification_report(
        torch.argmax(test_labels[:, :3], dim=1),
        torch.argmax(test_outputs_softmax[:, :3], dim=1),
    )

    parity_classification_softmax = classification_report(
        torch.argmax(test_labels[:, 3:], dim=1),
        torch.argmax(test_outputs_softmax[:, 3:], dim=1),
    )

    magnitude_classification_no_softmax = classification_report(
        torch.argmax(test_labels[:, :3], dim=1),
        torch.argmax(test_outputs_no_softmax[:, :3], dim=1),
    )

    parity_classification_no_softmax = classification_report(
        torch.argmax(test_labels[:, 3:], dim=1),
        torch.argmax(test_outputs_no_softmax[:, 3:], dim=1),
    )


print(
    f"magnitude_classification_softmax \n {magnitude_classification_softmax} \n\n",
    f"parity_classification_softmax \n {parity_classification_softmax} \n\n",
    f"magnitude_classification_no_softmax \n {magnitude_classification_no_softmax} \n\n",
    f"parity_classification_no_softmax \n {parity_classification_no_softmax} \n\n",
)
