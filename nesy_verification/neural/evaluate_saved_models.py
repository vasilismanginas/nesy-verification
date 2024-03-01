import os
import torch
from nesy_verification.data.MNIST_data_utils import MNISTSimpleEvents
from torch.utils.data import DataLoader, random_split
from model_definitions import SimpleEventCNN, SimpleEventCNNnoSoftmax
from sklearn.metrics import classification_report

models_dir = os.path.join(os.getcwd(), "nesy_verification/neural/saved_models/icl")

cnn_no_softmax = SimpleEventCNNnoSoftmax(num_classes=5)
cnn_no_softmax.load_state_dict(
    torch.load(os.path.join(models_dir, "cnn_no_softmax.pt"))
)


dataset = MNISTSimpleEvents()
num_train_samples, num_test_samples = round(0.8*len(dataset)), round(0.2*len(dataset))
_, test_dataset = random_split(dataset, [num_train_samples, num_test_samples])  # type: ignore
test_dl = DataLoader(test_dataset, batch_size=len(test_dataset))
cnn_no_softmax.eval()


for test_inputs, test_labels in test_dl:
    test_outputs_no_softmax = cnn_no_softmax(test_inputs)

    magnitude_classification_no_softmax = classification_report(
        torch.argmax(test_labels[:, :3], dim=1),
        torch.argmax(test_outputs_no_softmax[:, :3], dim=1),
    )

    parity_classification_no_softmax = classification_report(
        torch.argmax(test_labels[:, 3:], dim=1),
        torch.argmax(test_outputs_no_softmax[:, 3:], dim=1),
    )


print(
    f"magnitude_classification_no_softmax \n {magnitude_classification_no_softmax} \n\n",
    f"parity_classification_no_softmax \n {parity_classification_no_softmax} \n\n",
)
