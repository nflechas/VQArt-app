import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViltForQuestionAnswering

from utils import collate_fn
from vqa_dataset import VQADataset


class VQAForArt:
    """VQA Model for Art Dataset"""

    def __init__(
        self,
        model_checkpoint: str,
        dataset: VQADataset,
        batch_size: int = 4,
        epochs: int = 1,
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs

        self.load_model()
        self.create_data_loader()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-5)

    def load_model(self):
        """Load the model from the checkpoint"""
        self.model = ViltForQuestionAnswering.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.dataset.id2label),
            id2label=self.dataset.id2label,
            label2id=self.dataset.label2id,
        )
        self.model.to("cuda")

    def create_data_loader(self):
        """Create the data loader"""
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, self.dataset.processor),
        )

    def train_model(self):
        self.model.train()
        for epoch in range(self.epochs):  # loop over the dataset multiple times
            print(f"Epoch: {epoch}")
            for batch in tqdm(self.data_loader):
                # get the inputs;
                batch = {k: v.to("cuda") for k, v in batch.items()}

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            print(f"Epoch: {epoch} finished | Loss: {loss.item()}")

        print("Finished Training")

    def save_model(self, path: str):
        """Save the model to the given path"""
        self.model.save_pretrained(path)

    def test_model(self, test_ds: VQADataset):
        predictions = []
        for example in test_ds:
            # add batch dimension + move to GPU
            example = {k: v.unsqueeze(0).to("cuda") for k, v in example.items()}

            # forward pass
            outputs = self.model(**example)

            logits = outputs.logits
            predicted_class = logits.argmax(-1).item()
            pred = self.dataset.id2label[predicted_class]
            predictions.append(pred)

        return predictions

    def evaluate_model(self, test_ds: VQADataset):
        """Evaluate the model on the test dataset"""
        predictions = self.test_model(test_ds)
        labels = [
            annotation["answers"][0]["answer"] for annotation in test_ds.annotations
        ]

        return accuracy_score(labels, predictions)

    def generate_confusion_matrix(self, test_ds):
        """Generate a confusion matrix for the given test dataset"""
        predictions = self.test_model(test_ds)
        labels = [
            annotation["answers"][0]["answer"] for annotation in test_ds.annotations
        ]

        label_names = [
            label for label in list(self.dataset.label2id.keys()) if label in labels
        ]

        cm = confusion_matrix(labels, predictions, labels=label_names)
        df_cm = pd.DataFrame(cm, index=label_names, columns=label_names)
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        plt.show()
