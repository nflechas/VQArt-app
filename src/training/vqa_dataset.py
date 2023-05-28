from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import ViltProcessor


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(
        self,
        questions: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        processor_ckpt: str,
        imageid2image: Dict[int, Any],  # TODO: change to Image
        id2label: Dict[int, str],
        label2id: Dict[str, int],
    ):
        self.questions = questions
        self.annotations = annotations
        self.imageid2image = imageid2image
        self.id2label = id2label
        self.label2id = label2id

        self.load_processor(processor_ckpt)

    def load_processor(self, processor_ckpt: str):
        """Load the processor

        Args:
            processor_ckpt (str): Path to processor checkpoint
        """
        self.processor = ViltProcessor.from_pretrained(processor_ckpt)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # get image + text
        annotation = self.annotations[idx]
        questions = self.questions[idx]
        image = self.imageid2image[annotation["image_id"]]
        text = questions["question"]

        encoding = self.processor(
            image, text, padding="max_length", truncation=True, return_tensors="pt"
        )
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = annotation["labels"]
        scores = annotation["scores"]
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(self.id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding
