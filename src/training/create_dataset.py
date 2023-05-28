import random
from collections import Counter
from typing import Any, Dict, List, Tuple

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

from constants import QUES_TEMPLATES


def load_dataset_from_hf(dataset_name: str, split: str = "train") -> Dataset:
    """Load dataset from HuggingFace datasets library

    Args:
        dataset_name (str): Name of dataset to load
        split (str, optional): Data split to load. Defaults to "train".

    Returns:
        Dataset: HuggingFace dataset object
    """
    dataset = load_dataset(dataset_name)
    return dataset


def generate_count_df(data: List[str], col_name: str) -> pd.DataFrame:
    """Generate a dataframe with the count of each unique item in a list of tuples

    Args:
        data (List[str]): List containing the data
        col_name (str): Name of the column to store the data

    Returns:
        pd.DataFrame: _description_
    """
    return pd.DataFrame(Counter(data).most_common(), columns=[col_name, "Data Count"])


def filter_dataset(dataset: Dataset, col_name: str, filter_count: int) -> Dataset:
    """Filter a dataset based on the count of a column

    Args:
        dataset (Dataset): HuggingFace dataset object
        col_name (str): Name of the column to filter
        filter_count (int): Max number of count in the column to filter by

    Returns:
        Dataset: Filtered HuggingFace dataset object
    """
    count_df = generate_count_df(dataset[col_name], col_name)
    selected_column_values = list(
        count_df[count_df["Data Count"] >= filter_count][col_name]
    )
    return dataset.filter(lambda example: example[col_name] in selected_column_values)


def generate_ds(
    dataset_name: str, cols_to_filter: List[str], filter_count_per_col: List[int]
) -> Dataset:
    """Generate a dataset from the arguments

    Args:
        dataset_name (str): Name of the dataset to load
        cols_to_filter (List[str]): List of column names to filter
        filter_count_per_col (List[int]): List of max count per column to filter by

    Returns:
        Dataset: Filtered huggingFace dataset object
    """
    dataset = load_dataset_from_hf(dataset_name)
    return dataset


def add_question(example: Dict[str, str], col_name: str) -> Dict[str, str]:
    """Add a question to a HuggingFace dataset example

    Args:
        example (Dict[str, str]): HuggingFace dataset example
        question (str): Question to add

    Returns:
        Dict[str, str]: HuggingFace dataset example with the question added
    """
    template_num = random.randint(0, 2)

    question = QUES_TEMPLATES[col_name][template_num]
    example["question"] = question
    example["answer"] = example[col_name]
    return example


def generate_questions_from_metadata(dataset: Dataset, metadata_cols: str) -> Dataset:
    """Generate a dataset from a config file

    Args:
        dataset (Dataset): HuggingFace dataset object
        metadata_col (str): Name of the column containing the metadata

    Returns:
        Dataset: Huggingface dataset object with the metadata question added
    """
    datasets = {}
    for metadata_col in metadata_cols:
        datasets[metadata_col] = dataset.map(
            add_question, fn_kwargs={"col_name": metadata_col}
        )

    return datasets


def generate_train_test_splits(
    datasets: Dict[str, Dataset], test_size: float = 0.1
) -> Tuple[Dataset, Dataset]:
    """Generate train and test splits from a dictionary of datasets

    Args:
        datasets (Dict[str, Dataset]): Dictionary of datasets
        test_size (float, optional): Test split size. Defaults to 0.2.

    Returns: TODO: Change this to return a dictionary of train and test splits
        Tuple[Dataset, Dataset]: Train and test splits
    """
    train_datasets = {}
    test_datasets = {}
    for metadata_col, dataset in datasets.items():
        dataset = dataset.shuffle()
        splitted_ds = dataset.train_test_split(
            test_size=test_size
        )  # TODO: stratify by metadata_col

        train_datasets[metadata_col] = splitted_ds["train"]
        test_datasets[metadata_col] = splitted_ds["test"]

    return train_datasets, test_datasets


def generate_annotations(datasets: Dict[str, Dataset]) -> List[Dict[str, str]]:
    """Generate annotations for a list of datasets (in VQA format)

    Args:
        datasets (Dict[str, Dataset]): Dictionary of datasets

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[int, Image]]: _description_
    """
    imageid2image = {}
    questions = []
    annotations = []
    for i, dataset_name in enumerate(datasets):
        dataset = datasets[dataset_name]
        for j, sample in enumerate(dataset):
            image_id = j + 1
            question_id = (i + 1) * (j + 1)

            imageid2image[image_id] = sample["image"]
            question = sample["question"]
            answer = sample["answer"]

            questions.append(
                {
                    "image_id": image_id,
                    "question": question,
                    "question_id": question_id,
                }
            )

            annotations.append(
                {
                    "answers": [
                        {"answer": answer, "answer_confidence": "yes", "answer_id": 1}
                    ],
                    "image_id": image_id,
                    "question_id": question_id,
                }
            )

    return questions, annotations, imageid2image


def generate_label_id_mapping(datasets: Dict[str, Dataset]) -> Tuple[Any]:
    """Generate a mapping between labels and ids

    Args:
        datasets (Dict[str, Dataset]): Dictionary of datasets

    Returns:
        Tuple[Any]: Label to id and id to label mappings
    """
    label2id = {}
    for dataset_name in datasets:
        unique_answers = set(datasets[dataset_name]["answer"])
        for answer in unique_answers:
            if answer not in label2id:
                label2id[answer] = len(label2id)

    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def add_score_labels_to_annotations(
    annotations: List[Dict[str, Any]], label2id: Dict[str, int]
) -> List[Dict[str, Any]]:
    """Add scores and labels to annotations

    Args:
        annottations (List[Dict[str, Any]]): Annotations
        label2id (Dict[str, int]):  Mapping between labels and ids

    Returns:
        List[Dict[str, Any]]: Annotations with scores and labels
    """
    for annotation in tqdm(annotations):
        answers = annotation["answers"]
        answer_count = {}
        for answer in answers:
            answer_ = answer["answer"]
            answer_count[answer_] = answer_count.get(answer_, 0) + 1
        labels = []
        scores = []
        for answer in answer_count:
            if answer not in list(label2id.keys()):
                continue
            labels.append(label2id[answer])
            score = 1
            scores.append(score)
        annotation["labels"] = labels
        annotation["scores"] = scores

    return annotations
