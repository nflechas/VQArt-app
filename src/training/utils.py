from typing import Any, Dict, List, Tuple

import torch
import yaml


def collate_fn(batch, processor):
    input_ids = [item["input_ids"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    token_type_ids = [item["token_type_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )

    # create new batch
    batch = {}
    batch["input_ids"] = torch.stack(input_ids)
    batch["attention_mask"] = torch.stack(attention_mask)
    batch["token_type_ids"] = torch.stack(token_type_ids)
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = torch.stack(labels)

    return batch


def read_config(config_file: str) -> Tuple[Any]:
    """Read YAML config file

    Args:
        config_file (str): Path to config file

    Returns:
        Tuple[Any]: Tuple containing the configuration parameters
    """
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset_name = config["dataset_name"]
    checkpoint = config["checkpoint"]
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]

    cols_to_filter = config["sequence_of_filter_cols"]
    filter_count_per_col = config["max_num_of_values_per_col"]

    return (
        dataset_name,
        checkpoint,
        batch_size,
        num_epochs,
        cols_to_filter,
        filter_count_per_col,
    )
