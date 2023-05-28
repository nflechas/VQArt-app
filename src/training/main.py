import argparse

from create_dataset import (add_score_labels_to_annotations,
                            generate_annotations, generate_ds,
                            generate_label_id_mapping,
                            generate_questions_from_metadata,
                            generate_train_test_splits)
from utils import read_config
from vqa_dataset import VQADataset
from vqa_model import VQAForArt

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Train a VQA model")
    arg_parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file",
        default="test_config.yaml",
    )

    args = arg_parser.parse_args()
    config_file = args.config_file

    (
        dataset_name,
        checkpoint,
        batch_size,
        num_epochs,
        cols_to_filter,
        filter_count_per_col,
    ) = read_config(config_file)

    dataset = generate_ds(dataset_name, cols_to_filter, filter_count_per_col)

    datasets = generate_questions_from_metadata(dataset, cols_to_filter)

    train_ds, test_ds = generate_train_test_splits(datasets)

    questions, annotations, imageid2image = generate_annotations(train_ds)
    label2id, id2label = generate_label_id_mapping(train_ds)

    annotations = add_score_labels_to_annotations(annotations, label2id)

    train_vqa_dataset = VQADataset(
        questions=questions,
        annotations=annotations,
        processor_ckpt=checkpoint,
        imageid2image=imageid2image,
        id2label=id2label,
        label2id=label2id,
    )

    test_vqa_datasets = {}
    for ds_name, dataset in test_ds.items():
        test_ques, test_anns, test_imageid2image = generate_annotations(
            {ds_name: dataset}
        )
        test_anns = add_score_labels_to_annotations(test_anns, label2id)
        test_vqa_datasets[ds_name] = VQADataset(
            questions=test_ques,
            annotations=test_anns,
            processor_ckpt=checkpoint,
            imageid2image=test_imageid2image,
            id2label=id2label,
            label2id=label2id,
        )

    vqa_model = VQAForArt(
        model_checkpoint=checkpoint,
        dataset=train_vqa_dataset,
        batch_size=batch_size,
        epochs=num_epochs,
    )

    vqa_model.train_model()

    for ds_name, test_vqa_dataset in test_vqa_datasets.items():
        print(f"Evaluating on {ds_name} test set")
        acc = vqa_model.evaluate_model(test_vqa_dataset)
        print(f"Accuracy score: {acc}")

        print(f"Generating Confusion Matrix for {ds_name} test set")
        vqa_model.generate_confusion_matrix(test_vqa_dataset)
