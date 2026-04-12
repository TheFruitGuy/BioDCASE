import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib
import wandb

joining_dict = {'bma': 'bmabz',
                'bmb': 'bmabz',
                'bmz': 'bmabz',
                'bmd': 'd',
                'bpd': 'd',
                'bp20': 'bp',
                'bp20plus': 'bp'}


def compute_confusion_matrix(ground_truth, predictions, all_classes):
    conf_matrix = pd.DataFrame(columns=['tp', 'fp', 'fn'], index=ground_truth.annotation.unique())
    for class_id in all_classes:
        ground_truth_class = ground_truth.loc[ground_truth.annotation == class_id]
        class_predictions = predictions.loc[predictions.annotation == class_id]
        conf_matrix.loc[class_id, 'tp'] = ground_truth_class['detected'].sum()
        conf_matrix.loc[class_id, 'fp'] = len(class_predictions) - class_predictions['correct'].sum()
        conf_matrix.loc[class_id, 'fn'] = len(ground_truth_class) - ground_truth_class['detected'].sum()

    conf_matrix['recall'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    conf_matrix['precision'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])
    conf_matrix['f1'] = 2 * conf_matrix['precision'] * conf_matrix['recall'] / (conf_matrix['precision'] + conf_matrix['recall'])

    return conf_matrix


def compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
    for dataset_name, ground_truth_dataset in ground_truth.groupby('dataset'):
        predictions_dataset = predictions.loc[predictions.dataset == dataset_name]
        conf_matrix_dataset = compute_confusion_matrix(ground_truth_dataset, predictions_dataset, all_classes)
        yield dataset_name, conf_matrix_dataset


def join_annotations_if_dir(path_to_annotations):
    if path_to_annotations.is_dir():
        annotations_list = []
        for annotations_path in path_to_annotations.glob('*.csv'):
            annotations = pd.read_csv(annotations_path, parse_dates=['start_datetime', 'end_datetime'])
            annotations_list.append(annotations)
        total_annotations = pd.concat(annotations_list, ignore_index=True)
    else:
        total_annotations = pd.read_csv(path_to_annotations, parse_dates=['start_datetime', 'end_datetime'])
        total_annotations['start_datetime'] = pd.to_datetime(total_annotations['start_datetime'],
                                                             utc=True).dt.tz_localize(None)
        total_annotations['end_datetime'] = pd.to_datetime(total_annotations['end_datetime'], utc=True).dt.tz_localize(None)
    return total_annotations


def run(predictions_path, ground_truth_path, iou_threshold=0.3):
    if type(predictions_path) is str:
        predictions_path = pathlib.Path(predictions_path)
    if type(ground_truth_path) is str:
        ground_truth_path = pathlib.Path(ground_truth_path)

    # Init wandb for evaluation
    wandb.init(project="biodcase-task2", job_type="evaluation", config={
        "iou_threshold": iou_threshold,
        "predictions_path": str(predictions_path),
        "ground_truth_path": str(ground_truth_path),
    })

    ground_truth = join_annotations_if_dir(ground_truth_path)
    predictions = join_annotations_if_dir(predictions_path)

    ground_truth = ground_truth.replace(joining_dict)
    predictions = predictions.replace(joining_dict)
    ground_truth['detected'] = 0
    predictions['correct'] = 0

    for (dataset_name, wav_path_name), wav_predictions in tqdm(predictions.groupby(['dataset', 'filename']),
                                                               total=len(predictions.filename.unique())):
        for class_id, class_predictions in wav_predictions.groupby('annotation'):
            for i, row in class_predictions.iterrows():
                mask = (ground_truth['filename'] == wav_path_name) & \
                       (ground_truth['annotation'] == class_id) & \
                       (ground_truth['detected'] == 0)
                ground_truth_not_detected = ground_truth.loc[mask]
                if ground_truth_not_detected.empty:
                    continue

                min_end = ground_truth_not_detected['end_datetime'].clip(upper=row['end_datetime'])
                max_start = ground_truth_not_detected['start_datetime'].clip(lower=row['start_datetime'])
                inter = (min_end - max_start).dt.total_seconds().clip(lower=0)
                union = (row['end_datetime'] - row['start_datetime']).total_seconds() + (
                    (ground_truth_not_detected['end_datetime'] - ground_truth_not_detected[
                        'start_datetime']).dt.total_seconds()) - inter
                iou = inter / union

                if iou.max() > iou_threshold:
                    predictions.loc[i, 'correct'] = 1
                    ground_truth_index = ground_truth_not_detected.iloc[iou.argmax()].name
                    ground_truth.loc[ground_truth_index, 'detected'] = 1

    all_classes = ground_truth.annotation.unique()

    # Log per-dataset results
    for dataset_name, conf_matrix_dataset in compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
        print(f'\nResults dataset {dataset_name}')
        print(conf_matrix_dataset)

        for class_id in conf_matrix_dataset.index:
            row = conf_matrix_dataset.loc[class_id]
            wandb.log({
                f"{dataset_name}/{class_id}/recall": row['recall'],
                f"{dataset_name}/{class_id}/precision": row['precision'],
                f"{dataset_name}/{class_id}/f1": row['f1'],
                f"{dataset_name}/{class_id}/tp": row['tp'],
                f"{dataset_name}/{class_id}/fp": row['fp'],
                f"{dataset_name}/{class_id}/fn": row['fn'],
            })

    # Log overall results
    conf_matrix = compute_confusion_matrix(ground_truth, predictions, all_classes)
    print('\nFinal results')
    print(conf_matrix)

    for class_id in conf_matrix.index:
        row = conf_matrix.loc[class_id]
        wandb.log({
            f"overall/{class_id}/recall": row['recall'],
            f"overall/{class_id}/precision": row['precision'],
            f"overall/{class_id}/f1": row['f1'],
            f"overall/{class_id}/tp": row['tp'],
            f"overall/{class_id}/fp": row['fp'],
            f"overall/{class_id}/fn": row['fn'],
        })

    # Log summary table
    wandb.log({"confusion_matrix": wandb.Table(dataframe=conf_matrix.reset_index().rename(columns={'index': 'class'}))})

    # Log overall averages
    wandb.summary["mean_recall"] = conf_matrix['recall'].mean()
    wandb.summary["mean_precision"] = conf_matrix['precision'].mean()
    wandb.summary["mean_f1"] = conf_matrix['f1'].mean()

    wandb.finish()


if __name__ == '__main__':
    predictions_csv_path = pathlib.Path(input('Where are the predictions in csv format?\n> '))
    ground_truth_csv_path = pathlib.Path(input('Where are the ground truth in csv format?\n> '))
    run(predictions_csv_path, ground_truth_csv_path)