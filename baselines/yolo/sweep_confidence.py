import pandas as pd
import pathlib
import argparse

# Import the existing logic from your evaluation script
from evaluation import join_annotations_if_dir, joining_dict, compute_confusion_matrix

def run_sweep(predictions_path, ground_truth_path, iou_threshold=0.3):
    print("Loading data... (This might take a few seconds)")
    ground_truth_full = join_annotations_if_dir(pathlib.Path(ground_truth_path))
    predictions_full = join_annotations_if_dir(pathlib.Path(predictions_path))

    # Map the classes to the evaluation 3-class format
    ground_truth_full = ground_truth_full.replace(joining_dict)
    predictions_full = predictions_full.replace(joining_dict)

    # The thresholds we want to test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(f"\n{'Conf':<10} | {'Mean Recall':<15} | {'Mean Precision':<15} | {'Mean F1':<15}")
    print("-" * 60)

    for conf in thresholds:
        # 1. Filter the predictions by the current confidence threshold
        predictions = predictions_full[predictions_full['confidence'] >= conf].copy()
        ground_truth = ground_truth_full.copy()

        ground_truth['detected'] = 0
        predictions['correct'] = 0

        # 2. Run the IoU overlap matching loop
        for (dataset_name, wav_path_name), wav_predictions in predictions.groupby(['dataset', 'filename']):
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

        # 3. Compute final metrics
        all_classes = ground_truth.annotation.unique()
        conf_matrix = compute_confusion_matrix(ground_truth, predictions, all_classes)

        mean_recall = conf_matrix['recall'].mean()
        mean_precision = conf_matrix['precision'].mean()
        mean_f1 = conf_matrix['f1'].mean()

        print(f"{conf:<10.2f} | {mean_recall:<15.4f} | {mean_precision:<15.4f} | {mean_f1:<15.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds to find the best F1 score")
    parser.add_argument('--predictions', type=str, required=True, help="Path to predictions.csv")
    parser.add_argument('--ground_truth', type=str, required=True, help="Path to ground truth annotations folder")
    args = parser.parse_args()

    run_sweep(args.predictions, args.ground_truth)