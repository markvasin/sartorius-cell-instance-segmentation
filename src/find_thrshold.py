import argparse
import os

import fiftyone as fo
import pandas as pd
from utils.project_utils import get_data_path, read_json
import fiftyone.utils.coco as fouc


def find_threshold(dataset_name, image_path, label_path, model_results, label_type):
    # thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    print('Preparing dataset...')
    dataset_type = fo.types.COCODetectionDataset
    dataset = fo.Dataset.from_dir(dataset_type=dataset_type,
                                  data_path=image_path,
                                  labels_path=label_path,
                                  name=dataset_name,
                                  label_types=[label_type],
                                  include_id=True,
                                  label_field="")

    final_report = {}
    for i, threshold in enumerate(thresholds):
        print(f'### THRESHOLD: {threshold}')
        for name, path in model_results.items():
            name = name + '-' + str(i)
            print(f'Adding model prediction for {name}...')
            prediction = read_json(path)
            prediction = [ann for ann in prediction if ann['score'] > threshold]

            fouc.add_coco_labels(dataset, name, prediction, label_type=label_type, coco_id_field="coco_id")

            print('Running model evaluation')
            results = dataset.evaluate_detections(
                name,
                gt_field=label_type,
                method="coco",
                eval_key="eval",
                compute_mAP=True)

            report = results.report()
            results.print_report()
            final_report[threshold] = report
            dataset.delete_evaluation("eval")
            dataset.delete_sample_field(name)

    classes = ['shsy5y', 'astro', 'cort']
    data = {'class': classes}
    precision_df = pd.DataFrame(data)
    recall_df = pd.DataFrame(data)
    f1_df = pd.DataFrame(data)
    all_df = pd.DataFrame(data)

    for threshold, report in final_report.items():
        precision_df[threshold] = [round(report[cls]['precision'], 2) for cls in classes]
        recall_df[threshold] = [round(report[cls]['recall'], 2) for cls in classes]
        f1_df[threshold] = [round(report[cls]['f1-score'], 2) for cls in classes]
        all_df[f"p-r-f1-{threshold}"] = [
            f"{round(report[cls]['precision'], 2)}-{round(report[cls]['recall'], 2)}-{round(report[cls]['f1-score'], 2)}"
            for cls in classes]

    precision_df.to_csv(os.path.join(get_data_path(), 'eval_output', 'precision.csv'), index=False)
    recall_df.to_csv(os.path.join(get_data_path(), 'eval_output', 'recall.csv'), index=False)
    f1_df.to_csv(os.path.join(get_data_path(), 'eval_output', 'f1.csv'), index=False)
    all_df.to_csv(os.path.join(get_data_path(), 'eval_output', 'all.csv'), index=False)

    print('precision')
    print(precision_df)
    print('recall')
    print(recall_df)
    print('f1_score')
    print(f1_df)
    print('all')
    print(all_df)


def get_predictions(paths):
    pred_dir = os.path.join(get_data_path(), 'predictions')
    model_preds = {path: os.path.join(pred_dir, f'{path}.json') for path in paths}
    return model_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='car-damages-data', help="The path for image directory")
    parser.add_argument("--label_type", default='segmentations', help="The path for annotation")
    parser.add_argument("--predictions", nargs="+", default=['query_inst_1'])
    parser.add_argument('--threshold', type=float, default=0.5, help="The threshold to filter prediction")
    parser.add_argument('--eval_view', dest='eval_view', action='store_true')
    parser.set_defaults(eval_view=False)

    args = parser.parse_args()
    label_type = args.label_type
    prediction_paths = args.predictions
    threshold = args.threshold
    eval_view = args.eval_view
    image_path = os.path.join(get_data_path(), 'sartorius-cell-instance-segmentation')
    label_path = os.path.join(get_data_path(), 'sartorius-mask-data', 'annotations_val.json')
    dataset_name = args.dataset_name

    model_results = get_predictions(prediction_paths)
    find_threshold(dataset_name, image_path, label_path, model_results, label_type)
