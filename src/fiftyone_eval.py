import argparse
import os

import fiftyone as fo

from utils.project_utils import get_data_path, read_json
import fiftyone.utils.coco as fouc


def main(dataset_name, image_path, label_path, model_results, label_type, eval_view, score_threshold=0.5):
    print('Preparing dataset...')
    dataset_type = fo.types.COCODetectionDataset
    dataset = fo.Dataset.from_dir(dataset_type=dataset_type,
                                  data_path=image_path,
                                  labels_path=label_path,
                                  name=dataset_name,
                                  label_types=[label_type],
                                  include_id=True,
                                  label_field="")

    for i, (name, path) in enumerate(model_results.items()):
        print(f'Adding model prediction for {name}...')
        prediction = read_json(path)
        prediction = [ann for ann in prediction if ann['score'] > score_threshold]

        fouc.add_coco_labels(dataset, name, prediction, label_type=label_type, coco_id_field="coco_id")

        print('Running model evaluation')
        results = dataset.evaluate_detections(
            name,
            gt_field=label_type,
            method="coco",
            eval_key=f'eval{i}',
            compute_mAP=True,
            iou=0.5)

        print('mAP', results.mAP())
        results.print_report()

    # print('Launching app...')
    # session = fo.launch_app(dataset)

    # View patches in the App
    # if eval_view:
    #     print('Create evaluation view...')
    # view = dataset.to_evaluation_patches("eval")

    # view = dataset.sort_by("eval_fn", reverse=True)
    # session.view = view.take(20)
    # session.wait()


def get_predictions(paths):
    pred_dir = os.path.join(get_data_path(), 'predictions')
    model_preds = {path: os.path.join(pred_dir, f'{path}.json') for path in paths}
    return model_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default='sartorius', help="The path for image directory")
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
    'data/sartorius-cell-instance-segmentation/train'
    label_path = os.path.join(get_data_path(), 'sartorius-mask-data', 'annotations_val.json')
    dataset_name = args.dataset_name

    model_results = get_predictions(prediction_paths)
    main(dataset_name, image_path, label_path, model_results, label_type, eval_view, threshold)
