import argparse
import numpy as np
import pandas as pd
from loguru import logger

opt_overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)

def get_args_parser():
    parser = argparse.ArgumentParser('Compute AP', add_help=False)

    parser.add_argument('--result_file', default='results/our_single_scannet20_seen.csv', type=str)

    return parser

def evaluate_matches(
    results_file: str, clicks_num: int, len_gt_instances: int
):
    overlaps = opt_overlaps
    cur_true = np.ones(len_gt_instances)
    cur_score = np.ones(len_gt_instances) * (-float("inf"))
    cur_match = np.zeros(len_gt_instances, dtype=bool)
    ap = np.zeros((1, 1, len(overlaps)), float)
    for oi, overlap_th in enumerate(overlaps):
        hard_false_negatives = 0
        positives = 0
        cur_true = np.ones(len_gt_instances)
        cur_score = np.ones(len_gt_instances) * (-float("inf"))
        cur_match = np.zeros(len_gt_instances, dtype=bool)

        y_true = np.empty(0)
        y_score = np.empty(0)
        gti = 0
        with open(results_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                splits = line.rstrip().split(" ")
                # scene_name = splits[1].replace("scene", "")
                # object_id = splits[2]
                num_clicks = int(splits[3])
                iou = float(splits[4])

                if num_clicks == clicks_num:
                    if iou > overlap_th:
                        cur_match[gti] = True
                        cur_score[gti] = iou
                        positives += 1
                    else:
                        hard_false_negatives += 1
                        cur_score[gti] = iou
                    gti += 1

        # remove non-matched ground truth instances
        cur_true = cur_true[cur_match]
        cur_score = cur_score[cur_match]

        # append to overall results
        y_true = np.append(y_true, cur_true)
        y_score = np.append(y_score, cur_score)

        # sorting and cumsum
        score_arg_sort = np.argsort(y_score)
        y_score_sorted = y_score[score_arg_sort]
        y_true_sorted = y_true[score_arg_sort]
        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

        # unique thresholds
        (thresholds, unique_indices) = np.unique(
            y_score_sorted, return_index=True
        )
        num_prec_recall = len(unique_indices) + 1

        # prepare precision recall
        num_examples = len(y_score_sorted)
        num_true_examples = y_true_sorted_cumsum[-1]

        precision = np.zeros(num_prec_recall)
        recall = np.zeros(num_prec_recall)

        # deal with the first point
        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)

        # deal with remaining
        for idx_res, idx_scores in enumerate(unique_indices):
            cumsum = y_true_sorted_cumsum[idx_scores - 1]
            tp = num_true_examples - cumsum
            fp = num_examples - idx_scores - tp
            fn = cumsum + hard_false_negatives
            p = float(tp) / (tp + fp)
            r = float(tp) / (tp + fn)

            precision[idx_res] = p
            recall[idx_res] = r

        # first point in curve is artificial
        precision[-1] = 1.0
        recall[-1] = 0.0

        # compute average of precision-recall curve
        recall_for_conv = np.copy(recall)
        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
        recall_for_conv = np.append(recall_for_conv, 0.0)

        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")

        # integrate is now simply a dot product
        ap_current = np.dot(precision, stepWidths)

        ap[0, 0, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt_overlaps, 0.50))
    o25 = np.where(np.isclose(opt_overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt_overlaps, 0.25)))

    avg_dict = {}
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}

    for (li, label_name) in enumerate(["agnostic"]):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict


def get_num_instances(result_file: str) -> int:
    data = pd.read_csv(
        result_file,
        sep=" ",
        names=["id", "scene_id", "instance_id", "num_clicks", "iou"],
    )

    num_instances = 0

    for scene_id in data["scene_id"].unique():
        scene_data = data.loc[data["scene_id"] == scene_id]
        num_instances += np.unique(scene_data["instance_id"]).shape[0]

    return num_instances


def print_results(scores, num_clicks=-1):
    if num_clicks != -1:
        print(f"Results for {num_clicks} clicks.")
    print(f"AP:   {scores['all_ap']}")
    print(f"AP50: {scores['all_ap_50%']}")
    print(f"AP25: {scores['all_ap_25%']}")
    print("")


def evaluate(result_file: str):
    logger.info(f"Create scores for {result_file}")
    num_instances = get_num_instances(result_file)
    logger.info(
        f"Dataset contains {num_instances} ground truth instances in total"
    )

    for num_clicks in range(1, 21):
        ap_scores = evaluate_matches(result_file, num_clicks, num_instances)
        avgs = compute_averages(ap_scores)
        print_results(avgs, num_clicks=num_clicks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Script for computing AP for interactive single-object segmentation ', parents=[get_args_parser()])
    args = parser.parse_args()

    evaluate(result_file=args.result_file)
