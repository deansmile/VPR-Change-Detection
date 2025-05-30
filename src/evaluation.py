import numpy as np


def statistics_from_binary_img_pair(
    pred_img,
    target_img,
    return_details=False,
):

    pred_img = np.array(pred_img)
    target_img = np.array(target_img)

    pred_img = pred_img.astype(bool)
    target_img = target_img.astype(bool)

    TP_mask = pred_img & target_img
    FP_mask = pred_img & ~target_img
    FN_mask = ~pred_img & target_img
    TN_mask = ~pred_img & ~target_img

    R = {
        "TP": int(np.sum(TP_mask)),
        "FP": int(np.sum(FP_mask)),
        "FN": int(np.sum(FN_mask)),
        "TN": int(np.sum(TN_mask)),
    }

    details = {
        "TP": TP_mask,
        "FP": FP_mask,
        "FN": FN_mask,
        "TN": TN_mask,
    }

    if return_details:
        return R, details
    return R


def change_mask_metric(pred_img, target_img):

    statistics = statistics_from_binary_img_pair(
        pred_img, target_img, return_details=False
    )

    TP = statistics["TP"]
    FP = statistics["FP"]
    FN = statistics["FN"]
    TN = statistics["TN"]

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    f1_score = 2 * recall * precision / (precision + recall + 1e-8)

    R = {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1_score": f1_score,
    }

    return R
