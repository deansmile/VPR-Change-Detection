import os
import sys

_pre_cwd = os.path.realpath(os.getcwd())

# this file should place under .../<this repo>/baselines-scripts/
# change working directory to <this repo>
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(".")
sys.path.append("baselines/DR-TANet")

import argparse
import json
import time

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn

# DR-TANet
import TANet

import datasets
import datasets.transform
import evaluation
import torch_utils

from py_utils.src.utils import ProgressTimer
from py_utils.src import utils_img


def yield_CD_evaluation(model, dataset, device="cuda", crop_shape=(504, 504)):

    for input_1, input_2, target in dataset:

        # transfer from rgb to bgr
        input_1 = input_1[:, [2, 1, 0], ...]
        input_2 = input_2[:, [2, 1, 0], ...]

        x = torch.concat([input_1, input_2], dim=1)
        x = x.to(device)

        with torch.no_grad():
            predict = model(x)  # (batch, 2, m, n)
            predict = torch.argmax(predict, dim=1)  # (batch, m, n)

        target = target.detach().cpu().numpy()  # (batch, m, n)
        predict = predict.detach().cpu().numpy()  # (batch, m, n)

        assert len(target.shape) == 3 and len(predict.shape) == 3
        assert target.shape == predict.shape

        # run through each item in a batch
        for pre, tar in zip(predict, target):

            pre = utils_img.center_crop_image(pre, crop_shape)
            tar = utils_img.center_crop_image(tar, crop_shape)

            R = evaluation.change_mask_metric(pre, tar)
            yield R["precision"], R["recall"], R["accuracy"], R["f1_score"]


def wrap_eval_dataset(figsize=None):

    transform_loader = datasets.transform.get_transform_loader("w_norm")

    # from DR-TANet datasets.py
    # https://github.com/Herrccc/DR-TANet/blob/main/datasets.py
    # commit 37cc3929833d61451b2fa4a92ccd4286cfc4fd34
    # line 213
    # img_t0_r_ = np.asarray(img_t0_r).astype('f').transpose(2, 0, 1) / 128.0 - 1.0

    # Now, we need to evaluate the mean and std for torchvision.transforms.Normalize
    # https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    # input / 128 - 1.0 -> (input - 128) / 128
    # So mean is 128 and std is 128
    mean = 128 / 255.0
    std = 128 / 255.0

    def wrapper(dataset, **kwargs):

        transform, target_transform = transform_loader(
            dataset, mean, std, figsize=figsize
        )

        trans_opts = {
            "transform": transform,
            "target_transform": target_transform,
        }

        trans_opts.update(kwargs)

        dataset = torch_utils.CDDataWrapper(dataset, **trans_opts)
        dataset = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        return dataset

    return wrapper


def eval_on_datasets(model, sets, reports, crop_shape=(504, 504)):

    for key, dataset in sets.items():
        if key in reports:
            continue

        evaluator = yield_CD_evaluation(
            model,
            dataset,
            device="cuda",
            crop_shape=crop_shape,
        )

        result = torch_utils.image_change_detection_evaluation(
            model,
            dataset,
            prefix=f"{key} ",
            dry_run=False,
            evaluator=evaluator,
        )

        if isinstance(result, tuple):
            result = result[0]

        reports[key] = result


def eval_on_inference_time(model, dataset):

    _target_sample_number = 10000
    _results = []

    prog = ProgressTimer()
    prog.tic(_target_sample_number)

    while len(_results) < _target_sample_number:

        for input_1, input_2, _ in dataset:

            if len(_results) >= _target_sample_number:
                break

            batch_size = len(input_1)

            input_1 = input_1[:, [2, 1, 0], ...]
            input_2 = input_2[:, [2, 1, 0], ...]

            x = torch.concat([input_1, input_2], dim=1)
            x = x.to("cuda")

            start = time.time()

            with torch.no_grad():
                predict = model(x)  # (batch, 2, m, n)
                predict = torch.argmax(predict, dim=1)  # (batch, m, n)

            end = time.time()

            _results.extend([(end - start) / batch_size] * batch_size)

            prog.toc(add=batch_size)

    return np.mean(_results)


def average_multi_datasets(model, sets, crop_shape=(504, 504)):

    overall = {}

    for key, dataset in sets.items():

        evaluator = yield_CD_evaluation(
            model,
            dataset,
            device="cuda",
            crop_shape=crop_shape,
        )

        _, details = torch_utils.image_change_detection_evaluation(
            model,
            dataset,
            prefix=f"{key} ",
            dry_run=False,
            return_details=True,
            evaluator=evaluator,
        )

        for i, j in details.items():
            x = overall.get(i, [])
            x.extend(j)
            overall[i] = x

    overall = {i: np.mean(j) for i, j in overall.items()}
    return overall


############################################
# EVALUATE ON COARSELY REGISTERED DATASETS #
############################################


def CMU_Test(model, evaluation_report):

    # load dataset
    wrapper = wrap_eval_dataset(figsize=(512, 512))

    # fmt: off
    test_sets = {
        "VL-CMU-CD": wrapper(datasets.get_dataset("VL_CMU_CD", mode="test")),
    }

    diff_test_sets_adj_1 = {
        "VL-CMU-CD (adjacent distance 1)": wrapper(
            datasets.get_dataset(
                "VL_CMU_CD_Diff_View",
                mode="test",
                adjacent_distance=1,
            )
        ),

        "VL-CMU-CD (adjacent distance -1)": wrapper(
            datasets.get_dataset(
                "VL_CMU_CD_Diff_View",
                mode="test",
                adjacent_distance=-1,
            )
        ),
    }

    diff_test_sets_adj_2 = {
        "VL-CMU-CD (adjacent distance 2)": wrapper(
            datasets.get_dataset(
                "VL_CMU_CD_Diff_View",
                mode="test",
                adjacent_distance=2,
            )
        ),

        "VL-CMU-CD (adjacent distance -2)": wrapper(
            datasets.get_dataset(
                "VL_CMU_CD_Diff_View",
                mode="test",
                adjacent_distance=-2,
            )
        ),
    }
    # fmt: on

    # evaluation
    report = evaluation_report.get("VL-CMU-CD", {})

    print("\nEvaluate on VL-CMU-CD (test)\n")
    x = report.get("test", {})
    eval_on_datasets(model, test_sets, x, crop_shape=(504, 504))

    print("\nEvaluate on VL-CMU-CD (adjacent distance 1)\n")
    if "VL-CMU-CD (adjacent distance 1)" not in x:
        x["VL-CMU-CD (adjacent distance 1)"] = average_multi_datasets(
            model, diff_test_sets_adj_1, crop_shape=(504, 504)
        )

    print("\nEvaluate on VL-CMU-CD (adjacent distance 2)\n")
    if "VL-CMU-CD (adjacent distance 2)" not in x:
        x["VL-CMU-CD (adjacent distance 2)"] = average_multi_datasets(
            model, diff_test_sets_adj_2, crop_shape=(504, 504)
        )

    print("\nEvaluate on VL-CMU-CD (Inference Time)\n")
    if "inference_time" not in x:
        x["inference_time"] = eval_on_inference_time(
            model, test_sets["VL-CMU-CD"]
        )

    report["test"] = x
    evaluation_report["VL-CMU-CD"] = report


def PSCD_Test(model, evaluation_report):

    wrapper = wrap_eval_dataset(figsize=(224, 224))

    # fmt: off
    test_sets = {
        "PSCD": wrapper(datasets.get_dataset("PSCD", mode="test")),
    }

    diff_test_sets_25 = {
        "PSCD (right 25%)": wrapper(
            datasets.get_dataset(
                "PSCD_Diff_View",
                mode="test",
                adjacent_distance=1,
            )
        ),

        "PSCD (left 25%) ": wrapper(
            datasets.get_dataset(
                "PSCD_Diff_View",
                mode="test",
                adjacent_distance=-1,
            ),
        ),
    }

    diff_test_sets_50 = {
        "PSCD (right 50%)": wrapper(
            datasets.get_dataset(
                "PSCD_Diff_View",
                mode="test",
                adjacent_distance=2,
            ),
        ),

        "PSCD (left 50%) ": wrapper(
            datasets.get_dataset(
                "PSCD_Diff_View",
                mode="test",
                adjacent_distance=-2,
            ),
        ),
    }

    # fmt: on

    # evaluation
    report = evaluation_report.get("PSCD", {})

    print("\nEvaluate on PSCD (test)\n")
    x = report.get("test", {})
    eval_on_datasets(model, test_sets, x, crop_shape=(224, 224))

    print("\nEvaluate on PSCD(Trans 25%)\n")
    if "PSCD (Trans 25%)" not in x:
        x["PSCD (Trans 25%)"] = average_multi_datasets(
            model, diff_test_sets_25, crop_shape=(224, 224)
        )

    print("\nEvaluate on PSCD (Trans 50%)\n")
    if "PSCD (Trans 50%)" not in x:
        x["PSCD (Trans 50%)"] = average_multi_datasets(
            model, diff_test_sets_50, crop_shape=(224, 224)
        )

    report["test"] = x
    evaluation_report["PSCD"] = report


def Overall_Test(evaluation_report):

    # overall_report = evaluation_report.get("overall", {})
    overall_report = {}

    N1 = len(datasets.get_dataset("VL_CMU_CD", mode="test"))
    N2 = len(datasets.get_dataset("PSCD", mode="test"))
    N3 = len(
        datasets.get_dataset(
            "PSCD_Diff_View",
            mode="test",
            adjacent_distance=1,
        )
    )
    N4 = len(
        datasets.get_dataset(
            "PSCD_Diff_View",
            mode="test",
            adjacent_distance=2,
        )
    )
    N5 = len(
        datasets.get_dataset(
            "VL_CMU_CD_Diff_View",
            mode="test",
            adjacent_distance=1,
        )
    )
    N6 = len(
        datasets.get_dataset(
            "VL_CMU_CD_Diff_View",
            mode="test",
            adjacent_distance=2,
        )
    )

    cmu_report = evaluation_report["VL-CMU-CD"]["test"]
    pscd_report = evaluation_report["PSCD"]["test"]

    for key in ["f1_score", "precision", "recall", "accuracy"]:

        x = overall_report.get("VL-CMU-CD + diff", {})
        x[key] = (
            cmu_report["VL-CMU-CD"][key] * N1
            + cmu_report["VL-CMU-CD (adjacent distance 1)"][key] * N5 * 2
            + cmu_report["VL-CMU-CD (adjacent distance 2)"][key] * N6 * 2
        ) / (N1 + N5 * 2 + N6 * 2)
        overall_report["VL-CMU-CD + diff"] = x

        x = overall_report.get("VL-CMU-CD + diff + PSCD", {})
        x[key] = (
            cmu_report["VL-CMU-CD"][key] * N1
            + cmu_report["VL-CMU-CD (adjacent distance 1)"][key] * N5 * 2
            + cmu_report["VL-CMU-CD (adjacent distance 2)"][key] * N6 * 2
            + pscd_report["PSCD"][key] * N2
        ) / (N1 + N2 + N5 * 2 + N6 * 2)
        overall_report["VL-CMU-CD + diff + PSCD"] = x

        x = overall_report.get("PSCD + diff", {})
        x[key] = (
            pscd_report["PSCD"][key] * N2
            + pscd_report["PSCD (Trans 25%)"][key] * N3 * 2
            + pscd_report["PSCD (Trans 50%)"][key] * N4 * 2
        ) / (N2 + N3 * 2 + N4 * 2)
        overall_report["PSCD + diff"] = x

        x = overall_report.get("VL-CMU-CD + diff + PSCD + diff", {})
        x[key] = (
            cmu_report["VL-CMU-CD"][key] * N1
            + cmu_report["VL-CMU-CD (adjacent distance 1)"][key] * N5 * 2
            + cmu_report["VL-CMU-CD (adjacent distance 2)"][key] * N6 * 2
            + pscd_report["PSCD"][key] * N2
            + pscd_report["PSCD (Trans 25%)"][key] * N3 * 2
            + pscd_report["PSCD (Trans 50%)"][key] * N4 * 2
        ) / (N1 + N2 + N3 * 2 + N4 * 2 + N5 * 2 + N6 * 2)
        overall_report["VL-CMU-CD + diff + PSCD + diff"] = x

    evaluation_report["overall"] = overall_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    return parser.parse_args()


def main():

    args = parse_args()
    path = args.path

    if not path.endswith(".pth"):
        raise ValueError("input path should be a checkpoint(.pth) file")

    if path != os.path.abspath(path):
        path = os.path.join(_pre_cwd, path)

    if not os.path.exists(path):
        raise ValueError("checkpoint directory does not exist")

    best_pth = os.path.normpath(path)
    checkpoint = torch.load(best_pth)

    model = TANet.TANet(
        encoder_arch="resnet18",
        local_kernel_size=1,
        stride=1,
        padding=0,
        groups=4,
        drtam=True,
        refinement=True,
    )

    model.load_state_dict(checkpoint)
    model = model.to("cuda")
    model = model.eval()

    eval_path = os.path.join(os.path.dirname(best_pth), "eval.json")

    # initialize evaluation report
    if os.path.exists(eval_path):
        with open(eval_path, "r") as fd:
            evals = json.load(fd)
    else:
        evals = {}

    CMU_Test(model, evals)
    PSCD_Test(model, evals)
    Overall_Test(evals)

    with open(eval_path, "w") as fd:
        json.dump(evals, fd, indent=4)


if __name__ == "__main__":

    main()
