import os
import sys

_pre_cwd = os.path.realpath(os.getcwd())

# this file should place under .../<this repo>/scripts/
# change working directory to <this repo>
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(".")

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

import datasets
import models
import torch_utils

from py_utils.src import utils
from py_utils.src import utils_torch

_device = "cuda"

# TODO: REFACTOR
_crop = None


def eval_on_datasets(model, sets, reports):
    for key, dataset in sets.items():
        if key in reports:
            continue

        if _crop is None:
            evaluator = None
        else:
            evaluator = torch_utils._yield_CD_evaluation_after_cropping(
                model, dataset, _crop
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

    return reports


def eval_on_inference_time(model, dataset):

    _target_sample_number = 10000
    _results = []

    device = utils_torch.get_model_device(model)

    prog = utils.ProgressTimer()
    prog.tic(_target_sample_number)

    while len(_results) < _target_sample_number:

        for input_1, input_2, _ in dataset:

            if len(_results) >= _target_sample_number:
                break

            batch_size = len(input_1)

            input_1 = input_1.to(device)
            input_2 = input_2.to(device)

            start = time.time()

            with torch.no_grad():
                predict = model(input_1, input_2)  # (batch, m, n, 2)
                predict = torch.argmax(predict, dim=-1)  # (batch, m, n)

            end = time.time()

            _results.extend([(end - start) / batch_size] * batch_size)

            prog.toc(add=batch_size)

    return np.mean(_results)


def average_multi_datasets(model, sets):

    overall = {}

    for key, dataset in sets.items():

        if _crop is None:
            evaluator = None
        else:
            evaluator = torch_utils._yield_CD_evaluation_after_cropping(
                model, dataset, _crop
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


def get_model_name(path):
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    return checkpoint["args"]["model"]["name"]


def get_model(path, dataset="VL-CMU-CD"):

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model_name = checkpoint["args"]["model"]["name"]

    # load model
    key = [i for i in checkpoint["args"].keys() if "model" in i]
    assert len(key) == 1
    key = key[0]

    if dataset in ["VL-CMU-CD", "Our", "OurDataset", "S2Looking"]:
        if "resnet" in model_name:
            checkpoint["args"][key]["target-shp-row"] = 512
            checkpoint["args"][key]["target-shp-col"] = 512
        else:
            checkpoint["args"][key]["target-shp-row"] = 504
            checkpoint["args"][key]["target-shp-col"] = 504

    elif dataset == "PSCD":
        checkpoint["args"][key]["target-shp-row"] = 224
        checkpoint["args"][key]["target-shp-col"] = 224

    else:
        raise ValueError(f"dataset {dataset} not supported")

    model = models.get_model(**checkpoint["args"][key]).to(_device)
    model = nn.DataParallel(model)

    # load checkpoint
    model = utils_torch.load_grad_required_state(model, checkpoint["model"])

    utils_torch.freeze_model(model)
    model.eval()

    return model


############################################
# EVALUATE ON COARSELY REGISTERED DATASETS #
############################################

def OurDataset_Test(path, evaluation_report):

    model = get_model(path, dataset="Our")

    figsize = None
    if "resnet" in get_model_name(path):
        figsize = (512, 512)

    # prepare dataloader wrapper
    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 1,
            "num-workers": 1,
        },
        shuffle=False,
        figsize=figsize,
    )

    # load test set
    test_sets = {
        "OurDataset": wrapper(datasets.get_dataset("Our", mode="test"))
    }

    # prepare evaluation dict
    report = evaluation_report.get("OurDataset", {})

    print("\nEvaluate on Our Dataset (test)\n")
    x = report.get("test", {})
    eval_on_datasets(model, test_sets, x)

    report["test"] = x
    evaluation_report["OurDataset"] = report

def CMU_Test(path, evaluation_report):

    model = get_model(path, dataset="VL-CMU-CD")

    figsize = None
    if "resnet" in get_model_name(path):
        figsize = (512, 512)

    # load dataset
    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 1,
            "num-workers": 1,
        },
        shuffle=False,
        figsize=figsize,
    )

    # all test sets
    # fmt: off
    test_sets = {
        "VL-CMU-CD": wrapper(datasets.get_dataset("VL_CMU_CD", mode="test")),
    }

    # diff_test_sets_adj_1 = {
    #     "VL-CMU-CD (adjacent distance 1)": wrapper(
    #         datasets.get_dataset(
    #             "VL_CMU_CD_Diff_View",
    #             mode="test",
    #             adjacent_distance=1,
    #         )
    #     ),

    #     "VL-CMU-CD (adjacent distance -1)": wrapper(
    #         datasets.get_dataset(
    #             "VL_CMU_CD_Diff_View",
    #             mode="test",
    #             adjacent_distance=-1,
    #         )
    #     ),
    # }

    # diff_test_sets_adj_2 = {
    #     "VL-CMU-CD (adjacent distance 2)": wrapper(
    #         datasets.get_dataset(
    #             "VL_CMU_CD_Diff_View",
    #             mode="test",
    #             adjacent_distance=2,
    #         )
    #     ),

    #     "VL-CMU-CD (adjacent distance -2)": wrapper(
    #         datasets.get_dataset(
    #             "VL_CMU_CD_Diff_View",
    #             mode="test",
    #             adjacent_distance=-2,
    #         )
    #     ),
    # }
    # fmt: on

    # evaluation
    report = evaluation_report.get("VL-CMU-CD", {})

    print("\nEvaluate on VL-CMU-CD (test)\n")
    x = report.get("test", {})
    eval_on_datasets(model, test_sets, x)

    # print("\nEvaluate on VL-CMU-CD (adjacent distance 1)\n")
    # if "VL-CMU-CD (adjacent distance 1)" not in x:
    #     x["VL-CMU-CD (adjacent distance 1)"] = average_multi_datasets(
    #         model, diff_test_sets_adj_1
    #     )

    # print("\nEvaluate on VL-CMU-CD (adjacent distance 2)\n")
    # if "VL-CMU-CD (adjacent distance 2)" not in x:
    #     x["VL-CMU-CD (adjacent distance 2)"] = average_multi_datasets(
    #         model, diff_test_sets_adj_2
    #     )

    # print('\nEvaluate on "inference time"\n')
    # if "inference time" not in x:
    #     x["inference time"] = eval_on_inference_time(
    #         model, test_sets["VL-CMU-CD"]
    #     )

    report["test"] = x
    evaluation_report["VL-CMU-CD"] = report


def PSCD_Test(path, evaluation_report):

    model = get_model(path, dataset="PSCD")

    # load dataset
    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 1,
            "num-workers": 1,
        },
        shuffle=False,
    )

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
    eval_on_datasets(model, test_sets, x)

    # print("\nEvaluate on PSCD(Trans 25%)\n")
    # if "PSCD (Trans 25%)" not in x:
    #     x["PSCD (Trans 25%)"] = average_multi_datasets(
    #         model, diff_test_sets_25
    #     )

    # print("\nEvaluate on PSCD (Trans 50%)\n")
    # if "PSCD (Trans 50%)" not in x:
    #     x["PSCD (Trans 50%)"] = average_multi_datasets(
    #         model, diff_test_sets_50
    #     )

    report["test"] = x
    evaluation_report["PSCD"] = report

def S2Looking_Test(path, evaluation_report):
    model = get_model(path, dataset="S2Looking")

    figsize = None
    if "resnet" in get_model_name(path):
        figsize = (512, 512)

    # Prepare dataloader wrapper
    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 1,
            "num-workers": 1,
        },
        shuffle=False,
        figsize=figsize,
    )

    # Load test set
    test_sets = {
        "S2Looking": wrapper(datasets.get_dataset("S2Looking", mode="test"))
    }

    # Prepare evaluation dict
    report = evaluation_report.get("S2Looking", {})

    print("\nEvaluate on S2Looking Dataset (test)\n")
    x = report.get("test", {})
    eval_on_datasets(model, test_sets, x)

    report["test"] = x
    evaluation_report["S2Looking"] = report

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

        x = overall_report.get("PSCD + diff", {})
        x[key] = (
            pscd_report["PSCD"][key] * N2
            + pscd_report["PSCD (Trans 25%)"][key] * N3 * 2
            + pscd_report["PSCD (Trans 50%)"][key] * N4 * 2
        ) / (N2 + N3 * 2 + N4 * 2)
        overall_report["PSCD + diff"] = x

        x = overall_report.get("VL-CMU-CD + diff + PSCD", {})
        x[key] = (
            cmu_report["VL-CMU-CD"][key] * N1
            + cmu_report["VL-CMU-CD (adjacent distance 1)"][key] * N5 * 2
            + cmu_report["VL-CMU-CD (adjacent distance 2)"][key] * N6 * 2
            + pscd_report["PSCD"][key] * N2
        ) / (N1 + N2 + N5 * 2 + N6 * 2)
        overall_report["VL-CMU-CD + diff + PSCD"] = x

        x = overall_report.get("VL-CMU-CD + PSCD (no diff)", {})
        x[key] = (
            cmu_report["VL-CMU-CD"][key] * N1 + pscd_report["PSCD"][key] * N2
        ) / (N1 + N2)
        overall_report["VL-CMU-CD + PSCD (no diff)"] = x

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


def main(path):

    name = os.path.basename(path).replace(".pth", "")
    eval_path = os.path.join(os.path.dirname(path), f"eval.{name}.json")
    print("eval_path: ",eval_path)
    # initialize evaluation report
    if os.path.exists(eval_path):
        with open(eval_path, "r") as fd:
            evals = json.load(fd)
    else:
        evals = {}

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "epoch" not in evals:
        evals["epoch"] = checkpoint["epoch"]

    # perform a series evaluation
    # CMU_Test(path, evals)
    # PSCD_Test(path, evals)
    # Overall_Test(evals)
    # OurDataset_Test(path, evals)
    S2Looking_Test(path, evals)

    # save evaluation report
    with open(eval_path, "w") as fd:
        json.dump(evals, fd, indent=4)

    print(json.dumps(evals, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--crop", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    # print(torch.cuda.get_arch_list())  # Shows supported architectures
    # print(torch.__version__)  # PyTorch version
    # print(torch.version.cuda)  # CUDA version
    # exit()
    args = parse_args()
    path = args.path

    if args.crop:
        _crop = (504, 504)

    if not path.endswith(".pth"):
        raise ValueError("input path should be a checkpoint(.pth) file")

    if path != os.path.abspath(path):
        path = os.path.join(_pre_cwd, path)

    if not os.path.exists(path):
        raise ValueError("checkpoint directory does not exist")

    path = os.path.normpath(path)

    # print out the arguments
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    for i, j in checkpoint["args"].items():
        print(i)
        for k, l in j.items():
            print(f"\t{k}: {l}")
        print()

    main(path)
