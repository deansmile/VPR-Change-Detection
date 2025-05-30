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


def get_model(path, dataset="VL-CMU-CD"):

    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model_name = checkpoint["args"]["model"]["name"]

    # load model
    key = [i for i in checkpoint["args"].keys() if "model" in i]
    assert len(key) == 1
    key = key[0]

    if dataset == "VL-CMU-CD" and "resnet" in model_name:
        checkpoint["args"][key]["target-shp-row"] = 512
        checkpoint["args"][key]["target-shp-col"] = 512

    elif dataset == "VL-CMU-CD":
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


def CMU_translate(path, eval_path):

    model = get_model(path, "VL-CMU-CD")

    # check if the model is resnet18
    figsize = None
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "resnet18" in checkpoint["args"]["model"]["name"]:
        figsize = (512, 512)

    # initialize evaluation report
    evals = {}
    if os.path.exists(eval_path):
        with open(eval_path, "r") as fd:
            evals = json.load(fd)

    report = evals.get("VL-CMU-CD", {})
    diff_report = report.get("translate", {})

    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 4,
            "num-workers": 4,
        },
        shuffle=False,
        figsize=figsize,
    )

    origin_dataset = datasets.get_dataset("VL_CMU_CD", mode="test")

    for shift in range(256):

        if str(shift) in diff_report:
            continue

        diff_datasets = [
            wrapper(origin_dataset, translate0=(shift, 0)),
            wrapper(origin_dataset, translate0=(-shift, 0)),
            wrapper(origin_dataset, translate0=(0, shift)),
            wrapper(origin_dataset, translate0=(0, -shift)),
        ]

        overall = {}

        print("start shift:", shift)

        for diff_dataset in diff_datasets:

            evaluator = torch_utils._yield_CD_evaluation(
                model,
                diff_dataset,
            )

            if figsize is not None:

                evaluator = torch_utils._yield_CD_evaluation_after_cropping(
                    model,
                    diff_dataset,
                    (504, 504),
                )

            _, details = torch_utils.image_change_detection_evaluation(
                model,
                diff_dataset,
                verbose=True,
                evaluator=evaluator,
                return_details=True,
            )

            for i, j in details.items():
                x = overall.get(i, [])
                x.extend(j)
                overall[i] = x

        print("finished shift:", shift)

        overall = {i: np.mean(j) for i, j in overall.items()}
        diff_report[shift] = overall
        report["translate"] = diff_report
        evals["VL-CMU-CD"] = report

        with open(eval_path, "w") as fd:
            json.dump(evals, fd, indent=4)


def CMU_rotate(path, eval_path):

    model = get_model(path, "VL-CMU-CD")

    # check if the model is resnet18
    figsize = None
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "resnet18" in checkpoint["args"]["model"]["name"]:
        figsize = (512, 512)

    # initialize evaluation report
    evals = {}
    if os.path.exists(eval_path):
        with open(eval_path, "r") as fd:
            evals = json.load(fd)

    report = evals.get("VL-CMU-CD", {})
    rot_report = report.get("rotate", {})

    wrapper = datasets.wrap_eval_dataset(
        {
            "transform-option": "wo_norm",
            "batch-size": 4,
            "num-workers": 4,
        },
        shuffle=False,
        figsize=figsize,
    )

    origin_dataset = datasets.get_dataset("VL_CMU_CD", mode="test")

    for rot in range(60):

        if str(rot) in rot_report:
            continue

        rot_datasets = [
            wrapper(origin_dataset, rotate_angle0=-rot),
            wrapper(origin_dataset, rotate_angle0=rot),
        ]

        overall = {}

        print("start rotate:", rot)

        for rot_dataset in rot_datasets:

            evaluator = torch_utils._yield_CD_evaluation(
                model,
                rot_dataset,
            )

            if figsize is not None:

                evaluator = torch_utils._yield_CD_evaluation_after_cropping(
                    model,
                    rot_dataset,
                    (504, 504),
                )

            _, details = torch_utils.image_change_detection_evaluation(
                model,
                rot_dataset,
                verbose=True,
                evaluator=evaluator,
                return_details=True,
            )

            for i, j in details.items():
                x = overall.get(i, [])
                x.extend(j)
                overall[i] = x

        print("finished rotate:", rot)

        overall = {i: np.mean(j) for i, j in overall.items()}
        rot_report[rot] = overall
        report["rotate"] = rot_report
        evals["VL-CMU-CD"] = report

        with open(eval_path, "w") as fd:
            json.dump(evals, fd, indent=4)


def main(path):

    checkpoint_name = os.path.basename(path).replace(".pth", "")

    eval_path = os.path.join(
        os.path.dirname(path),
        f"eval.affine.{checkpoint_name}.json",
    )

    CMU_translate(path, eval_path=eval_path)
    CMU_rotate(path, eval_path=eval_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--crop", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    path = args.path

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
