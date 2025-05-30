import os
import sys

_pre_cwd = os.path.realpath(os.getcwd())

# this file should place under .../<this repo>/scripts/
# change working directory to <this repo>
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(".")

import argparse

import torch
import torch.nn as nn

import datasets
import evaluation
import models

from py_utils.src import data_repo
from py_utils.src import utils
from py_utils.src import utils_img
from py_utils.src import utils_torch

import numpy as np
import cv2

_device = "cuda"

_wrapper = datasets.wrap_eval_dataset(
    {
        "transform-option": "wo_norm",
        "batch-size": 1,
        "num-workers": 1,
    },
    shuffle=False,
)

_dataset_dict = {
    "VL-CMU-CD": _wrapper(datasets.get_dataset("VL_CMU_CD", mode="test")),
    # "PSCD": _wrapper(datasets.get_dataset("PSCD", mode="test")),
    # "VL-CMU-CD-diff_1": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=1
    #     )
    # ),
    # "VL-CMU-CD-diff_-1": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=-1
    #     )
    # ),
    # "VL-CMU-CD-diff_2": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=2
    #     )
    # ),
    # "VL-CMU-CD-diff_-2": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=-2
    #     )
    # ),
    # "VL-CMU-CD-diff_3": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=3
    #     )
    # ),
    # "VL-CMU-CD-diff_-3": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=-3
    #     )
    # ),
    # "VL-CMU-CD-diff_4": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=4
    #     )
    # ),
    # "VL-CMU-CD-diff_-4": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=-4
    #     )
    # ),
    # "VL-CMU-CD-diff_5": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=5
    #     )
    # ),
    # "VL-CMU-CD-diff_-5": _wrapper(
    #     datasets.get_dataset(
    #         "VL_CMU_CD_Diff_View", mode="test", adjacent_distance=-5
    #     )
    # ),
}

# # different viewpoints
# _dataset_dict.update(
#     {
#         "VL-CMU-CD-diff-20-right": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(104, 0),
#         ),
#         "VL-CMU-CD-diff-20-left": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(-104, 0),
#         ),
#         "VL-CMU-CD-diff-20-up": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(0, -104),
#         ),
#         "VL-CMU-CD-diff-20-down": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(0, 104),
#         ),
#         "VL-CMU-CD-diff-40-right": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(205, 0),
#         ),
#         "VL-CMU-CD-diff-40-left": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(-205, 0),
#         ),
#         "VL-CMU-CD-diff-40-up": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(0, -205),
#         ),
#         "VL-CMU-CD-diff-40-down": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             translate0=(0, 205),
#         ),
#         "VL-CMU-CD-rot-30-CW": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             rotate_angle0=30,
#         ),
#         "VL-CMU-CD-rot-30-CCW": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             rotate_angle0=-30,
#         ),
#         "VL-CMU-CD-rot-60-CW": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             rotate_angle0=60,
#         ),
#         "VL-CMU-CD-rot-60-CCW": _wrapper(
#             datasets.get_dataset("VL_CMU_CD", mode="test"),
#             rotate_angle0=-60,
#         ),
#     }
# )

# # different viewpoints
# _dataset_dict.update(
#     {
#         "PSCD-diff-25-right": _wrapper(
#             datasets.get_dataset(
#                 "PSCD_Diff_View", mode="test", adjacent_distance=1
#             ),
#         ),
#         "PSCD-diff-25-left": _wrapper(
#             datasets.get_dataset(
#                 "PSCD_Diff_View", mode="test", adjacent_distance=-1
#             ),
#         ),
#         "PSCD-diff-50-right": _wrapper(
#             datasets.get_dataset(
#                 "PSCD_Diff_View", mode="test", adjacent_distance=2
#             ),
#         ),
#         "PSCD-diff-50-left": _wrapper(
#             datasets.get_dataset(
#                 "PSCD_Diff_View", mode="test", adjacent_distance=-2
#             ),
#         ),
#         "PSCD-rot-30-CW": _wrapper(
#             datasets.get_dataset("PSCD", mode="test"),
#             rotate_angle0=30,
#         ),
#         "PSCD-rot-30-CCW": _wrapper(
#             datasets.get_dataset("PSCD", mode="test"),
#             rotate_angle0=-30,
#         ),
#         "PSCD-rot-60-CW": _wrapper(
#             datasets.get_dataset("PSCD", mode="test"),
#             rotate_angle0=60,
#         ),
#         "PSCD-rot-60-CCW": _wrapper(
#             datasets.get_dataset("PSCD", mode="test"),
#             rotate_angle0=-60,
#         ),
#     }
# )


def get_model(path, dataset="VL-CMU-CD"):

    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    # load model
    key = [i for i in checkpoint["args"].keys() if "model" in i]
    assert len(key) == 1
    key = key[0]

    if dataset == "VL-CMU-CD":
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


def main(path, option, output):

    if "VL-CMU-CD" in option:
        dataset = "VL-CMU-CD"
    elif "PSCD" in option:
        dataset = "PSCD"
    else:
        raise ValueError("dataset not supported")

    model = get_model(path, dataset=dataset)
    dataset = _dataset_dict[option]

    repo = data_repo.DataRepositoryObserver(output)

    prog = utils.ProgressTimer()
    prog.tic(len(dataset))

    # save_dir_t0 = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/t0"
    # save_dir_t1 = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/t1"
    # save_dir_target = "/scratch/ds5725/alvpr/cd_datasets/PSCD/crop_test/target"
    # os.makedirs(save_dir_t0, exist_ok=True)
    # os.makedirs(save_dir_t1, exist_ok=True)
    # os.makedirs(save_dir_target, exist_ok=True)

    for i, (img0, img1, target, caption) in enumerate(dataset):

        name = dataset.dataset.dataset.filenames[i]
        name = name.replace(".png", "")

        img0 = img0.to(_device)  # (1, 3, H, W)
        img1 = img1.to(_device)  # (1, 3, H, W)

        with torch.no_grad():
            x = model(img0, img1, caption)  # (1, m, n, 2)
            x = torch.argmax(x, dim=-1)  # (1, m, n)

        x = x.squeeze().cpu().numpy()

        # img0_np = img0.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        # img1_np = img1.squeeze().permute(1, 2, 0).cpu().numpy()
        # target_np = target.squeeze().cpu().numpy()  # (H, W)

        # # --- Save img0 and img1 as color images ---
        # img0_bgr = (img0_np * 255).astype(np.uint8)[..., ::-1]  # RGB to BGR
        # img1_bgr = (img1_np * 255).astype(np.uint8)[..., ::-1]

        # cv2.imwrite(os.path.join(save_dir_t0, f"{name}.png"), img0_bgr)
        # cv2.imwrite(os.path.join(save_dir_t1, f"{name}.png"), img1_bgr)

        # # --- Save target as a grayscale mask (0 and 255) ---
        # target_mask = (target_np * 255).astype(np.uint8)
        # cv2.imwrite(os.path.join(save_dir_target, f"{name}.png"), target_mask)

        img0 = img0.squeeze().permute(1, 2, 0).cpu().numpy()
        target = target.squeeze().cpu().numpy()

        statistics = {}
        statistics.update(evaluation.change_mask_metric(x, target))
        statistics.update(
            evaluation.statistics_from_binary_img_pair(x, target)
        )

        x = utils_img.overlay_image(img0, [1, 0, 0], mask=x)

        repo.add_item(name, x, extension=".png", **statistics)
        prog.toc()

    print("Time Costs: ", prog.total_seconds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--option", type=str, choices=_dataset_dict.keys())
    parser.add_argument("--output", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    path = args.path
    option = args.option
    output = args.output

    if not path.endswith(".pth"):
        raise ValueError("input path should be a checkpoint(.pth) file")

    if path != os.path.abspath(path):
        path = os.path.join(_pre_cwd, path)

    if output != os.path.abspath(output):
        output = os.path.join(_pre_cwd, output)

    if not os.path.exists(path):
        raise ValueError("checkpoint directory does not exist")

    path = os.path.normpath(path)
    output = os.path.normpath(output)

    # print out the arguments
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    for i, j in checkpoint["args"].items():
        print(i)
        for k, l in j.items():
            print(f"\t{k}: {l}")
        print()

    os.makedirs(output, exist_ok=True)
    main(path, option, output)
