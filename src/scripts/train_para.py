import os
import sys
import numpy as np
_pre_cwd = os.path.realpath(os.getcwd())

# this file should place under .../<this repo>/scripts/
# change working directory to <this repo>
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(".")

import argparse
import json
import pickle

import torch
import torch.multiprocessing
import torch.nn as nn
import wandb
import yaml

torch.multiprocessing.set_sharing_strategy("file_system")

# local modules
import datasets
import models
import torch_utils
from py_utils.src import utils, utils_torch

import torch.distributed as dist
import torch.nn.parallel
import argparse

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

def cleanup_ddp():
    dist.destroy_process_group()

    
# setting environment
# if not utils.is_connect_to_network():
os.environ["WANDB_MODE"] = "offline"

# global variable
_dry = False
_seed = 123
_device = "cuda"
_verbose = True
_prefix = " " * 4

def strip_module_prefix(state_dict):
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}

def xprint(*args):
    if _verbose:
        print(_prefix, *args)


def train_one_epoch(
    model, criterion, scaler, optimizer, data_loader, verbose=False
):

    def xxprint(*x):
        if verbose:
            print(_prefix, *x)

    xxprint("")
    xxprint("*** train one epoch ***")
    
    # make any layer that requires_grad to training mode
    # and others in evaluation mode
    utils_torch.set_grad_required_layer_train(model)
    print("after set grad")

    prog = utils.ProgressTimer(verbose=False)
    prog.tic(len(data_loader))
    print("after prog")
    total_loss = 0
    data_inds = []
    batch_losses = []

    # try:
    #     print("Start for loop")
    #     for idx, (inds, (input_1, input_2, targets)) in enumerate(data_loader):
    #         print(f"In for loop, iter {idx}")  # This should print if loop starts
    # except Exception as e:
    #     print(f"[ERROR] Exception in DataLoader loop: {e}")

    # try:
    #     batch = next(iter(data_loader))
    #     print("Successfully loaded first batch.")
    #     inds, (input_1, input_2, targets) = batch
    #     print("Batch structure OK")
    #     print(f"indices: {inds}")
    #     print(f"input_1 shape: {input_1.shape}")
    #     print(f"input_2 shape: {input_2.shape}")
    #     print(f"targets shape: {targets.shape}")
    # except Exception as e:
    #     print(f"Error loading batch: {e}")
    print("start loop")
    # exit()
    # print("memory summary")
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    for idx, (inds, (input_1, input_2, targets, caption)) in enumerate(data_loader):
        # if idx == 0:
        #     print(torch.cuda.memory_summary(device=_device, abbreviated=False))
        # if torch.isnan(input_1).any() or torch.isnan(input_2).any() or torch.isnan(targets).any():
        #     print(f"[ERROR] NaN detected at batch {idx}")
        #     break

        # print("caption",caption)
        # exit()
        
        # print("in for loop iter",idx)
        data_inds.append(torch.as_tensor(inds))

        if idx == 0:
            xxprint("input_1: ", input_1.shape)
            xxprint("input_2: ", input_2.shape)
            xxprint("targets: ", targets.shape)
            xxprint("caption: ", caption)

        input_1 = input_1.to(_device)
        input_2 = input_2.to(_device)
        targets = targets.to(_device)

        optimizer.zero_grad()

        outputs = model(input_1, input_2, caption)   # (n, 504, 504, 2)
        # exit()
        transform_outputs = outputs.permute(0, 3, 1, 2)

        if idx == 0:
            xxprint("output: ", outputs.shape, "->", transform_outputs.shape)

        # not sure why it need to specify long format
        targets = targets.to(torch.long)

        loss = criterion(transform_outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] NaN or Inf detected in loss at batch {idx}")
            return  # Exit early to prevent crash

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

        msg = f"Train Epoch: [{idx}/{len(data_loader)}]"
        msg += f"\tLoss: {loss.item():.6f}"

        if idx % 20 == 0 or _dry:  # log_interval can be set as needed
            xprint(msg)

        prog.toc()

        if _dry:
            break

    # print("end for loop")
    xxprint("*** train one epoch ***")
    xxprint("")

    # Calculate average loss over the epoch
    average_loss = total_loss / len(data_loader)

    data_ind = torch.concat(data_inds)
    return average_loss, data_ind, batch_losses, prog.total_seconds


def main(args):
    
    setup_ddp()

    # print("in main")
    if _verbose:
        print(json.dumps(args, indent=4))

    utils_torch.seed_everything(_seed)

    ######### model selection #########
    # model = models.get_model(**args["model"]).to(_device)

    # # ✅ Convert BatchNorm to SyncBatchNorm before wrapping in DDP
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # # ✅ Then wrap with DDP
    # model = nn.parallel.DistributedDataParallel(
    #     model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
    # )
    # model._set_static_graph()
    # 🔍 DEBUG: List all named parameters with index
    # print("\n=== Model Parameter Index Map ===")
    # for idx, (name, param) in enumerate(model.module.named_parameters()):
    #     print(f"{idx}: {name}")
    # print("=== End of Parameter Index Map ===\n")
    # exit()
    ###################################

    ######### get batchify datasets #########

    opt = args["dataset"]

    # adjust dataset setting for resnet based model
    name = args["model"]["name"]  # model name
    opt["figsize"] = None
    if "resnet" in name:
        opt["figsize"] = 512

    if opt["dataset"] == "VL_CMU_CD":
        print("get CMU dataset")
        trainset = datasets.get_CMU_training_datasets(**opt)

        valset_1 = datasets.get_dataset("VL_CMU_CD", mode="val")
        testset_1 = datasets.get_dataset("VL_CMU_CD", mode="test")

        wrapper = datasets.wrap_eval_dataset(
            opt,
            shuffle=False,
            figsize=opt.get("figsize", None),
        )
        valset_1 = wrapper(valset_1)
        testset_1 = wrapper(testset_1)

    elif opt["dataset"] == "Our":
        print("get our dataset")
        trainset = datasets.get_CMU_training_datasets(**opt)

        valset_1 = datasets.get_dataset("Our", mode="val")
        testset_1 = datasets.get_dataset("Our", mode="test")

        wrapper = datasets.wrap_eval_dataset(
            opt,
            shuffle=False,
            figsize=opt.get("figsize", None),
        )
        valset_1 = wrapper(valset_1)
        testset_1 = wrapper(testset_1)

    elif opt["dataset"] == "S2Looking":
        print("get S2Looking dataset")
        trainset = datasets.get_CMU_training_datasets(**opt)

        valset_1 = datasets.get_dataset("S2Looking", mode="val")
        testset_1 = datasets.get_dataset("S2Looking", mode="test")

        wrapper = datasets.wrap_eval_dataset(
            opt,
            shuffle=False,
            figsize=opt.get("figsize", None),
        )
        valset_1 = wrapper(valset_1)
        testset_1 = wrapper(testset_1)

    elif opt["dataset"] == "PSCD":

        trainset = datasets.get_PSCD_training_datasets(**opt)

        pscd_opts = {"use_mask_t0": True, "use_mask_t1": False}

        valset_1 = datasets.get_dataset("PSCD", mode="val", **pscd_opts)
        testset_1 = datasets.get_dataset("PSCD", mode="test", **pscd_opts)

        wrapper = datasets.wrap_eval_dataset(opt, shuffle=False, figsize=np.array([504, 504]))
        valset_1 = wrapper(valset_1)
        testset_1 = wrapper(testset_1)

    elif opt["dataset"] == "Combined":
        print("get combined dataset")
        trainset = datasets.get_combined_dataset("train", **opt)
        valset_1 = datasets.get_combined_dataset("val", **opt)
        testset_1 = datasets.get_combined_dataset("test", **opt)
    else:
        raise NotImplementedError
    #########################################

    # print("dataset loaded")
    utils_torch.seed_everything(_seed)

    ######### optimizer #########
    opt = args["optimizer"]
    epochs = opt["epochs"]
    warmup_epoch = opt["warmup-epoch"]
    learn_rate = opt["learn-rate"]
    loss_weight = opt["loss-weight"]
    lr_scheduler = opt["lr-scheduler"]
    grad_scaler = opt["grad-scaler"]

    # set up loss weight
    weight = None
    if loss_weight:
        # this value comes from examples/datasets.ipynb
        weight = torch.tensor([0.025, 0.975]).float().to(_device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    scaler = None
    if grad_scaler:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    #############################

    ######### training and evaluation #########
    opt = args["evaluation"]
    freq_train = opt["trainset"]

    best_val_f1_score = -1

    early_stop_counter = 0
    early_stop_patience = args["optimizer"].get("early-stop", None)
    epochs = 200
    print("epochs: "+str(epochs))
    resume_ckpt_path = "/scratch/ds5725/alvpr/Robust-Scene-Change-Detection/src/output/cd_rs_lang/best.val.pth"
    if os.path.exists(resume_ckpt_path):
        print(f"Resuming from checkpoint: {resume_ckpt_path}")
        checkpoint = torch.load(resume_ckpt_path, map_location=_device)

        # overwrite model args to match checkpoint exactly
        args["model"] = checkpoint["args"]["model"]

        # Rebuild model with same config
        model = models.get_model(**args["model"]).to(_device)

        # Convert to SyncBN and wrap with DDP BEFORE loading state dict
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )

        # Load checkpoint safely using utility
        model = utils_torch.load_grad_required_state(model, checkpoint["model"])
        model._set_static_graph()

         # ✅ define optimizer before loading its state
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        if lr_scheduler.lower() != "none":
            lr_scheduler = utils_torch.CustomizedLRScheduler(
                optimizer,
                start_scale=0.0,
                warmup_epoch=warmup_epoch,
                final_scale=0.2 * learn_rate,
                total_epoch=epochs,
                mode=lr_scheduler,
            )
        else:
            lr_scheduler = None

        # ✅ now it's safe to load the states
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_f1_score = checkpoint["logs"]["evaluation.val"]["f1_score"]

    else:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 0
        best_val_f1_score = -1

        model = models.get_model(**args["model"]).to(_device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=True
        )
        model._set_static_graph()

        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
        if lr_scheduler.lower() != "none":
            lr_scheduler = utils_torch.CustomizedLRScheduler(
                optimizer,
                start_scale=0.0,
                warmup_epoch=warmup_epoch,
                final_scale=0.2 * learn_rate,
                total_epoch=epochs,
                mode=lr_scheduler,
            )
        else:
            lr_scheduler = None

    ######### optimizer setup #########
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    # if lr_scheduler.lower() != "none":
    #     lr_scheduler = utils_torch.CustomizedLRScheduler(
    #         optimizer,
    #         start_scale=0.0,
    #         warmup_epoch=warmup_epoch,
    #         final_scale=0.2 * learn_rate,
    #         total_epoch=epochs,
    #         mode=lr_scheduler,
    #     )
    # else:
    #     lr_scheduler = None

    # if os.path.exists(resume_ckpt_path):
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        
    for epoch in range(start_epoch, epochs):

        verbose = (epoch == 0) and _verbose
        utctime = utils.get_utc_time()

        xprint(f"===== running {epoch} epoch =====")
        xprint(utctime)

        # training
        loss, data_inds, batch_losses, train_time = train_one_epoch(
            model, criterion, scaler, optimizer, trainset, verbose
        )
        lr_scheduler.step()

        # logging batch loss
        N = len(batch_losses)
        for n, batch_loss in enumerate(batch_losses):
            wandb.log(
                {
                    "batch_report": {
                        "epoch": epoch,
                        "batch": epoch * N + n,
                        "loss": batch_loss,
                    },
                }
            )

        logs = {
            "loss": loss,
            "epoch": epoch,
            "time.train": train_time,
        }

        checkpoint = {
            "model": utils_torch.get_grad_required_state(model),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
            "data_inds": data_inds,
        }

        # evaluation on trainset
        if freq_train > 0 and (epoch + 1) % freq_train == 0:
            statics, time = torch_utils.image_change_detection_evaluation(
                model,
                trainset,
                verbose=_verbose,
                prefix="Trainset: ",
                return_duration=True,
                dry_run=_dry,
                device=_device,
            )

            logs["evaluation.train"] = statics
            logs["time.eval.train"] = time

        # evaluation on testset (original dataset)
        statics, time = torch_utils.image_change_detection_evaluation(
            model,
            testset_1,
            verbose=_verbose,
            prefix="Testset: ",
            return_duration=True,
            dry_run=_dry,
            device=_device,
        )

        logs["evaluation.test"] = statics
        logs["time.eval.test"] = time

        # evaluation on valset (original dataset)
        statics, time = torch_utils.image_change_detection_evaluation(
            model,
            valset_1,
            verbose=_verbose,
            prefix="Valset : ",
            return_duration=True,
            dry_run=_dry,
            device=_device,
        )

        logs["evaluation.val"] = statics
        logs["time.eval.val"] = time

        path = str(epoch) + ".layer"

        log_path = os.path.join(
            args["wandb"]["output-path"], "logs", path + ".pkl"
        )

        checkpoint_path = os.path.join(
            args["wandb"]["output-path"], "checkpoints", path + ".pth"
        )

        wandb.log(logs)
        with open(log_path, "wb") as fd:
            pickle.dump(logs, fd)

        checkpoint["logs"] = logs

        save_freq = args["wandb"]["save-checkpoint-freq"]
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            torch.save(checkpoint, checkpoint_path)

        # save the best validation
        if logs["evaluation.val"]["f1_score"] > best_val_f1_score:
            best_val_f1_score = logs["evaluation.val"]["f1_score"]
            early_stop_counter = 0
            best_val_path = os.path.join(
                args["wandb"]["output-path"], "best.val.pth"
            )
            torch.save(checkpoint, best_val_path)
        # else:
        #     early_stop_counter += 1
        #     xprint(f"No improvement in validation F1. Patience: {early_stop_counter}/{early_stop_patience}")

        # if early_stop_patience is not None and early_stop_counter >= early_stop_patience:
        #     xprint(f"Early stopping triggered after {early_stop_patience} no-improve epochs.")
        #     break

        if _dry:
            break

    # save the last checkpoint
    last_checkpoint_path = os.path.join(
        args["wandb"]["output-path"], "last.pth"
    )
    torch.save(checkpoint, last_checkpoint_path)
    cleanup_ddp()
    ###########################################


def get_output_path_by_utc(prefix="", suffix=""):

    # specify directory by using utc time
    # YYYY-MM-DD.hh-mm-ss
    utctime = utils.get_utc_time()
    output_folder = prefix + utctime + suffix
    output_path = os.path.join(os.getcwd(), "output", output_folder)

    # try to avoid race-condition(save checkpoint in the same output_path)
    cnt = 0
    while os.path.exists(output_path):
        if cnt == 0:
            output_path += "." + str(cnt)
            continue
        cnt += 1
        output_path = ".".join((output_path.split(".")[:-1] + [str(cnt)]))

    return output_path


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML configuration file")
    args = parser.parse_args()

    config = args.config
    config = (
        config
        if config == os.path.abspath(config)
        else os.path.join(_pre_cwd, args.config)
    )

    with open(config, "r") as fd:
        args = yaml.safe_load(fd)

    return args


if __name__ == "__main__":
    print("start program")
    # exit()
    args = parse_args()

    # setting global variable
    env = args["environment"]
    _dry = False
    _seed = env["seed"]
    _verbose = env["verbose"]
    _device = env["device"]

    # setting output directory
    output_path = args["wandb"]["output-path"]
    if len(output_path) == 0:
        output_path = get_output_path_by_utc()
    args["wandb"]["output-path"] = output_path

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)

    logs = {
        "project": args["wandb"]["project"],
        "name": args["wandb"]["name"],
        "config": args,
        "dir": output_path,
    }

    # save input arguments for future reference
    with open(os.path.join(output_path, "args.json"), "w") as fd:
        json.dump(args, fd, indent=4)

    if _dry:
        os.environ["WANDB_MODE"] = "offline"

    wandb.login()
    wandb.init(**logs)

    main(args)
