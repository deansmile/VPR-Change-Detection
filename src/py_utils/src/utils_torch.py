import collections
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def check_module(x):

    if isinstance(x, nn.Module):
        return

    raise ValueError("Only accept nn.Module input.")


def seed_everything(seed=42, verbose=False):
    """
    Set the `seed` value for torch and numpy seeds. Also turns on
    deterministic execution for cudnn.

    Parameters:
    - seed:     A hashable seed value

    copied from AnyLoc(https://github.com/AnyLoc/AnyLoc)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if verbose:
        print(f"Seed set to: {seed} (type: {type(seed)})")


def is_model_pair_exact(model_1, model_2, verbose=False):
    """
    check state_dicts from model_1 and model_2 are exact or not.
    """
    check_module(model_1)
    check_module(model_2)

    def xprint(s):
        if verbose:
            print(s)

    prefix = "perform check on "

    exact = True
    for (name_1, tensor_1), (name_2, tensor_2) in zip(
        model_1.state_dict().items(), model_2.state_dict().items()
    ):

        if not exact and not verbose:
            break

        xprint(prefix + f"src -> {name_1}")
        xprint(" " * len(prefix) + f"dst -> {name_2}")

        r = torch.equal(tensor_1, tensor_2)
        exact &= r

        if r:
            xprint(" " * len(prefix) + "pass !")
        else:
            xprint(" " * len(prefix) + "fail !")

    return exact


def get_model_device(model):

    check_module(model)
    return next(model.parameters()).device


def freeze_model(model):

    check_module(model)
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model):

    check_module(model)
    for param in model.parameters():
        param.requires_grad = True


def is_all_frozen(model):

    check_module(model)
    for param in model.parameters():
        if not param.requires_grad:
            continue

        return False
    return True


def is_any_frozen(model):

    check_module(model)
    for param in model.parameters():
        if param.requires_grad:
            continue
        return True
    return False


def get_grad_required_state(model):
    """
    TODO: docstring
    """

    check_module(model)

    state = collections.OrderedDict()

    def write(x, prefix):
        for i, j in x.items():
            state[prefix + i] = j

    def dfs(x, prefix):

        if is_all_frozen(x):
            return

        if not is_any_frozen(x):
            write(x.state_dict(), prefix)
            return

        # dive into deeper (prefix naming is referred from torch source code)
        for name, module in x._modules.items():
            dfs(module, prefix + name + ".")

    dfs(model, "")

    return state


def load_grad_required_state(model, state, verbose=True, return_details=False):
    """
    TODO: docstring
    """

    check_module(model)

    state = state.copy()

    def xprint(x):
        if verbose:
            print(x)

    def write(x, prefix):
        names = [i for i in state.keys() if i.startswith(prefix)]

        n = len(prefix)
        pop_states = collections.OrderedDict()

        for name in names:
            pop_states[name[n:]] = state.pop(name)

        x.load_state_dict(pop_states, strict=True)

    def dfs(x, prefix):

        if is_all_frozen(x):
            return

        if not is_any_frozen(x):
            write(x, prefix)
            return

        # dive into deeper (prefix naming is referred from torch source code)
        for name, module in x._modules.items():
            dfs(module, prefix + name + ".")

    dfs(model, "")

    if len(state) > 0:
        for name in state.keys():
            xprint("<%s do not match in model>" % name)
    else:
        xprint("<All keys matched successfully>")

    if return_details:
        return model, state
    return model


def set_grad_required_layer_train(model):
    """
    TODO: docstring
    """

    check_module(model)

    model.train()

    if is_all_frozen(model):
        model.eval()
        return

    for module in model.children():
        module = set_grad_required_layer_train(module)


class CustomizedLRScheduler(optim.lr_scheduler._LRScheduler):
    """
    TODO: docstring
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        start_scale=0.0,
        warmup_epoch=-1,
        final_scale=0.0,
        total_epoch=-1,
        mode=None,
    ):

        if last_epoch > total_epoch:
            raise ValueError

        if start_scale > 1:
            raise ValueError

        self._check_mode(mode)

        self.mode = mode
        self.start_scale = start_scale
        self.final_scale = final_scale
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

        # implicitly do a series operation: steps() -> get_lr()
        # That is the reason why it need to put at the end to ensure
        # initialization is done.
        super(CustomizedLRScheduler, self).__init__(
            optimizer, last_epoch=last_epoch
        )

    @staticmethod
    def _check_mode(mode):
        if mode in ["cosine", "linear", "exp", None]:
            return
        raise ValueError

    def get_scale(self, epoch):
        # modified from C-3PO(https://github.com/DoctorKey/C-3PO)
        if epoch < 0:
            return 1

        if self.mode is None:
            return 1

        if self.mode == "exp":
            gamma = math.log(self.final_scale + 1e-7) / self.total_epoch
            ratio = math.exp(gamma * epoch)  # (1, final_scale)
            return ratio

        if self.mode == "linear":
            return 1 - (1 - self.final_scale) / self.total_epoch * epoch

        if self.mode == "cosine":

            # let `black` formatter ignore the following line
            # fmt: off
            x = epoch / self.total_epoch * math.pi  # (0, pi)
            x = math.cos(x) + 1                     # (2, 0)
            x = x / 2 * (1 - self.final_scale)      # (1 - final_scale, 0)
            x = x + self.final_scale                # (1, final_scale)
            # fmt: on

            return x

    def get_lr(self):

        if self.last_epoch < self.warmup_epoch:
            # warmup process
            target_scale = self.get_scale(self.warmup_epoch)
            ratio = 1.0 * self.last_epoch / self.warmup_epoch
            scale = (target_scale - self.start_scale) * ratio
            scale = self.start_scale + scale
        else:
            # applied specific mode
            scale = self.get_scale(self.last_epoch)

        return [base_lr * scale for base_lr in self.base_lrs]
