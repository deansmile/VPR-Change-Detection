import torch
from torchvision import transforms as tvf


def prepare_transform(dataset, mean, std, figsize=None):

    if figsize is None:
        orig_figsize = dataset.figsize
        figsize = orig_figsize // 14 * 14  # be compatible with DINOv2
        figsize = figsize.tolist()

    transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            tvf.Normalize(mean=mean, std=std),
        ]
    )

    target_transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            torch.squeeze,
        ]
    )

    return transform, target_transform


def prepare_transform_wo_normalization(dataset, figsize=None):

    orig_figsize = dataset.figsize
    if figsize is None:
        figsize = orig_figsize // 14 * 14  # be compatible with DINOv2
        figsize = figsize.tolist()

    transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
        ]
    )

    target_transform = tvf.Compose(
        [
            tvf.ToTensor(),
            tvf.CenterCrop(figsize),
            torch.squeeze,
        ]
    )

    return transform, target_transform


####################
# Module Interface #
####################

_transform_factory = {
    "w_norm": prepare_transform,
    "wo_norm": prepare_transform_wo_normalization,
}


def list_transform_option():
    return _transform_factory.keys()


def get_transform_loader(option):
    if option not in list_transform_option():
        raise ValueError

    return _transform_factory[option]


def get_transform(option, *args, **kwargs):
    transform_loader = get_transform_loader(option)
    return transform_loader(*args, **kwargs)
