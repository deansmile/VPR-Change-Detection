from .backbone_dinov2 import get_dino
from .backbone_resnet import ResNet
from .CD_model import (
    CoAttention,
    CrossAttention,
    SingleCrossAttention1,
    MergeTemporal,
    TemporalAttention,
    ResNet18CrossAttention,
)

_model_factory = {
    "dino2 + cross_attention": CrossAttention,
    "dino2 + single_cross_attention1": SingleCrossAttention1,
    "dino2 + merge_temporal": MergeTemporal,
    "dino2 + co_attention": CoAttention,
    "dino2 + temporal_attention": TemporalAttention,
    "resnet18 + cross_attention": ResNet18CrossAttention,
}

_kwargs_map = {
    "dino2": {
        "layer1": "layer1",
        "facet1": "facet1",
        "facet2": "facet2",
        "num-heads": "num_heads",
        "dropout-rate": "dropout_rate",
        "target-shp-row": "target_shp_row",
        "target-shp-col": "target_shp_col",
        "num-blocks": "num_blocks",
    },
    "resnet18": {
        "num-heads": "num_heads",
        "dropout-rate": "dropout_rate",
        "target-shp-row": "target_shp_row",
        "target-shp-col": "target_shp_col",
        "target-feature": "target_feature",
    },
}


def get_kwargs_map(name):
    for key in _kwargs_map.keys():
        if not name.startswith(key):
            continue

        return _kwargs_map[key]
    raise ValueError(f"no kwargs map for {name}")


def get_kwargs(name, **opts):
    kwargs_map = get_kwargs_map(name)

    # check about kwargs map
    msg = ""
    for key in kwargs_map.keys():
        if key in opts.keys():
            continue
        msg += f"'{key}' is missing for {name}\n"

    if len(msg) > 0:
        raise ValueError(msg)

    kwargs = {}
    for key, value in kwargs_map.items():
        kwargs[value] = opts[key]

    return kwargs


def list_models():
    return list(_model_factory.keys())


def get_model_loader(name):
    if name not in list_models():
        msg = "'%s' do not support. please check other models.\n" % name
        msg += ", ".join(list_models())
        raise RuntimeError(msg)

    return _model_factory[name]


def get_dino_backbone(**opts):

    dino_model = opts.get("dino-model", "dinov2_vits14")
    freeze_dino = opts.get("freeze-dino", True)
    unfreeze_dino_last_n_layer = opts.get("unfreeze-dino-last-n-layer", 0)

    backbone = get_dino(
        dino_model,
        freeze=freeze_dino,
        unfreeze_last_n_layers=unfreeze_dino_last_n_layer,
    )

    return backbone


def get_model(**opts):

    if "name" not in opts:
        raise ValueError(f"no model for {name}")
    name = opts.pop("name")

    kwargs = get_kwargs(name, **opts)

    target_shp_row = kwargs.pop("target_shp_row")
    target_shp_col = kwargs.pop("target_shp_col")
    kwargs["target_shp"] = (target_shp_row, target_shp_col)

    loader = get_model_loader(name)

    if "dino" in name:
        backbone = get_dino_backbone(**opts)

    elif "resnet18" in name:
        backbone = ResNet("resnet18")

    else:
        raise ValueError(f"no backbone loader for {name}")

    return loader(backbone, **kwargs)
