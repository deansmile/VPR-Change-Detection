# Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms



### Installation

```bash
# clone main repo and corresponding submodule
$ git clone https://github.com/deansmile/VPR-Change-Detection.git
$ cd <this repository>
$ git submodule init
$ git submodule update

# create a Python 3.9.6 virtual environment
$ source <directory of virtual environment>/bin/activate
$ cd <this repository>
$ pip install -r requirements.txt
```

### Datasets

* download datasets from [huggingface](https://huggingface.co/ai4ce/vpr_change_detection/tree/main)

* update the both dataset directories to `datasets/data_factory`

### ICRA 2025 Paper

* [arXiv link](http://arxiv.org/abs/2409.16850)

### Example usage

* unittest
    ``` bash
    $ cd <this repository>/src/unittest
    $ python -m unittest
    ```

* training
    ```bash
    # modify the configuration in scripts/configs/train.yml
    $ python <this repository>/src/scripts/train.py \
        <this repository>/src/scripts/configs/train.yml
    ```

* fine-tune
    ```bash
    # modify the configuration in scripts/configs/fine_tune.yml
    $ python <this repository>/src/scripts/fine_tune.py \
        <this repository>/src/scripts/configs/fine_tune.yml
    ```

* evaluation
    ```bash
    $ python <this repository>/src/scripts/evaluate.py \
        <checkpoint directory>/<name>.pth
    ```

* qualitive results
    ```bash
    $ python <this repository>/scripts/visualize.py \
        <checkpoint directory>/best.val.pth \
        --option <option> \
        --output <directory for qualitive results>
    ```

    | options           | comments                            |
    | ----------------- | ----------------------------------- |
    | VL-CMU-CD         | aligned                             |
    | PSCD              | aligned                             |
    | VL-CMU-CD-diff_1  | unaligned (adjacent distance == 1)  |
    | VL-CMU-CD-diff_-1 | unaligned (adjacent distance == -1) |
    | VL-CMU-CD-diff_2  | unaligned (adjacent distance == 2)  |
    | VL-CMU-CD-diff_-2 | unaligned (adjacent distance == -2) |

### Pretrained Weight

* Train on VL-CMU-CD

| name             | train on VL-CMU-CD    | train on diff VL-CMU-CD   | fine-tune on PSCD   |
| ---------------- | :-------------------: | :-----------------------: | :-----------------: |
| ours (DinoV2)    | [dinov2.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) | [dinov2.2CrossAttn.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.Diff-CMU.pth) | [dinov2.2CrossAttn.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.PSCD.pth) |
| ours (Resnet-18) | [resnet18.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) | / | [resnet18.2CrossAttn.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.PSCD.pth) |
| [C-3PO](https://github.com/DoctorKey/C-3PO) | [resnet18_id_4_deeplabv3_VL_CMU_CD](https://github.com/DoctorKey/C-3PO) | [baseline.c3po.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.c3po.Diff-CMU.pth) | [baseline.c3po.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.c3po.PSCD.pth) |
| [DR-TANet](https://github.com/Herrccc/DR-TANet) | [baseline.drtanet.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.CMU.pth) | [baseline.drtanet.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.Diff-CMU.pth) | [baseline.drtanet.PSCD](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.drtanet.PSCD.pth) |
| [CDNet](https://github.com/kensakurada/sscdnet) | [baseline.cdnet.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.cdnet.CMU.pth) | [baseline.cdnet.Diff-CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/baseline.cdnet.Diff-CMU.pth) | / |
| [TransCD](https://github.com/wangle53/TransCD) | [VL-CMU-CD -> Res-SViT_E1_D1_16.pth](https://github.com/wangle53/TransCD) | / | / |

* backbone v.s. comparator

| backbone  | comparator         | train on VL-CMU-CD |
| --------- | ------------------ | ------------------ |
| DinoV2    | Co-Attention       | [dinov2.CoAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.CoAttn.CMU.pth) |
| DinoV2    | Temporal Attention | [dinov2.TemporalAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.TemporalAttn.CMU.pth) |
| DinoV2    | MTF                | [dinov2.MTF.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.MTF.CMU.pth) |
| DinoV2    | 1 CrossAttn        | [dinov2.1CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.1CrossAttn.CMU.pth) |
| DinoV2    | 2 CrossAttn        | [dinov2.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/dinov2.2CrossAttn.CMU.pth) |
| Resnet-18 | 2 CrossAttn        | [resnet18.2CrossAttn.CMU](https://github.com/ChadLin9596/Robust-Scene-Change-Detection/releases/download/v0.0.0/resnet18.2CrossAttn.CMU.pth) |
