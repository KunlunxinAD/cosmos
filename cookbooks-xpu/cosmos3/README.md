<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 XPU Cookbooks: Environment Setup

Shared environment setup for every Cosmos3 cookbook adapted to **Kunlun P800 (XPU)**. These are the
P800 counterparts of the [NVIDIA Cosmos3 cookbooks](../../cookbooks/cosmos3/README.md): the same
`cosmos_framework.scripts.train` flow and the same recipe TOMLs, run on a single 8-GPU P800 node with
the P800 fused kernels enabled. Each recipe README links back here for
setup and for the P800 switch reference.

| Backend | Use it for | Used by |
| --- | --- | --- |
| [Cosmos Framework (Kunlun branch)](#cosmos-framework-kunlun-branch) | Native PyTorch training on P800, launched with `torchrun` | Generator (Audiovisual, Action) |

Currently validated on P800 (more recipes will be added here over time):

| Cookbook | Recipes | Where |
| --- | --- | --- |
| Audiovisual SFT | `vision_sft_nano` (T2V / I2V / V2V) | [`generator/audiovisual/finetune/`](generator/audiovisual/finetune/) |
| Action SFT | DROID policy, DROID forward dynamics, LIBERO-10, LIBERO-all | [`generator/action/finetune/`](generator/action/finetune/) |

## Prerequisites

- Linux host with **Kunlun P800 (XPU)** access (a single 8-GPU node for these recipes).
- **P800 container image.** These recipes are validated inside the Kunlun XAV image, which ships the
  matching P800 `torch` + `torch_xmlir` stack. Pull it with:

  ```shell
  docker pull hub.kunlunxin.com/public/kunlunxin-self-driving/xav:26.08-py312-torch29
  ```

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `git`, and `git-lfs` installed.
- Hugging Face access to the gated Cosmos3 model repos and datasets. Every model and dataset is
  pulled from its **official Hugging Face repo** (`nvidia/*`, `Wan-AI/*`, `Qwen/*`) — no mirror is
  required. Authenticate once before the first run:

  ```shell
  uvx hf@latest auth login
  # or:
  export HF_TOKEN=<your_token>
  ```

- Enough local disk for the venv, the base checkpoint, and the datasets (the DROID splits are
  TB-scale; the small inputs — VAE + tokenizer + DCP — are tens of GiB).

## Cosmos Framework (Kunlun branch)

The P800 fused kernels these recipes rely on live **only on the Kunlun
fork**, not upstream NVIDIA. Install
[`KunlunxinAD/cosmos-framework@kunlun_dev`](https://github.com/KunlunxinAD/cosmos-framework/tree/kunlun_dev)
into `packages/cosmos3`.

1. **Clone the Kunlun branch** into `packages/cosmos3`.
   ```shell
   git clone --branch kunlun_dev https://github.com/KunlunxinAD/cosmos-framework.git packages/cosmos3
   ```
2. **Keep the host P800 Torch/XPU stack.** The P800 container image above already ships a matching
   `torch` + `torch_xmlir` build. Install the framework's Python dependencies **without** the CUDA
   dependency groups — do **not** pull a `cu128`/`cu130` wheel, it will replace the working XPU Torch.
3. **Install the extra dependencies.** The base image does not ship the framework's Python
   requirements. Install `ffmpeg` and the Python packages for the recipe family you are running —
   the audiovisual and action sets pin a few packages differently, so use a **separate venv per
   family**. Training runs from the checkout via `PYTHONPATH` (the launch scripts set it for you).

   System package (both families):
   ```shell
   apt-get update && apt-get install -y ffmpeg
   ```

   **Audiovisual** (`vision_sft_nano`) — from the framework checkout in `packages/cosmos3`,
   install the package (without the CUDA groups) and then the audiovisual extras:
   ```shell
   python -m pip install . --upgrade-strategy only-if-needed
   python -m pip install \
     loguru==0.7.3 iopath==0.1.10 diffusers==0.39.0 hydra-core==1.3.5 omegaconf==2.3.1 \
     transformers==4.57.6 peft==0.20.0 boto3==1.43.75 botocore==1.43.75 cattrs==26.1.0 \
     ImageIO==2.37.4 imageio-ffmpeg==0.6.0 msgpack==1.2.1 obstore==0.11.0 \
     nvidia-cudnn-frontend==1.27.0
   python -m pip install --no-deps --force-reinstall wandb==0.27.2
   python -m pip install 'GitPython>=3.1.29'
   ```

   **Action** (DROID / LIBERO) — from the framework checkout in `packages/cosmos3`, install the
   package (without the CUDA groups) and then the action extras:
   ```shell
   python -m pip install . --upgrade-strategy only-if-needed
   python -m pip install \
     'transformers>=4.57.1,<5.0.0' 'peft>=0.17.0,<1.0' 'datasets==4.6.1' \
     'boto3==1.40.61' 'botocore==1.40.61' 'aiobotocore==2.25.1' 'aioboto3==15.5.0' \
     's3fs==2026.2.0' 'fsspec==2026.2.0' 'iopath>=0.1.10' 'tensorstore==0.1.81' \
     'wandb==0.27.2' 'nvdlfw-inspect' 'onnxscript' 'blobfile' 'deepdiff>=7.0.1,<9.0.0' \
     'einx' 'ftfy' 'fvcore' 'kornia' 'lz4' 'megatron-core' 'more-itertools' 'ninja' \
     'qwen-vl-utils' 'tiktoken' 'timm' 'torchtitan' 'typeguard' 'webdataset'
   # lerobot 0.4.4's metadata would downgrade the diffusers/torch stack; install it without deps.
   python -m pip install --no-deps 'lerobot==0.4.4'
   ```

   **FlashAttention-2** (both families) — the `flash2` attention backend (`I4_ATTN_BACKENDS`),
   built for the image's Python 3.12 + Torch 2.9:
   ```shell
   wget https://klx-sdk-release-public.su.bcebos.com/v1/XFlashAttention/release/1.0.5.1/xFlashAttention_py312_torch290.tar.gz
   tar -xvzf xFlashAttention_py312_torch290.tar.gz && cd xFlashAttention_py312_torch290
   pip install flash_attn-2.8.0+6ef99a2-cp312-cp312-linux_x86_64.whl
   ```

Each launch script then resolves the framework root automatically from `cosmos_framework.__file__`
(override with `COSMOS_FRAMEWORK_ROOT`), stages its inputs, and trains.

## P800 runtime switches

The framework-level fused-kernel switches default to the reference implementation (off). Every
launcher explicitly exports the P800 switches with the `${VAR:-1}` idiom, so the launcher default is
on while you can A/B any one by exporting it as `0` before the run — e.g.
`COSMOS_FUSED_RMSNORM=0 bash launch_sft_action_policy_libero_10_nano.sh`. The fused helpers probe
for their backend and fall back to the reference path when it is unavailable.

### Fused kernels (all recipes)

| Switch | Default | Effect |
| --- | --- | --- |
| `XDNN_USE_FAST_SWISH` | `1` | XDNN fast SwiSH/SiLU activation |
| `COSMOS_FUSED_RMSNORM` | `1` | fused RMSNorm |
| `COSMOS_FUSED_ROPE` | `1` | fused rotary position embedding |
| `COSMOS_FUSED_CAUSAL_CONV_PAD` | `1` | fused causal-conv padding |
| `COSMOS_FUSED_VAE_NORM` | `1` | fused VAE normalization |

### Runtime / determinism (all recipes)

| Switch | Default | Effect |
| --- | --- | --- |
| `COSMOS_TRAINING` | `1` | select the training runtime path |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | the 8 P800 devices to use |
| `PYTHONHASHSEED` | `42` | reproducible hashing (paired with `trainer.seed=42`) |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | allocator config (audiovisual + DROID action policy) |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` | force offline once inputs are staged |

Each launch script also pins the **single-node P800 shape** through Hydra tail overrides —
`torch.compile` disabled, `data_parallel_shard_degree=8` / `data_parallel_replicate_degree=1`
(overriding the TOML's multi-node HSDP shape), selective activation checkpointing, and
`tokenizer.dtype=float16`. These are part of the recipe; you do not need to set them.
