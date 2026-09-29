<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3-Nano Action-Policy Fine-Tuning (SFT)— Kunlun P800

Supervised fine-tuning of the Cosmos3 action models on a **single 8-GPU Kunlun P800 (XPU) node**.
This is the P800 counterpart of the
[NVIDIA action fine-tuning cookbook](../../../../../cookbooks/cosmos3/generator/action/finetune/README.md):
same `cosmos_framework.scripts.train` flow and recipe TOMLs, plus the P800 runtime switches and the
single-node parallelism baked into each launcher.

| Recipe | Launch shell | Task | Dataset |
| --- | --- | --- | --- |
| DROID action policy | `launch_sft_action_policy_droid_nano.sh` | action policy (`joint_pos`) | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) (success) |
| DROID forward dynamics | `launch_sft_action_fd_droid_posttrain.sh` | forward dynamics | [Cosmos3-DROID](https://huggingface.co/datasets/nvidia/Cosmos3-DROID) (success + failure) |
| LIBERO-10 action policy | `launch_sft_action_policy_libero_10_nano.sh` | action policy | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) (`libero_10`) |
| LIBERO-all action policy | `launch_sft_action_policy_libero_all_nano.sh` | action policy (4-suite mix) | [LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) (all 4 suites) |

Each launcher is a complete recipe: it stages the small inputs (Wan2.2 VAE, base DCP checkpoint,
Qwen3-VL tokenizer, and — for DROID action policy — the keep-ranges filter), then runs 8-GPU
training. Every download/convert step is idempotent — skipped if its output already exists.

## Prerequisites

1. **Set up the Kunlun-branch framework and the P800 environment** — follow the shared
   [Cosmos3 XPU environment setup](../../../README.md#cosmos-framework-kunlun-branch) (Kunlun
   `kunlun_dev` branch into `packages/cosmos3`, host `torch`/`torch_xmlir` stack, `LD_LIBRARY_PATH`,
   venv). The P800 fused kernels only exist on that branch.
2. **Hugging Face access.** Some model/dataset repos are license-gated — accept the terms and
   authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`).
3. **Run from this directory** (`cookbooks-xpu/cosmos3/generator/action/finetune/`). Downloads,
   converted checkpoints, and run outputs default to `data/`, `checkpoints/`, and `outputs/` here
   (all git-ignored).

## Quick start

From this folder, with the framework venv active, pick a recipe and run it. LIBERO auto-downloads its
suites; DROID is TB-scale and must be staged first (see below).

```shell
# LIBERO-10 action policy — auto-downloads the libero_10 suite
bash launch_sft_action_policy_libero_10_nano.sh

# LIBERO-all action policy (4-suite 1:1:1:1 mix) — auto-downloads all 4 suites
bash launch_sft_action_policy_libero_all_nano.sh

# DROID action policy — stage the DROID dataset first (see below)
bash launch_sft_action_policy_droid_nano.sh

# DROID forward dynamics — stage DROID success/ AND failure/ first (see below)
bash launch_sft_action_fd_droid_posttrain.sh
```

The P800 switches (default-on) and the single-node shape are set inside each launcher — see the
[P800 runtime switches](../../../README.md#p800-runtime-switches) reference for the full table and how
to toggle any one.

Run a short smoke test to validate the pipeline end-to-end:

```shell
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10" \
    bash launch_sft_action_policy_libero_10_nano.sh
```

## Staging the DROID dataset

The two DROID recipes do **not** auto-download (the dataset is TB-scale); they check for a staged copy
and fail with instructions if it is missing. The parent directory **must** be named
`droid_plus_lerobot_640x360_20260412` — the loader resolves the dataset schema from that name. Action
policy needs `success/`; forward dynamics needs `success/` **and** `failure/`. Both recipes default to
`data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412` under this folder; stage it with:

```shell
uvx hf@latest download nvidia/Cosmos3-DROID --repo-type dataset \
    --local-dir data/Cosmos3-DROID/droid_plus_lerobot_640x360_20260412
```

To stage it elsewhere, point the action-policy recipe at it with `DROID_ROOT=/path/to/...` and the
forward-dynamics recipe with `DATASET_PATH=/path/to/...` (each is the parent dir, not the split).

## Relocating inputs and outputs

Each launcher anchors its paths under this folder by default. Point any of them elsewhere by exporting
the matching env var before the run (all accept absolute or folder-relative paths):

| Env var | Meaning | Used by |
| --- | --- | --- |
| `DROID_ROOT` | DROID action-policy parent dir (holds `success/`) | DROID action policy |
| `DATASET_PATH` | DROID forward-dynamics parent dir (holds `success/` + `failure/`) | DROID FD |
| `FILTER_PATH` | DROID action-policy `keep_ranges_1_0_1.json` window filter (auto-staged) | DROID action policy |
| `LIBERO_ROOT` | libero_10 suite dir, or the parent of all 4 suites | LIBERO recipes |
| `BASE_CHECKPOINT_PATH` | base DCP checkpoint dir (converted from `Cosmos3-Nano`) | all |
| `WAN_VAE_PATH` | Wan2.2 VAE `.pth` | all |
| `TOKENIZER_PATH` | local `Qwen3-VL-8B-Instruct` dir | all |
| `OUTPUT_ROOT` | run outputs + logs (default `outputs/train`) | all |
| `NPROC_PER_NODE` | `torchrun --nproc_per_node` (default 8) | all |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` | multi-node torchrun topology | all |
| `EXTRA_TAIL_OVERRIDES` | space-separated extra Hydra overrides appended to the run | all |

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state);
  `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, and the full console log under `outputs/train/logs/`.

Re-running a launcher against the same `OUTPUT_ROOT` **auto-resumes** from the latest DCP.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

## Advanced configuration

For the full post-training reference — raw `torchrun`, resuming, every TOML field, closed-loop
evaluation, embodiment coverage — see the upstream
[action fine-tuning cookbook](../../../../../cookbooks/cosmos3/generator/action/finetune/README.md) and
the canonical [cosmos-framework docs](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md).
The only P800-specific additions are the runtime switches and the single-node shape documented above.
