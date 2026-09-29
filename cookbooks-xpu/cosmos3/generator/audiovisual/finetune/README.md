<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 Vision Generator Fine-Tuning (SFT) — Kunlun P800

Supervised fine-tuning of the Cosmos3 video generator on captioned video data, on a **single 8-GPU
Kunlun P800 (XPU) node**. This is the P800 counterpart of the
[NVIDIA audiovisual fine-tuning cookbook](../../../../../cookbooks/cosmos3/generator/audiovisual/finetune/README.md):
same `cosmos_framework.scripts.train` flow and TOML, plus the P800 runtime switches and single-node
FSDP shape baked into the launcher.

| Recipe | Launch shell | Base model | Dataset |
| --- | --- | --- | --- |
| Vision SFT (full)  | `launch_sft_vision_nano.sh` | Cosmos3-Nano | [BridgeData2-Subset-Synthetic-Captions](https://huggingface.co/datasets/nvidia/BridgeData2-Subset-Synthetic-Captions) |

The recipe trains on structured-JSON captions (`caption_json`, the model's native prompt format), so
training stays aligned with inference.

## Prerequisites

1. **Set up the Kunlun-branch framework and the P800 environment** — follow the shared
   [Cosmos3 XPU environment setup](../../../README.md#cosmos-framework-kunlun-branch) (Kunlun
   `kunlun_dev` branch into `packages/cosmos3`, preserving the host `torch`/`torch_xmlir` stack).
   The P800 kernels only exist on that branch.
2. **Hugging Face access.** The dataset is license-gated — accept the terms on the dataset page and
   authenticate once with `uvx hf@latest auth login` (or export `HF_TOKEN`).
3. **Run from this directory** (`cookbooks-xpu/cosmos3/generator/audiovisual/finetune/`). Downloads,
   converted checkpoints, and run outputs default to `data/`, `checkpoints/`, and `outputs/` here
   (all git-ignored).

## Quick start

The launcher is a complete recipe — run it from this folder and it downloads the dataset, fetches the
Wan2.2 VAE, stages the Qwen3-VL tokenizer, converts the base checkpoint, then runs 8-GPU FSDP
training (the download/convert steps are skipped if their outputs already exist):

```shell
bash launch_sft_vision_nano.sh
```

Paths are anchored under this git-ignored folder. Relocate any of them by exporting the matching env
var before the run (`DATASET_PATH`, `BASE_CHECKPOINT_PATH`, `WAN_VAE_PATH`, `TOKENIZER_PATH`,
`OUTPUT_ROOT`); `NPROC_PER_NODE` / `NNODES` / `NODE_RANK` / `MASTER_ADDR` map to `torchrun`, and
`EXTRA_TAIL_OVERRIDES` appends extra Hydra overrides.

The P800 switches (default-on) and the single-node shape are set inside the launcher — see the
[P800 runtime switches](../../../README.md#p800-runtime-switches) reference for the full table and how
to toggle any one.

Run a short smoke test to validate the pipeline end-to-end:

```shell
EXTRA_TAIL_OVERRIDES="trainer.max_iter=10 checkpoint.save_iter=10" bash launch_sft_vision_nano.sh
```

## Outputs

Training writes to `outputs/train/<project>/<group>/<name>/`:

- `checkpoints/iter_<N>/` — DCP checkpoint (model / optim / scheduler / trainer state);
  `checkpoints/latest_checkpoint.txt` names the newest.
- `config.yaml`, launch metadata, and the full console log under `outputs/train/logs/`.

Re-running the launcher against the same `OUTPUT_ROOT` **auto-resumes** from the latest DCP.

## Export to Hugging Face safetensors

```shell
RUN_DIR=outputs/train/<project>/<group>/<name>
CKPT=$RUN_DIR/checkpoints/$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt")
python -m cosmos_framework.scripts.export_model \
    --checkpoint-path "$CKPT" --config-file "$RUN_DIR/config.yaml" -o "$RUN_DIR/model"
```

## Advanced configuration

For the full post-training reference — raw `torchrun`, resuming, every TOML field, LoRA / EMA knobs —
see the upstream [audiovisual fine-tuning cookbook](../../../../../cookbooks/cosmos3/generator/audiovisual/finetune/README.md)
and the canonical [cosmos-framework docs](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md).
The only P800-specific additions are the runtime switches and the single-node shape documented above.
