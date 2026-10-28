<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos XPU Cookbook

Kunlun P800 (XPU) environment setup, launch scripts and training configurations
are maintained in [Cosmos Framework's Kunlun branch][framework]. This page is the
entry point; the detailed recipes live in Cosmos Framework, following the same
separation as the upstream [Cosmos Cookbook](COOKBOOK.md).
Start with the [XPU environment setup][setup].

## Recipes

The documented topology is a single node with eight P800 devices. See the
[vision guide][vision] or [action guide][action] for data preparation, runtime
switches, path overrides, smoke-test commands, outputs and checkpoint export.

| Recipe | Base model | Industry | Device budget |
| --- | --- | :-: | :-: |
| [Nano vision SFT][vision-nano] | Cosmos3-Nano | `ANY` | `M` |
| [Edge vision SFT][vision-edge] | Cosmos3-Edge | `ANY` | `M` |
| [Super vision LoRA][vision-super] | Cosmos3-Super | `ANY` | `M` |
| [Nano map-style vision SFT][vision-mapstyle] | Cosmos3-Nano | `ANY` | `M` |
| [DROID policy][droid-policy] | Cosmos3-Nano | `ROB` | `M` |
| [DROID forward dynamics][droid-fd] | Cosmos3-Nano | `ROB` | `M` |
| [LIBERO-10 policy][libero-10] | Cosmos3-Nano | `ROB` | `M` |
| [LIBERO-all policy][libero-all] | Cosmos3-Nano | `ROB` | `M` |

`ANY`: any industry. `ROB`: robotics. `M`: 2–8 P800 devices; these recipes
use eight devices on one node.

[framework]: https://github.com/KunlunxinAD/cosmos-framework/tree/kunlun_dev
[setup]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/.agents/skills/cookbooks-xpu/cosmos3/README.md
[vision]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/.agents/skills/cookbooks-xpu/cosmos3/generator/audiovisual/finetune/README.md
[action]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/.agents/skills/cookbooks-xpu/cosmos3/generator/action/finetune/README.md
[vision-nano]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_vision_nano_p800.sh
[vision-edge]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_vision_edge_p800.sh
[vision-super]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/.agents/skills/cookbooks-xpu/cosmos3/generator/audiovisual/finetune/launch_sft_vision_super.sh
[vision-mapstyle]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_vision_nano_mapstyle_dataloader_p800.sh
[droid-policy]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_action_policy_droid_nano_p800.sh
[droid-fd]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_action_fd_droid_posttrain_p800.sh
[libero-10]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_action_policy_libero_10_nano_p800.sh
[libero-all]: https://github.com/KunlunxinAD/cosmos-framework/blob/kunlun_dev/examples/launch_sft_action_policy_libero_all_nano_p800.sh
