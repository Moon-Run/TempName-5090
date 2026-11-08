# 全新 H100 容器一键配置

RTX 5090 / NGC PyTorch 25.06 使用独立的 [setup_5090.sh 配置说明](README_5090.md)，沿用镜像自带的 Python/PyTorch/CUDA/Apex。下文为 H100 专用流程。

适用环境：`nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`，Ubuntu22.04 x86_64，容器内 root，已通过 `--gpus all --ipc=host --ulimit memlock=-1` 暴露八卡。按当前挂载布局：

```text
/workspace/
├── TempName-New/
└── Megatron-LM/
```

直接执行：

```bash
cd /workspace/TempName-New
bash env/setup_h100.sh
```

无需手工创建 Conda 环境，也不需要另外添加 `--execute`。脚本完成环境配置与验证，**不会启动正式训练或性能测试**。首次需要访问 Ubuntu／NVIDIA 软件源、GitHub、conda-forge、PyPI 和 PyTorch 下载站；安装与 Apex 编译可能耗时较长。

## 安装的内容

引导阶段先安装系统 Python 和 `tzdata`，以支持生成 `Asia/Shanghai` 时区的日志时间。若旧版脚本报 `ZoneInfoNotFoundError`，在容器中执行 `apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y tzdata` 后重跑即可；该报错发生在训练环境安装之前，不能据此执行激活或开始训练。

| 项目 | 版本／路径 |
| --- | --- |
| Miniforge 引导程序 | 固定 `25.3.1-0`，下载后检查固定 SHA256；放在 `/workspace/.h100-env/miniforge` |
| 独立环境 | **`/workspace/conda-envs/tempname-h100`** |
| Python | **3.10.20** |
| PyTorch | **2.6.0+cu124**，检查 C++ ABI=0 |
| CUDA Toolkit | 补齐 **12.4**；复用镜像的 **12.8**，缺失时安装对应 Toolkit |
| Apex | 固定提交 `e13873debc4699d39c6861074b9a3b2a02327f92`，用 CUDA12.4 编译 `apex_C`、`amp_C`、`fused_layer_norm_cuda` |
| Megatron-LM | 固定提交 `3ea68ad6042cc1204386ae9364358f7c4de1bc37` |
| 构建工具 | GCC/G++11、CMake3.30.0、Ninja1.11.1.3、Make、Git 等 |
| 基础 Python 依赖 | 沿用 `restore_container.py` 的固定版本，包括 NumPy1.26.4 |
| 通信用户态依赖 | `libibverbs1`、`ibverbs-providers`、`ibverbs-utils`、NUMA 库 |

CUDA12.4 与 12.8 分别保留在版本目录，不依赖可能被安装器重新指向的 `/usr/local/cuda`。只安装 Toolkit 和用户态依赖，不安装或替换宿主机 NVIDIA 驱动。Flux／TACO 本体由测试入口在运行时编译。

已有 `/workspace/Megatron-LM` 时，脚本先检查是否为独立且没有受版本控制文件改动的 Git 仓库。干净但提交不匹配的仓库会获取并切换到上述固定提交；有修改或非 Git 目录则停止并保留内容，不使用 `reset --hard`、`git clean` 或删除目录。此检查在下载 PyTorch、编译 Apex之前完成。CUTLASS／NCCL 按当前仓库 Git 子模块条目初始化并验证。

## 成功后使用

```bash
source /workspace/TempName-New/env/activate.sh
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

脚本生成 `outputs/h100/container-recovery/env.sh`，已有 `run_ma*.sh` 入口会自动读取正确的 Python、编译器和 CUDA 路径。因此环境配置成功后，可按需要另行运行：

```bash
cd /workspace/TempName-New
bash tools/h100/phase6/run_ma0_layer.sh --execute
```

默认验证 PyTorch／NumPy版本、三项 Apex 扩展导入、CUDA12.8 SM90 编译、八张132-SM H100及两两 peer access，最后验证真实 Megatron 训练模块及 libibverbs 导入。这里的 GPU 检查不是正式训练、通信性能基准或 MA0 数值验收。

## 日志、重复执行与参数

脚本会显示当前阶段。引导过程及详细安装日志分别位于：

```text
outputs/h100/env-setup/<时间>/install.log
outputs/h100/env-setup/last-report.json
outputs/h100/container-recovery/runs/<时间>/install.log
outputs/h100/container-recovery/last-report.json
```

以 `env-setup/last-report.json` 的 `status=completed` 为整个配置流程完成的依据。失败保留日志，修复网络或前置条件后可以重跑相同命令。已有且由此脚本管理的 Python 环境会复用并恢复固定依赖；无管理记录的同名环境不会被覆盖。安装中断留下的不完整 Miniforge／Python 目录会移到本次日志目录保存后重建。

Python 环境、下载缓存位于挂载的 `/workspace`，重建容器后可继续复用；APT 安装的 Toolkit 和系统库属于容器文件系统，重建容器后应重新执行脚本检查。不要在不同容器中同时修改同一个环境。

可选命令：

```bash
# 只查看计划，不安装、不查询 GPU。
bash env/setup_h100.sh --dry-run

# 暂无 GPU 时先配置软件；报告会明确标记 GPU 未验证。
bash env/setup_h100.sh --skip-gpu-check

# 自定义路径，必须位于源码仓库以外。
bash env/setup_h100.sh \
  --prefix /workspace/conda-envs/tempname-h100 \
  --megatron /workspace/Megatron-LM \
  --state-dir /workspace/.h100-env
```

如果自定义 Megatron 路径，启动测试时也应显式传递 `--megatron`。需要断开终端时，可在容器里使用 `nohup bash env/setup_h100.sh > /workspace/h100-env-setup.log 2>&1 &`，再查看日志；配置脚本默认在前台运行。

## 验证与来源

开发机只执行无安装、无 GPU 的脚本回归，不能代替目标容器中的真实下载、Apex 编译及 GPU 验证：

```bash
python3 -B -m unittest discover -s env -p 'test_setup_h100.py' -v
python3 -B -m unittest discover -s tools/h100/phase6 -p 'test_restore_container.py' -v
```

2026-10-28 本地检查：新增引导测试 **14 项通过**，复用的环境恢复测试 **15 项通过**；Bash／Python 语法及无副作用预览通过。覆盖固定环境名、下载摘要不匹配时禁止执行、已有目录保护、干净 Megatron 固定版本切换、失败状态以及完整配置控制流程。

安装参数参考 [PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[固定 Apex 源码](https://github.com/NVIDIA/apex/tree/e13873debc4699d39c6861074b9a3b2a02327f92)、[Miniforge 固定版本及资产摘要](https://github.com/conda-forge/miniforge/releases/expanded_assets/25.3.1-0) 和 [CUDA12.4 安装指南](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html)。项目版本依据 [H100 部署说明](../docs/design/design-h100/deploy-tp8.md)。
