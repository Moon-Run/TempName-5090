# RTX 5090：复用已有环境，配置 CUTLASS

在新的服务器上进入 `TempName-5090`，使用已有的 Python、CUDA PyTorch 和 CUDA Toolkit：

```bash
bash env/setup_5090.sh
source env/activate_5090.sh
bash tools/5090/run_init.sh --trials 1 --blocks 1 --execute
```

配置脚本会检查现有软件、准备固定版本的 CUTLASS，并做一次不启动 GPU 的 SM120 小编译检查。它不创建 venv，不执行 pip/apt，不安装或升级 Python、PyTorch、CUDA，也不依赖 Megatron、Apex 或 H100 配置脚本。最后一条命令才启动四组实验，先做算子检查，再测 GPT 6.7B / TP8 完整训练步。

实验入口默认后台运行，结果写入仓库同级 `Data-5090/SL3061/`。使用 `--place 60` 切换到 `SL3060`，使用 `--foreground` 在当前终端等待。配置脚本自身及其可选 `--gpu-check` 仍在前台执行，以确认检查完成后再发布环境配置。

## CUTLASS 如何保存和恢复

Git 保留 [`third_party/cutlass.lock.json`](../third_party/cutlass.lock.json)，其中固定 CUTLASS 3.9.0 提交 `df8a550d3917b0e97f416b2ed8c2d786f7f686a3` 和 835 个文件的 SHA256。实际的 `third_party/cutlass/` 由 `.gitignore` 忽略，在目标服务器配置时恢复。

- 目录不存在时，从锁文件指定的官方 Git 仓库获取固定提交，只复制锁定的头文件和许可证；逐文件校验成功后才发布目录。
- 目录已存在时，校验后直接复用，不重复下载。内容不同或缺文件会报错并保留原目录，不覆盖用户修改。
- 可以从已有 CUTLASS 目录离线复制。只要锁定文件的内容完全相同即可，不要求源目录必须是 Git checkout。

此处改变的是依赖的配置方式，没有升级 CUTLASS 版本。编译和实验报告仍以 Git 中的外部锁为准；每批实验的源码快照继续包含实际用到的 CUTLASS 头文件和锁文件，所以结果复核不依赖未来重新下载。

## 常用参数

```bash
# 只看配置计划；不下载、不写文件、不查询GPU。
bash env/setup_5090.sh --dry-run

# 使用指定的已有解释器和Toolkit。
bash env/setup_5090.sh \
  --base-python /path/to/existing/python \
  --cuda-home /usr/local/cuda

# 网络受限时，从已有源码目录恢复锁定文件。
bash env/setup_5090.sh --cutlass-source /path/to/existing/cutlass

# 只准备CUTLASS；不检查Torch/CUDA，不生成可用环境声明。
bash env/setup_5090.sh --dependencies-only

# 可选：配置后立即跑八卡预检，不运行训练。
bash env/setup_5090.sh --gpu-check
```

`--python` 是 `--base-python` 的别名；默认保留运行脚本所用的当前解释器，因此已激活的 venv/conda 环境可以直接复用。脚本入口解释器也可用 `RTX5090_BASE_PYTHON` 指定。CUDA 路径可通过 `--cuda-home`、已有 `CUDA_HOME` / `CUDA_PATH` 或发现的 `nvcc` 确定；`--nvcc` 可显式指定编译器。

运行实验需要 Python 3.10+、CUDA PyTorch、支持 SM120 的 CUDA 12.8+ Toolkit，以及默认八张 RTX 5090。系统需已有 Git 和可用的主机编译器；缺失项会提示，不自动安装。配置通过后执行 `source env/activate_5090.sh`，将验证过的解释器和 CUDA 路径提供给实验入口。

默认配置不使用 GPU，`--skip-gpu-check` 保留为兼容参数，与默认的软件配置模式相同。软件配置成功不代表 GPU/NCCL 数值检查或性能测试通过；`run_init.sh --execute` 会执行自己的预检。

## 记录和复核

配置记录位于：

```text
outputs/5090/env-setup/runs/<时间>/setup.log
outputs/5090/env-setup/runs/<时间>/report.json
outputs/5090/env-setup/last-report.json
outputs/5090/env-setup/env.sh
```

`env.sh` 仅在软件配置成功后生成；`--dependencies-only` 只准备依赖，不发布激活配置。失败时查看本次 `report.json` 与日志。该输出目录也被 Git 忽略，迁移服务器后应重新配置，避免沿用另一台机器的绝对路径。

配置流程的 CPU 回归测试：

```bash
python3 -B -m unittest discover -s env -p 'test_setup_5090.py' -v
```

完整实验和报告说明见 [设计文档](../docs/design-5090/init.md)。
