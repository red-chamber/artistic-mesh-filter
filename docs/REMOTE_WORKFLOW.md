# Remote workflow

## 1. Prepare environment

远端拉取与本机相同的 Git commit，创建独立 Python 环境，然后安装：

```bash
python -m pip install -e ".[train]"
```

推荐保存以下信息：

```bash
python --version
python -m pip freeze
git rev-parse HEAD
```

## 2. Configure paths

复制 `configs/paths.example.yaml` 为 `configs/paths.remote.yaml`，只修改真实路径、格式白名单和运行时限制。该文件不进入 Git。

## 3. Smoke test

首次运行时设置：

```yaml
runtime:
  limit_per_source: 20
```

按顺序执行：

```bash
python -m mesh_filter build-manifest --config configs/paths.remote.yaml
python -m mesh_filter validate --config configs/paths.remote.yaml
python -m mesh_filter deduplicate --config configs/paths.remote.yaml
python -m mesh_filter split --config configs/paths.remote.yaml
```

检查：

- 两个来源是否都发现文件；
- 解析失败是否有明确 `error_code`；
- 顶点数和面数是否合理；
- 一个文件失败是否不会中断整个批次；
- 输出是否写入预期目录；
- 峰值内存和单模型耗时是否可接受。

## 4. Full data pass

将 `limit_per_source` 设为 `null`，删除或更换冒烟测试输出路径，再执行完整流程。不要把冒烟测试 manifest 与正式 manifest 混用。

## 5. Training

计数基线只用于回答“面数/顶点数本身能达到什么效果”：

```bash
python -m mesh_filter train-count-baseline \
  --config configs/paths.remote.yaml \
  --experiment configs/experiments/count_baseline.yaml
```

训练完成后将小型结果传回本机：manifest、features、splits、`metrics.json`、`predictions.csv` 和少量错误案例。大模型文件和全量渲染可以留在远端。

## 6. Production cautions

- 正式特征提取必须在同一远端环境完成；
- 不同 Blender/Assimp/OpenGL 版本可能改变导入或渲染结果；
- 多进程、GPU、混合精度和 batch size 的实际值必须记录；
- 在确认真实模型格式前，不启用自动修复、合并顶点或覆盖式三角化。
