# 远程环境配置指南（GPU/CPU）

## 当前镜像选择 PyTorch 2.3.0
- Ubuntu 22.04
- Python 3.12
- CUDA 12.1
- PyTorch 2.3.x + TorchVision 0.18.x（镜像：PyTorch 2.3, CUDA 12.1）

## 目标
- 给出与设备无关的依赖清单：见 `requirements2.txt`
- 按设备（GPU/CPU）正确安装 PyTorch 与 TorchVision，并说明注意事项

## 适用范围
- 系统：Ubuntu 22.04（你选择的镜像）
- Python：3.12（镜像自带；本仓库依赖已调整以兼容 3.12）

## 准备工作
- 如使用 Ubuntu，请安装 OpenCV 运行时依赖：
  - `sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0`
- 建议使用 Conda 管理环境（Miniconda/Miniforge 均可）

## 创建与激活环境
- 可直接使用镜像自带的 Python 3.12 搭配 venv；或使用 Conda：
```bash
conda create -n charrec python=3.12 -y
conda activate charrec
```

## 安装 PyTorch（与你的镜像匹配）
> 说明：已选择 CUDA 12.1 + PyTorch 2.3。确保驱动与镜像匹配（`nvidia-smi` 可查看）。

- GPU（CUDA 12.1，对应 PyTorch 2.3.x）：
```bash
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

- CPU（无 GPU，仅作备选）：
```bash
pip install torch==2.3.1+cpu torchvision==0.18.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

- 验证安装：
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 安装与设备无关的依赖
```bash
pip install -r requirements2.txt
```

内容包含：`numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `Pillow`（均为设备无关，已为 Python 3.12 调整版本）。

## 数据与路径
- 将数据集解压至 `data/` 目录
- 检查并修改 `config.py` 中的数据路径与类别数配置
- 建议使用英文路径（避免中文/空格在某些远程环境引发 I/O 问题）

## 快速冒烟测试
在小 batch/短 epoch 下验证环境 OK：
```bash
python train.py --model LeNet --epochs 1 --batch_size 32
```
如需测试推理：
```bash
python test.py --model ResNet
```

## 常见问题与提示
- OpenCV 报错（GL/GTK 相关）通常是系统库缺失，执行上面的 `apt-get` 安装
- Torch 与 TorchVision 需要匹配：`torch 2.3.x ↔ torchvision 0.18.x`
- 远程镜像自带 CUDA 的情况下，优先使用与镜像说明匹配的下载索引（本镜像为 cu121）

## 关于 d2l 依赖
已移除项目中的 d2l 导入，并从 `requirements2.txt` 删除该依赖，避免与 Python 3.12 的 `numpy` 版本冲突。

## 运行建议（训练）
示例（GPU）：
```bash
python train.py --model ResNet --epochs 50 --batch_size 256 --lr 1e-3
```
请根据显存大小调整 `--batch_size`（8–24GB 显存可尝试 128–512）。
