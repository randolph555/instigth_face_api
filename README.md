# 人脸识别API服务

基于InsightFace和Milvus的人脸识别服务，提供人脸注册、搜索、比较和属性分析功能。

## 功能特性

- 🔍 人脸检测和特征提取
- 📊 人脸属性分析（年龄、性别等）
- 🔄 人脸比较和验证
- 🗄️ 向量数据库存储和检索
- 🚀 高性能GPU加速
- 📝 完整的API文档

## 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- Milvus 2.3+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```
![alt text](image.png)


## 注意点
```
注意安装pip install要确认是在conda环境里
export PATH="/opt/conda/envs/face-recognition/bin:$PATH"
which python


conda activate common_ljk
conda remove --name face-recognition --all -y 
conda create -n face-recognition python=3.10 -y
conda activate face-recognition
pip install -r requirements.txt


pip install nvidia-cublas-cu11

如果要使用cude的话需要安装
conda install -c nvidia cudnn=8.9.2

cuda
export LD_LIBRARY_PATH=/opt/conda/envs/face-recognition/lib:$LD_LIBRARY_PATH

一劳永逸
echo 'export LD_LIBRARY_PATH=/opt/conda/envs/face-recognition/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```