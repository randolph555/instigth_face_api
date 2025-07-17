# 人脸识别API服务

基于InsightFace和Milvus的人脸识别服务，提供人脸注册、搜索、比较和属性分析功能。

## 功能特性

- 🔍 人脸检测和特征提取
- 📊 人脸属性分析（年龄、性别等）
- 🔄 人脸比较和验证
- 🗄️ 向量数据库存储和检索
- 🚀 高性能GPU加速
- 📝 完整的API文档

<img width="996" height="806" alt="image" src="https://github.com/user-attachments/assets/8a80f47f-3c3d-4680-a040-a2c74f292501" />


## 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- Milvus 2.5+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 模型文件下载地址
人脸识别相关：
https://github.com/deepinsight/insightface/releases
<img width="1368" height="676" alt="image" src="https://github.com/user-attachments/assets/f3f06b68-4675-4907-9c5a-ca2810689059" />

换脸：
https://drive.google.com/file/d/1krOLgjW2tAPaqV-Bw4YALz0xT5zlb5HF/view
或者
https://huggingface.co/ezioruan/inswapper_128.onnx/tree/main

模型位置：
<img width="345" height="406" alt="image" src="https://github.com/user-attachments/assets/49f68f49-c732-42e6-944b-64a91a4beec6" />

milvus文档参考：https://milvus.io/docs/zh/overview.md

## 注意点
```
注意安装pip install要确认是在conda环境里
export PATH="/opt/conda/envs/face-recognition/bin:$PATH"
which python



conda create -n face-recognition python=3.10 -y
conda activate face-recognition
pip install -r requirements.txt

如果要使用gpu需要安装cude
conda install -c nvidia cudnn=8.9.2

cuda
export LD_LIBRARY_PATH=/opt/conda/envs/face-recognition/lib:$LD_LIBRARY_PATH

一劳永逸
echo 'export LD_LIBRARY_PATH=/opt/conda/envs/face-recognition/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```
