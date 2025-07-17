import os
from typing import Optional

class Config:
    # Milvus配置
    MILVUS_URI = "http://localhost:19530"
    MILVUS_TOKEN = "root:Milvus"
    COLLECTION_NAME = "face_vectors"
    
    # InsightFace配置
    INSIGHTFACE_MODEL_NAME = "buffalo_l"
    INSIGHTFACE_MODEL_ROOT = "models" 
    
    # 向量维度
    VECTOR_DIM = 512
    
    # GPU配置
    GPU_ID = -1  # -1 for CPU, 0+ for GPU
    
    # 搜索-相似度阈值
    SEARCH_SIMILARITY_THRESHOLD = 0.5
    
    # 人脸比较-相似度阈值
    SIMILARITY_THRESHOLD = 0.6

config = Config()
