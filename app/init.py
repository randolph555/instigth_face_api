from functools import lru_cache
import time
from app.face_service import FaceService
from app.milvus_client import FaceMilvusClient

"""" 避免每个api都引入一次，造成重复加载，浪费时间和资源 """

@lru_cache()
def _get_face_service() -> FaceService:
    """获取FaceService客户端实例（缓存）"""
    start_time = time.time()
    service = FaceService()
    print(f"FaceService初始化耗时: {time.time() - start_time:.2f}s")
    return service


@lru_cache()
def _get_milvus_client() -> FaceMilvusClient:
    """获取Milvus客户端实例（缓存）"""
    start_time = time.time()
    service = FaceMilvusClient()
    print(f"FaceMilvusClient初始化耗时: {time.time() - start_time:.2f}s")
    return service

# 全局单例实例
face_service = _get_face_service()
milvus_client = _get_milvus_client()