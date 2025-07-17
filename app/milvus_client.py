from pymilvus import MilvusClient, DataType
from typing import List, Dict, Any, Optional
import logging
from app.config import config

logger = logging.getLogger(__name__)

class FaceMilvusClient:
    def __init__(self):
        self.client = MilvusClient(
            uri=config.MILVUS_URI,
            token=config.MILVUS_TOKEN
        )
        self.collection_name = config.COLLECTION_NAME
        if not self.client.has_collection(self.collection_name):
            self._init_collection()
    
    def _init_collection(self):
        """初始化集合"""
        try:
            # 如果集合存在，先删除
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"已删除现有集合: {self.collection_name}")
            
            # 创建新集合
            schema = self.client.create_schema(
                auto_id=True,
                enable_dynamic_field=True
            )
            
            # 添加字段
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(field_name="face_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="person_name", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=config.VECTOR_DIM)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            
            # 创建索引参数
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024}
            )
            
            # 创建集合
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            
            logger.info(f"成功创建集合: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"初始化集合失败: {e}")
            raise
    
    def insert_face(self, face_id: str, person_name: str, vector: List[float], metadata: Dict = None) -> bool:
        """插入人脸向量"""
        try:
            data = {
                "face_id": face_id,
                "person_name": person_name,
                "vector": vector,
                "metadata": metadata or {}
            }
            
            result = self.client.insert(
                collection_name=self.collection_name,
                data=[data]
            )
            
            logger.info(f"成功插入人脸数据: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"插入人脸数据失败: {e}")
            return False
    
    def search_face(self, vector: List[float], top_k: int = 5) -> List[Dict]:
        """搜索相似人脸"""
        try:
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            results = self.client.search(
                collection_name=self.collection_name,
                data=[vector],
                limit=top_k,
                search_params=search_params,
                output_fields=["face_id", "person_name", "metadata"]
            )
            
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:
                    formatted_results.append({
                        "face_id": hit["entity"]["face_id"],
                        "person_name": hit["entity"]["person_name"],
                        "similarity": float(hit["distance"]),
                        "metadata": hit["entity"]["metadata"]
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"搜索人脸失败: {e}")
            return []
    
    def delete_face(self, face_id: str) -> bool:
        """删除人脸数据"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f'face_id == "{face_id}"'
            )
            logger.info(f"成功删除人脸数据: {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除人脸数据失败: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return {"total_faces": stats["row_count"]}
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"total_faces": 0}
