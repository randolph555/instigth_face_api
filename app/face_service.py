import insightface
import numpy as np
import cv2
from PIL import Image
import io
import os
import logging
from typing import List, Dict, Optional, Tuple
from app.config import config

logger = logging.getLogger(__name__)


class FaceService:
    def __init__(self):
        self.app = None
        self._init_model()
    
    def _init_model(self):
        """初始化InsightFace模型"""
        try:
            # 设置模型根目录
            os.makedirs(config.INSIGHTFACE_MODEL_ROOT, exist_ok=True)
            
            # 检查GPU显存并决定使用CPU还是GPU
            use_gpu = self._should_use_gpu()

            # 初始化InsightFace
            self.app = insightface.app.FaceAnalysis(
                name=config.INSIGHTFACE_MODEL_NAME,
                root=config.INSIGHTFACE_MODEL_ROOT,
                #providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.GPU_ID >= 0 else ['CPUExecutionProvider']
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

            )
            
            # 准备模型
            ctx_id = config.GPU_ID if config.GPU_ID >= 0 else -1
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            logger.info("InsightFace模型初始化成功")
            
             # 初始化换脸模型
            self.face_swapper = insightface.model_zoo.get_model(
                'models/models/inswapper_128.onnx',
                root=config.INSIGHTFACE_MODEL_ROOT,
                #providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if config.GPU_ID >= 0 else ['CPUExecutionProvider']
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

            )
            
        except Exception as e:
            logger.error(f"InsightFace模型初始化失败: {e}")
            raise
        
        
    def _should_use_gpu(self):
        """检查是否应该使用GPU"""
        # 如果配置为CPU模式，直接返回False
        if config.GPU_ID < 0:
            return False
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            # 检查指定GPU的显存
            handle = pynvml.nvmlDeviceGetHandleByIndex(config.GPU_ID)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # 计算可用显存（GB）
            free_memory_gb = (info.total - info.used) / (1024**3)
            
            logger.info(f"GPU {config.GPU_ID} 可用显存: {free_memory_gb:.1f}GB")
            
            # 如果可用显存少于3GB，使用CPU
            if free_memory_gb < 3.0:
                logger.warning(f"GPU显存不足({free_memory_gb:.1f}GB < 3GB)，切换到CPU模式")
                return False
            
            # 检查ONNX Runtime是否支持CUDA
            import onnxruntime as ort
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                logger.warning("ONNX Runtime不支持CUDA，切换到CPU模式")
                return False
                
            return True
            
        except ImportError:
            logger.warning("pynvml未安装，无法检查GPU显存，使用CPU模式")
            return False
        except Exception as e:
            logger.warning(f"GPU检查失败: {e}，切换到CPU模式")
            return False

        
    
    def _preprocess_image(self, image_data: bytes) -> np.ndarray:
        """预处理图像"""
        try:
            # 将bytes转换为PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 转换为BGR格式（OpenCV格式）
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise
    
    def extract_face_features(self, image_data: bytes) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """提取人脸特征向量和属性"""
        try:
            # 预处理图像
            img = self._preprocess_image(image_data)
            
            # 检测和分析人脸
            faces = self.app.get(img)
            
            if not faces:
                return None, []
            
            # 取第一个检测到的人脸
            face = faces[0]
            
            # 提取特征向量
            embedding = face.embedding
            
            # 提取人脸属性
            attributes = {
                "bbox": face.bbox.tolist(),
                "landmark": face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else [],
                "age": int(face.age) if hasattr(face, 'age') else None,
                "gender": "male" if hasattr(face, 'sex') and face.sex == "M" else "female" if hasattr(face, 'sex') else None,
                "embedding_norm": float(np.linalg.norm(embedding)),
                "det_score": float(face.det_score)
            }
            
            return embedding, [attributes]
            
        except Exception as e:
            logger.error(f"人脸特征提取失败: {e}")
            return None, []
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """比较两个人脸特征向量的相似度"""
        try:
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
            
        except Exception as e:
            logger.error(f"人脸比较失败: {e}")
            return 0.0
    
    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """判断是否为同一人"""
        similarity = self.compare_faces(embedding1, embedding2)
        return similarity >= config.SIMILARITY_THRESHOLD





    # 换脸逻辑代码
    def _postprocess_image(self, img_bgr: np.ndarray) -> bytes:
        """后处理图像，转换为bytes"""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # 转换为bytes
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format='JPEG', quality=95)
            img_bytes.seek(0)
            return img_bytes.getvalue()
        except Exception as e:
            logger.error(f"图像后处理失败: {e}")
            raise
    
    def swap_face(self, source_image: bytes, target_image: bytes, face_index: int = 0) -> Tuple[Optional[bytes], dict]:
        """
        换脸功能
        
        Args:
            source_image: 源人脸图像（提供人脸特征）
            target_image: 目标图像（被替换人脸的图像）
            face_index: 目标图像中要替换的人脸索引（默认0，即第一个人脸）
        
        Returns:
            换脸后的图像bytes和处理信息
        """
        try:
            # 预处理图像
            source_img = self._preprocess_image(source_image)
            target_img = self._preprocess_image(target_image)
            
            # 检测源图像中的人脸
            source_faces = self.app.get(source_img)
            if not source_faces:
                return None, {"error": "源图像中未检测到人脸"}
            
            # 检测目标图像中的人脸
            target_faces = self.app.get(target_img)
            if not target_faces:
                return None, {"error": "目标图像中未检测到人脸"}
            
            # 检查face_index是否有效
            if face_index >= len(target_faces):
                return None, {"error": f"目标图像中只有{len(target_faces)}个人脸，索引{face_index}无效"}
            
            # 获取源人脸（使用第一个检测到的人脸）
            source_face = source_faces[0]
            
            # 获取目标人脸
            target_face = target_faces[face_index]
            
            # 执行换脸
            result_img = self.face_swapper.get(target_img, target_face, source_face, paste_back=True)
            
            # 后处理
            result_bytes = self._postprocess_image(result_img)
            
            # 处理信息
            info = {
                "success": True,
                "source_faces_count": len(source_faces),
                "target_faces_count": len(target_faces),
                "swapped_face_index": face_index,
                "source_face_info": {
                    "bbox": source_face.bbox.tolist(),
                    "age": int(source_face.age) if hasattr(source_face, 'age') else None,
                    "gender": "male" if hasattr(source_face, 'sex') and source_face.sex == "M" else "female" if hasattr(source_face, 'sex') else None,
                    "det_score": float(source_face.det_score)
                },
                "target_face_info": {
                    "bbox": target_face.bbox.tolist(),
                    "age": int(target_face.age) if hasattr(target_face, 'age') else None,
                    "gender": "male" if hasattr(target_face, 'sex') and target_face.sex == "M" else "female" if hasattr(target_face, 'sex') else None,
                    "det_score": float(target_face.det_score)
                }
            }
            
            return result_bytes, info
            
        except Exception as e:
            logger.error(f"换脸处理失败: {e}")
            return None, {"error": f"换脸处理失败: {str(e)}"}
    
    def swap_face_multiple(self, source_image: bytes, target_image: bytes) -> Tuple[Optional[bytes], dict]:
        """
        多人脸换脸（将源人脸替换到目标图像的所有人脸上）
        """
        try:
            source_img = self._preprocess_image(source_image)
            target_img = self._preprocess_image(target_image)
            
            source_faces = self.app.get(source_img)
            if not source_faces:
                return None, {"error": "源图像中未检测到人脸"}
            
            target_faces = self.app.get(target_img)
            if not target_faces:
                return None, {"error": "目标图像中未检测到人脸"}
            
            source_face = source_faces[0]
            result_img = target_img.copy()
            
            # 对每个目标人脸进行换脸
            swapped_faces = []
            for i, target_face in enumerate(target_faces):
                result_img = self.face_swapper.get(result_img, target_face, source_face, paste_back=True)
                swapped_faces.append({
                    "index": i,
                    "bbox": target_face.bbox.tolist(),
                    "age": int(target_face.age) if hasattr(target_face, 'age') else None,
                    "gender": "male" if hasattr(target_face, 'sex') and target_face.sex == "M" else "female" if hasattr(target_face, 'sex') else None,
                    "det_score": float(target_face.det_score)
                })
            
            result_bytes = self._postprocess_image(result_img)
            
            info = {
                "success": True,
                "source_faces_count": len(source_faces),
                "target_faces_count": len(target_faces),
                "swapped_faces": swapped_faces,
                "source_face_info": {
                    "bbox": source_face.bbox.tolist(),
                    "age": int(source_face.age) if hasattr(source_face, 'age') else None,
                    "gender": "male" if hasattr(source_face, 'sex') and source_face.sex == "M" else "female" if hasattr(source_face, 'sex') else None,
                    "det_score": float(source_face.det_score)
                }
            }
            
            return result_bytes, info
            
        except Exception as e:
            logger.error(f"多人脸换脸失败: {e}")
            return None, {"error": f"多人脸换脸失败: {str(e)}"}
    
    def preview_faces(self, image: bytes) -> Tuple[Optional[bytes], dict]:
        """
        预览图像中的所有人脸（用于选择要替换的人脸）
        """
        try:
            img = self._preprocess_image(image)
            faces = self.app.get(img)
            
            if not faces:
                return None, {"error": "图像中未检测到人脸"}
            
            # 在图像上标注所有人脸
            result_img = img.copy()
            faces_info = []
            
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                
                # 绘制人脸框
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制人脸索引
                cv2.putText(result_img, f"Face {i}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 收集人脸信息
                faces_info.append({
                    "index": i,
                    "bbox": bbox.tolist(),
                    "age": int(face.age) if hasattr(face, 'age') else None,
                    "gender": "male" if hasattr(face, 'sex') and face.sex == "M" else "female" if hasattr(face, 'sex') else None,
                    "det_score": float(face.det_score)
                })
            
            result_bytes = self._postprocess_image(result_img)
            
            info = {
                "success": True,
                "faces_count": len(faces),
                "faces_info": faces_info
            }
            
            return result_bytes, info
            
        except Exception as e:
            logger.error(f"人脸预览失败: {e}")
            return None, {"error": f"人脸预览失败: {str(e)}"}