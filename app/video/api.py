import cv2
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict
import logging
# from app.face_service import FaceService
# from app.milvus_client import FaceMilvusClient
from app.init import face_service, milvus_client
from app.config import config



logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化服务
# face_service = FaceService()
# milvus_client = FaceMilvusClient()


@router.post("/search_face_in_video")
async def search_face_in_video(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    max_frames: int = Form(100)
):
    """从视频中搜索人脸"""
    try:
        # 验证文件类型
        allowed_video_types = [
            'video/mp4', 'video/avi', 'video/mov', 'video/mkv', 
            'video/wmv', 'video/flv', 'video/webm', 'video/m4v',
            'video/3gp', 'video/quicktime'
        ]
        
        allowed_extensions = [
            '.mp4', '.avi', '.mov', '.mkv', '.wmv', 
            '.flv', '.webm', '.m4v', '.3gp'
        ]
        
        # 检查MIME类型和文件扩展名
        is_valid_mime = file.content_type in allowed_video_types or file.content_type.startswith('video/')
        is_valid_extension = file.filename and any(file.filename.lower().endswith(ext) for ext in allowed_extensions)
        
        if not (is_valid_mime or is_valid_extension):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件格式。支持的格式: {', '.join(allowed_extensions)}"
            )
        
        # 创建临时文件保存上传的视频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(await file.read())
            temp_video_path = temp_file.name
        
        try:
            # 获取视频信息并抽帧
            frames_data = _extract_frames_from_video(temp_video_path, max_frames)
            
            if not frames_data:
                raise HTTPException(status_code=400, detail="无法从视频中提取帧")
            
            # 逐帧进行人脸识别
            for frame_info in frames_data:
                frame_number = frame_info['frame_number']
                timestamp = frame_info['timestamp']
                frame_image = frame_info['image_data']
                
                # 提取人脸特征
                embedding, attributes = face_service.extract_face_features(frame_image)
                
                if embedding is not None:
                    # 在Milvus中搜索
                    results = milvus_client.search_face(embedding.tolist(), top_k)
                    
                    # 过滤结果（根据相似度阈值）
                    filtered_results = [
                        result for result in results 
                        if result["similarity"] >= config.SEARCH_SIMILARITY_THRESHOLD
                    ]
                    
                    # 如果找到匹配的人脸，立即返回结果
                    if filtered_results:
                        return {
                            "success": True,
                            "found_match": True,
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "frame_image_base64": _encode_image_to_base64(frame_image),
                            "detected_attributes": attributes[0] if attributes else {},
                            "matches": filtered_results,
                            "total_matches": len(filtered_results),
                            "total_frames_processed": frame_number
                        }
            
            # 如果所有帧都处理完了还没找到匹配
            return {
                "success": True,
                "found_match": False,
                "message": "在视频中未找到匹配的人脸",
                "total_frames_processed": len(frames_data)
            }
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"视频人脸搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频人脸搜索失败: {str(e)}")



def _extract_frames_from_video(video_path: str, max_frames: int = 100) -> List[Dict]:
    """从视频中提取帧"""
    frames_data = []
    
    try:
        # 使用OpenCV读取视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error("无法打开视频文件")
            return frames_data
        
        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"视频信息: 总帧数={total_frames}, FPS={fps}, 时长={duration:.2f}秒")
        
        # 计算抽帧间隔
        if total_frames <= max_frames:
            # 如果总帧数小于等于最大帧数，每帧都取
            frame_interval = 1
            frames_to_extract = total_frames
        else:
            # 计算间隔，确保不超过最大帧数
            frame_interval = max(1, total_frames // max_frames)
            frames_to_extract = min(max_frames, total_frames // frame_interval)
        
        logger.info(f"抽帧策略: 间隔={frame_interval}, 预计提取={frames_to_extract}帧")
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # 按间隔提取帧
            if frame_count % frame_interval == 0:
                # 计算时间戳
                timestamp = frame_count / fps if fps > 0 else 0
                
                # 将帧转换为字节数据
                _, buffer = cv2.imencode('.jpg', frame)
                image_data = buffer.tobytes()
                
                frames_data.append({
                    'frame_number': frame_count + 1,
                    'timestamp': timestamp,
                    'image_data': image_data
                })
                
                extracted_count += 1
                logger.info(f"提取第 {extracted_count} 帧 (原始帧号: {frame_count + 1}, 时间: {timestamp:.2f}s)")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"成功提取 {len(frames_data)} 帧")
        
    except Exception as e:
        logger.error(f"视频帧提取失败: {e}")
    return frames_data




def _encode_image_to_base64(image_data: bytes) -> str:
    """将图像数据编码为base64字符串"""
    import base64
    return base64.b64encode(image_data).decode('utf-8')
