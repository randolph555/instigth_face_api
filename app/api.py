from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import List, Dict, Any, Optional
import base64
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import json
import uuid
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


@router.post("/add_face")
async def add_face(
    file: UploadFile = File(...),
    person_name: str = Form(...),
    face_id: Optional[str] = Form(None)
):
    """添加人脸到数据库"""
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        image_data = await file.read()
        
        # 提取人脸特征
        embedding, attributes = face_service.extract_face_features(image_data)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="未检测到人脸")
        
        # 生成face_id
        if not face_id:
            face_id = str(uuid.uuid4())
        
        # 准备元数据
        metadata = {
            "filename": file.filename,
            "attributes": attributes[0] if attributes else {}
        }
        
        # 插入到Milvus
        success = milvus_client.insert_face(
            face_id=face_id,
            person_name=person_name,
            vector=embedding.tolist(),
            metadata=metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="人脸数据插入失败")
        
        return {
            "success": True,
            "face_id": face_id,
            "person_name": person_name,
            "attributes": attributes[0] if attributes else {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加人脸失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加人脸失败: {str(e)}")


@router.post("/search_face")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = Form(5)
):
    """搜索相似人脸"""
    try:
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        image_data = await file.read()
        
        # 提取人脸特征
        embedding, attributes = face_service.extract_face_features(image_data)
        
        if embedding is None:
            raise HTTPException(status_code=400, detail="未检测到人脸")
        
        # 在Milvus中搜索
        results = milvus_client.search_face(embedding.tolist(), top_k)
        
        # 过滤结果（根据相似度阈值）
        filtered_results = [
            result for result in results 
            if result["similarity"] >= config.SEARCH_SIMILARITY_THRESHOLD
        ]
        
        return {
            "success": True,
            "query_attributes": attributes[0] if attributes else {},
            "matches": filtered_results,
            "total_matches": len(filtered_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"搜索人脸失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索人脸失败: {str(e)}")


@router.post("/compare_faces")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    """比较两张人脸图片"""
    try:
        # 验证文件类型
        for file in [file1, file2]:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        image_data1 = await file1.read()
        image_data2 = await file2.read()
        
        # 提取人脸特征
        embedding1, attributes1 = face_service.extract_face_features(image_data1)
        embedding2, attributes2 = face_service.extract_face_features(image_data2)
        
        if embedding1 is None:
            raise HTTPException(status_code=400, detail="第一张图片未检测到人脸")
        
        if embedding2 is None:
            raise HTTPException(status_code=400, detail="第二张图片未检测到人脸")
        
        # 计算相似度
        similarity = face_service.compare_faces(embedding1, embedding2)
        is_same = face_service.is_same_person(embedding1, embedding2)
        
        return {
            "success": True,
            "similarity": similarity,
            "is_same_person": is_same,
            "threshold": config.SIMILARITY_THRESHOLD,
            "face1_attributes": attributes1[0] if attributes1 else {},
            "face2_attributes": attributes2[0] if attributes2 else {}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸比较失败: {e}")
        raise HTTPException(status_code=500, detail=f"人脸比较失败: {str(e)}")


@router.delete("/delete_face/{face_id}")
async def delete_face(face_id: str):
    """删除人脸数据"""
    try:
        success = milvus_client.delete_face(face_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="人脸数据不存在或删除失败")
        
        return {
            "success": True,
            "message": f"成功删除人脸数据: {face_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除人脸失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除人脸失败: {str(e)}")

@router.get("/stats")
async def get_stats():
    """获取系统统计信息"""
    try:
        stats = milvus_client.get_collection_stats()
        
        return {
            "success": True,
            "stats": stats,
            "config": {
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "vector_dim": config.VECTOR_DIM,
                "model_name": config.INSIGHTFACE_MODEL_NAME
            }
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")



@router.websocket("/ws/camera_recognition")
async def camera_recognition_websocket(websocket: WebSocket):
    """实时摄像头人脸识别WebSocket接口 - 接收客户端图像"""
    await websocket.accept()
    
    try:
        await websocket.send_text(json.dumps({"status": "ready", "message": "服务器已准备就绪，等待图像数据"}))
        
        while True:
            try:
                # 接收来自客户端的消息
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "frame":
                    # 接收base64编码的图像数据
                    image_base64 = data.get("image")
                    if not image_base64:
                        continue
                    
                    # 解码base64图像
                    try:
                        # 移除data:image/jpeg;base64,前缀（如果存在）
                        if image_base64.startswith('data:image'):
                            image_base64 = image_base64.split(',')[1]
                        
                        image_bytes = base64.b64decode(image_base64)
                        
                        # 使用你现有的face_service提取人脸特征
                        embedding, attributes = face_service.extract_face_features(image_bytes)
                        
                        results = []
                        if embedding is not None and attributes:
                            # 使用你现有的milvus_client搜索人脸
                            search_results = milvus_client.search_face(embedding.tolist(), top_k=1)
                            
                            for attr in attributes:
                                result = {
                                    "bbox": attr["bbox"],
                                    "age": attr.get("age"),
                                    "gender": attr.get("gender"),
                                    "recognition": None
                                }
                                
                                # 如果找到匹配的人脸
                                if search_results and len(search_results) > 0:
                                    best_match = search_results[0]
                                    if best_match.get("similarity", 0) >= config.SIMILARITY_THRESHOLD:
                                        result["recognition"] = {
                                            "person_name": best_match.get("person_name", "Unknown"),
                                            "similarity": best_match.get("similarity", 0),
                                            "face_id": best_match.get("face_id")
                                        }
                                
                                results.append(result)
                        
                        # 发送识别结果回客户端
                        response = {
                            "type": "results",
                            "results": results,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                        
                        await websocket.send_text(json.dumps(response))
                        
                    except Exception as e:
                        logger.error(f"处理图像时出错: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error", 
                            "message": f"图像处理失败: {str(e)}"
                        }))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket消息处理错误: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": str(e)
                }))
                
    except Exception as e:
        logger.error(f"WebSocket连接错误: {e}")
    finally:
        await websocket.close()





