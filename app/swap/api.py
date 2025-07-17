
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import logging
# from app.face_service import FaceService
# from app.milvus_client import FaceMilvusClient
from app.init import face_service, milvus_client
from fastapi.responses import StreamingResponse
import io


logger = logging.getLogger(__name__)

router = APIRouter()

# 初始化服务
# face_service = FaceService()
# milvus_client = FaceMilvusClient()


@router.post("/face_swap")
async def face_swap(
    source_image: UploadFile = File(..., description="源人脸图像"),
    target_image: UploadFile = File(..., description="目标图像"),
    face_index: int = Form(0, description="目标图像中要替换的人脸索引")
):
    """
    单人脸换脸
    
    - source_image: 提供人脸特征的源图像
    - target_image: 被替换人脸的目标图像  
    - face_index: 目标图像中要替换的人脸索引（从0开始）
    """
    try:
        # 验证文件类型
        for file in [source_image, target_image]:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        source_data = await source_image.read()
        target_data = await target_image.read()
        
        # 执行换脸
        result_image, info = face_service.swap_face(
            source_image=source_data,
            target_image=target_data,
            face_index=face_index
        )
        
        if result_image is None:
            raise HTTPException(status_code=400, detail=info.get("error", "换脸失败"))
        
        # 返回图像
        return StreamingResponse(
            io.BytesIO(result_image),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=face_swapped.jpg",
                "X-Face-Swap-Info": str(info)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"换脸接口错误: {e}")
        raise HTTPException(status_code=500, detail=f"换脸失败: {str(e)}")


@router.post("/face_swap_multiple")
async def face_swap_multiple(
    source_image: UploadFile = File(..., description="源人脸图像"),
    target_image: UploadFile = File(..., description="目标图像")
):
    """
    多人脸换脸（替换目标图像中的所有人脸）
    """
    try:
        # 验证文件类型
        for file in [source_image, target_image]:
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        source_data = await source_image.read()
        target_data = await target_image.read()
        
        # 执行多人脸换脸
        result_image, info = face_service.swap_face_multiple(
            source_image=source_data,
            target_image=target_data
        )
        
        if result_image is None:
            raise HTTPException(status_code=400, detail=info.get("error", "换脸失败"))
        
        # 返回图像
        return StreamingResponse(
            io.BytesIO(result_image),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=face_swapped_multiple.jpg",
                "X-Face-Swap-Info": str(info)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多人脸换脸接口错误: {e}")
        raise HTTPException(status_code=500, detail=f"多人脸换脸失败: {str(e)}")


@router.post("/preview_faces")
async def preview_faces(
    image: UploadFile = File(..., description="要预览人脸的图像")
):
    """
    预览图像中的所有人脸（用于选择要替换的人脸）
    """
    try:
        # 验证文件类型
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 读取图像数据
        image_data = await image.read()
        
        # 预览人脸
        result_image, info = face_service.preview_faces(image_data)
        
        if result_image is None:
            raise HTTPException(status_code=400, detail=info.get("error", "人脸预览失败"))
        
        # 返回标注了人脸的图像
        return StreamingResponse(
            io.BytesIO(result_image),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=faces_preview.jpg",
                "X-Faces-Info": str(info)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"人脸预览接口错误: {e}")
        raise HTTPException(status_code=500, detail=f"人脸预览失败: {str(e)}")



@router.post("/face_swap_with_database")
async def face_swap_with_database(
    face_id: str = Form(..., description="数据库中的人脸ID"),
    target_image: UploadFile = File(..., description="目标图像"),
    face_index: int = Form(0, description="目标图像中要替换的人脸索引")
):
    """
    使用数据库中的人脸进行换脸
    """
    try:
        # 验证文件类型
        if not target_image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="文件必须是图像格式")
        
        # 从数据库搜索人脸
        search_results = milvus_client.search_face_by_id(face_id)
        if not search_results:
            raise HTTPException(status_code=404, detail="数据库中未找到指定的人脸")
        
        # 获取人脸向量
        face_vector = search_results[0]["vector"]
        
        # 这里需要你实现从向量重建人脸图像的功能
        # 或者在数据库中存储原始图像
        # 暂时返回错误提示
        raise HTTPException(
            status_code=501, 
            detail="此功能需要在数据库中存储原始人脸图像，暂未实现"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"数据库换脸接口错误: {e}")
        raise HTTPException(status_code=500, detail=f"数据库换脸失败: {str(e)}")
