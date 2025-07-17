# DEBUG时添加
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  
# DEBUG时添加

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

import logging
from app.api import router
from app.video.api import router as video_router
from app.swap.api import router as swap_router
from app.config import config


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="人脸识别API",
    description="基于InsightFace和Milvus的人脸识别系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#前端静态页面引入
app.mount("/html", StaticFiles(directory="html"), name="static_html")


# 注册路由
app.include_router(router, prefix="/api/v1", tags=["人脸识别"])
app.include_router(video_router, prefix="/api/v1", tags=["视频处理"])
app.include_router(swap_router, prefix="/api/v1", tags=["人脸替换"])

@app.get("/docs")
async def root():
    return {
        "message": "人脸识别API服务",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}



@app.get("/")
async def root():
    """根路径重定向到主页"""
    return RedirectResponse(url="/html/home.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
