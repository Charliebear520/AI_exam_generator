# main.py
import os
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from database import get_db, engine, Base
from models import Question
from parse_pdf import save_to_json  # 只保留 save_to_json，其他函數不再需要
from use_gemini import process_pdf_with_gemini

# 創建資料表
Base.metadata.create_all(bind=engine)

app = FastAPI(title="考試題目 OCR API", description="將 PDF 考試題目轉換為 JSON 格式")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...), 
    exam_name: str = Form(...),
    password: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    上傳 PDF 檔案，使用 Gemini API 轉換為結構化 JSON，並儲存到資料庫與檔案中。
    """
    progress = []
    
    # 檢查是否有上傳檔案
    if not file:
        raise HTTPException(status_code=400, detail="未收到上傳的檔案")
    
    # 檢查檔案名稱
    if not file.filename:
        raise HTTPException(status_code=400, detail="檔案名稱無效")
    
    # 清理檔案名稱
    def clean_filename(filename):
        # 移除特殊字符
        special_chars = '【】()（）、，。：:!！?？'
        clean_name = filename
        for char in special_chars:
            clean_name = clean_name.replace(char, '')
        # 替換空格
        clean_name = clean_name.replace(' ', '_')
        return clean_name
    
    original_filename = file.filename
    cleaned_filename = clean_filename(original_filename)
    print(f"原始檔案名稱: {original_filename}")
    print(f"處理後檔案名稱: {cleaned_filename}")
    
    # 檢查檔案類型
    if not original_filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail=f"不支援的檔案類型: {original_filename}。只接受 PDF 檔案")
    
    # 檢查考試名稱
    if not exam_name or exam_name.strip() == "":
        raise HTTPException(status_code=400, detail="考試名稱不能為空")
    
    try:
        progress.append("開始接收 PDF 檔案")
        pdf_bytes = await file.read()
        
        # 檢查檔案是否為空
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="檔案內容為空")
        
        # 檢查檔案大小
        file_size_mb = len(pdf_bytes) / (1024 * 1024)
        print(f"檔案大小: {file_size_mb:.1f}MB")
        
        MAX_FILE_SIZE_MB = 20  # 增加到20MB
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400, 
                detail={
                    "message": f"PDF檔案太大 ({file_size_mb:.1f}MB)",
                    "current_size": f"{file_size_mb:.1f}MB",
                    "max_size": f"{MAX_FILE_SIZE_MB}MB",
                    "suggestion": "請嘗試壓縮PDF檔案大小，或分割成多個較小的檔案"
                }
            )
        elif file_size_mb == 0:
            raise HTTPException(status_code=400, detail="檔案大小為0，請確認檔案是否正確")
        
        progress.append(f"PDF 檔案接收完成 (大小: {file_size_mb:.1f}MB)")
        
        # 使用 Gemini API 處理 PDF
        gemini_result = process_pdf_with_gemini(pdf_bytes, cleaned_filename, password)
        
        # 檢查是否有錯誤
        if "error" in gemini_result:
            error_message = gemini_result["error"]
            print(f"Gemini API 處理失敗: {error_message}")
            if "已加密" in error_message and not password:
                raise HTTPException(status_code=422, detail=error_message)
            raise HTTPException(status_code=500, detail=error_message)
        
        if gemini_result and "questions" in gemini_result:
            questions = gemini_result["questions"]
            progress.append("Gemini API 結構化處理完成")
        else:
            print(f"Gemini API 返回的結果格式不正確: {gemini_result}")
            raise HTTPException(status_code=422, detail="Gemini API 未返回有效結果")
        
        if not questions:
            raise HTTPException(status_code=422, detail="無法從 PDF 中識別題目格式")
        
        progress.append(f"共解析出 {len(questions)} 道題目，開始儲存至資料庫")
        
        # 儲存到資料庫
        try:
            for question in questions:
                # 驗證題目資料
                if not isinstance(question, dict):
                    print(f"跳過無效題目資料: {question}")
                    continue
                
                # 確保必要欄位存在
                question_id = question.get("id", 0)
                if not question_id:
                    question_id = questions.index(question) + 1
                
                content = question.get("content", "")
                if not content:
                    print(f"題目 {question_id} 缺少內容，跳過")
                    continue
                
                options = question.get("options", {})
                if not options or not isinstance(options, dict):
                    print(f"題目 {question_id} 缺少選項或選項格式不正確，使用空選項")
                    options = {}
                
                # 建立題目記錄，允許answer和explanation為空
                db_question = Question(
                    exam_name=exam_name,
                    question_number=question_id,
                    content=content,
                    options=options,
                    answer=question.get("answer") if question.get("answer") is not None else "",  # 確保不是None
                    explanation=question.get("explanation") if question.get("explanation") is not None else ""  # 確保不是None
                )
                db.add(db_question)
            
            db.commit()
            progress.append("資料庫儲存完成")
        except Exception as db_error:
            db.rollback()
            print(f"資料庫儲存錯誤: {str(db_error)}")
            raise HTTPException(status_code=500, detail=f"儲存題目到資料庫時發生錯誤: {str(db_error)}")
        
        # 生成 JSON 檔案
        try:
            json_filename = save_to_json(questions, exam_name, DOWNLOAD_DIR)
            progress.append("JSON 檔案生成完成")
        except Exception as json_error:
            print(f"生成JSON檔案錯誤: {str(json_error)}")
            raise HTTPException(status_code=500, detail=f"生成JSON檔案時發生錯誤: {str(json_error)}")
        
        return {
            "message": f"成功解析 {len(questions)} 道題目",
            "download_url": f"/download/{json_filename}",
            "questions_count": len(questions),
            "progress": progress
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"處理 PDF 時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"處理 PDF 時發生錯誤: {str(e)}")

@app.delete("/questions/{exam_name}")
async def delete_questions(
    exam_name: str,
    db: Session = Depends(get_db)
):
    """
    根據考試名稱刪除資料庫中的題目記錄
    """
    try:
        # 打印接收到的考试名称，用于调试
        print(f"收到删除请求，考试名称: '{exam_name}'")
        
        # 检查考试名称是否有效
        if not exam_name or exam_name == "undefined":
            raise HTTPException(status_code=400, detail=f"考試名稱無效: '{exam_name}'")
        
        # 查詢要刪除的記錄數量
        count = db.query(Question).filter(Question.exam_name == exam_name).count()
        if count == 0:
            raise HTTPException(status_code=404, detail=f"找不到考試名稱為 '{exam_name}' 的題目")
        
        # 執行刪除操作
        db.query(Question).filter(Question.exam_name == exam_name).delete()
        db.commit()
        
        # 同時刪除對應的JSON檔案（如果存在）
        json_filename = f"{exam_name.replace(' ', '_')}.json"
        file_path = os.path.join(DOWNLOAD_DIR, json_filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        else:
            print(f"文件不存在: {file_path}")
            
        return {"message": f"成功刪除 {count} 道題目", "exam_name": exam_name}
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"删除题目时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"刪除題目時發生錯誤: {str(e)}")

@app.get("/download/{filename}")
async def download_json(filename: str):
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="檔案不存在")
    return FileResponse(path=file_path, filename=filename, media_type="application/json")

@app.get("/questions")
async def get_questions(
    exam_name: Optional[str] = None,
    skip: int = 0, 
    limit: int = 100,
    db: Session = Depends(get_db)
):
    query = db.query(Question)
    if exam_name:
        query = query.filter(Question.exam_name == exam_name)
    # 按 created_at 降序排序
    query = query.order_by(Question.created_at.desc())
    questions = query.offset(skip).limit(limit).all()
    result = {
        "exam": exam_name,
        "questions": [q.to_dict() for q in questions],
        "total": query.count()
    }
    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)