# main.py
import os
import json
from typing import List, Optional,Dict,Any
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import random

from database import get_db, engine, Base
from models import Question
from parse_pdf import save_to_json  # 只保留 save_to_json，其他函數不再需要
from use_gemini import process_pdf_with_gemini
from vector_store import vector_store
from use_gemini import adapt_questions, retrieve_online_questions  # 新增的題目改編函數


# 創建資料表
Base.metadata.create_all(bind=engine)

app = FastAPI(title="考試題目 OCR API", description="將PDF轉JSON、向量檢索與AI改編模擬考題")
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
            # 添加到向量資料庫
            vector_store.add_questions(questions, exam_name)  # 新增這一行
            progress.append("題目已添加到向量資料庫")
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
    刪除指定考試名稱的題目：
      1. 從資料庫中刪除該考試題目
      2. 從 vector store 刪除該考試題目的向量
      3. 從 downloads 資料夾中刪除所有 exam 欄位等於該考試名稱的 JSON 檔案
    """
    try:
        # 檢查資料庫是否有該考試的題目
        count = db.query(Question).filter(Question.exam_name == exam_name).count()
        if count == 0:
            raise HTTPException(status_code=404, detail=f"找不到考試名稱為 '{exam_name}' 的題目")
        
        # 從資料庫中刪除
        db.query(Question).filter(Question.exam_name == exam_name).delete()
        db.commit()
        
        # 刪除 vector store 中該考試的資料
        vector_store.delete_exam_questions(exam_name)
        return {"message": "刪除成功", "exam_name": exam_name}
        # 刪除 downloads 資料夾中所有考試欄位為 exam_name 的 JSON 檔案
        import json
        for filename in os.listdir(DOWNLOAD_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(DOWNLOAD_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # 當 JSON 裡的 exam 欄位符合則刪除
                    if data.get("exam") == exam_name:
                        os.remove(filepath)
                        print(f"已删除文件: {filepath}")
                except Exception as e:
                    print(f"處理文件 {filename} 時出錯: {str(e)}")
        
        return {"message": f"成功刪除 {count} 道題目，並同步更新相關 JSON 和向量庫", "exam_name": exam_name}
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"删除題目時發生錯誤: {str(e)}")
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
    limit: Optional[int] = None,  # 移除預設限制
    db: Session = Depends(get_db)
):
    """
    獲取題目列表，可以按考試名稱篩選
    如果沒有指定考試名稱，則返回所有考試的題目統計和最新題目
    """
    try:
        if exam_name:
            # 如果指定了考試名稱，返回該考試的所有題目
            query = db.query(Question).filter(Question.exam_name == exam_name)
            total = query.count()
            
            # 只有在明確指定limit時才進行限制
            if limit is not None:
                query = query.offset(skip).limit(limit)
            
            questions = query.order_by(Question.question_number).all()
            
            return {
                "exam": exam_name,
                "questions": [q.to_dict() for q in questions],
                "total": total
            }
        else:
            # 如果沒有指定考試名稱，返回所有考試的統計資訊
            from sqlalchemy import func
            
            # 獲取每個考試的題目數量
            exam_stats = db.query(
                Question.exam_name,
                func.count(Question.id).label('total_questions')
            ).group_by(Question.exam_name).all()
            
            # 獲取每個考試的最新題目作為預覽
            exams_data = []
            for exam_name, total in exam_stats:
                latest_questions = db.query(Question).filter(
                    Question.exam_name == exam_name
                ).order_by(Question.question_number).all()
                
                exams_data.append({
                    "exam_name": exam_name,
                    "total_questions": total,
                    "questions": [q.to_dict() for q in latest_questions]
                })
            
            return {
                "exams": exams_data,
                "total_exams": len(exam_stats)
            }
            
    except Exception as e:
        print(f"獲取題目時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取題目時發生錯誤: {str(e)}")

# API 1: 一鍵生成模擬考題
@app.post("/generate_exam")
async def generate_exam(
    exam_name: str = Form(...),
    num_questions: int = Form(10),
    keyword: Optional[str] = Form(None)
):
    """
    根據使用者輸入的考試名稱與題目數量生成模擬考題：
      1. 若 keyword 有填寫：先以關鍵字進行內部向量檢索，若無結果則從網路檢索
      2. 若 keyword 未填寫：則從整個向量資料庫隨機抽題
      3. 利用 LLM (Gemini-2.0-flash) 改編題目生成新的模擬考題
      4. 改編後的題目附上使用者指定的考試名稱（僅作標記用途）
    """
    if keyword and keyword.strip() != "":
        # 當有關鍵字時，先嘗試從內部題庫檢索相關題目
        internal_questions = vector_store.search_similar_questions(query=keyword, n_results=num_questions)
        if internal_questions and len(internal_questions) > 0:
            selected_questions = internal_questions
            print(f"從內部題庫檢索到 {len(selected_questions)} 道相關題目")
        else:
            # 內部檢索無結果，網路檢索
            print("內部題庫無相關題目，開始從網路上檢索...")
            selected_questions = retrieve_online_questions(keyword, num_questions)
            if not selected_questions or len(selected_questions) == 0:
                raise HTTPException(status_code=404, detail="無法檢索到與關鍵字相關的題目")
    else:
        # 無關鍵字則從整個向量庫中隨機選題
        all_questions = vector_store.collection.get()
        num_available = len(all_questions["documents"])
        if num_available == 0:
            raise HTTPException(status_code=404, detail="向量庫中沒有題目")
        if num_available < num_questions:
            num_questions = num_available
        sampled_indices = random.sample(range(num_available), num_questions)
        selected_questions = []
        for idx in sampled_indices:
            selected_questions.append({
                "content": all_questions["documents"][idx],
                "metadata": all_questions["metadatas"][idx]
            })

    # 利用 LLM 改編題目生成模擬考題
    adapted_exam = adapt_questions(selected_questions)
    # 在改編後的每道題目中加入使用者指定的考試名稱（僅作標記）
    for q in adapted_exam:
        q["new_exam_name"] = exam_name

    return {
        "original_exam": selected_questions,
        "adapted_exam": adapted_exam
    }

# --------------------------------------------
# API 2: 提交答案並評分
@app.post("/submit_answers")
async def submit_answers(payload: Dict[str, Any]):
    """
    接收使用者作答數據：
      payload 結構：
      {
         "adapted_exam": [ { "id": 1, "answer": "A", "explanation": "...", ...}, ... ],
         "answers": { "1": "A", "2": "C", ... }  # 使用者提交答案
      }
    對比每題正確答案，計算得分並返回每題的詳細結果及解析
    """
    adapted_exam = payload.get("adapted_exam", [])
    user_answers = payload.get("answers", {})
    if not adapted_exam:
        raise HTTPException(status_code=400, detail="缺少改編後的考題數據")
    
    score = 0
    results = []
    for question in adapted_exam:
        qid = str(question.get("id"))
        correct = question.get("answer", "").strip().upper()
        user_ans = user_answers.get(qid, "").strip().upper()
        is_correct = (user_ans == correct)
        if is_correct:
            score += 1
        results.append({
            "question_id": qid,
            "user_answer": user_ans,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
            "source_info": f" {question.get('source', '未知')} 試題"
        })
    
    return {
        "score": score,
        "total": len(adapted_exam),
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)