# main.py
import os
import json
from typing import List, Optional,Dict,Any
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
import random

from database import get_db, engine, Base
from models import Question
from parse_pdf import save_to_json
from use_gemini import process_pdf_with_gemini, clean_metadata_for_json
from vector_store import vector_store
from use_gemini import adapt_questions, retrieve_online_questions, retrieve_questions_by_keyword, generate_questions_with_llm, format_question_for_llm
from process_exam_points import process_and_update_vector_store, split_exam_point, format_exam_point


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
async def handleUploadPDF(
    file: UploadFile = File(...),
    exam_name: str = Form(...),
    password: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    處理上傳的 PDF 檔案:
    1. 提取文字
    2. 使用 Gemini API 解析、結構化題目
    3. 向量化並保存到向量資料庫
    """
    try:
        # 獲取檔案內容
        content = await file.read()
        filename = file.filename
        
        # 記錄處理過程
        progress = []
        progress.append(f"原始檔案名稱: {filename}")
        progress.append(f"處理後檔案名稱: {filename}")
        progress.append(f"檔案大小: {len(content)/1024/1024:.1f}MB")
        
        # 處理 PDF
        result = process_pdf_with_gemini(content, filename, password)
        
        # 清理數據中的None值
        if result:
            result = clean_metadata_for_json(result)
        
        # 檢查是否有錯誤
        if "error" in result:
            error_msg = result["error"]
            
            # 如果是配額錯誤但有部分題目已處理成功
            if "quota_exceeded" in result and result.get("questions"):
                questions = result["questions"]
                progress.append(f"由於API配額限制，僅成功提取了 {len(questions)} 道題目")
                # 繼續處理已提取的題目
            else:
                # 如果是其他錯誤或沒有提取到任何題目
                print(f"Gemini API 處理失敗: {error_msg}")
                raise HTTPException(status_code=500, detail=f"處理PDF時發生錯誤: {error_msg}")
        
        questions = result["questions"]
        
        # 檢查警告
        if "warning" in result:
            progress.append(f"警告: {result['warning']}")
            
        # 確保所有數據格式正確，特別是JSON相關字段
        for question in questions:
            # 處理選項缺失或格式不正確的情況
            if "options" not in question or question["options"] is None or not isinstance(question["options"], dict):
                question_number = question.get("id", "未知")
                print(f"題目 {question_number} 缺少選項或選項格式不正確，使用空選項")
                question["options"] = {}
            
            # 確保其他JSON字段不為None
            if question.get("keywords") is None:
                question["keywords"] = []
            if question.get("law_references") is None:
                question["law_references"] = []
            if question.get("exam_point") is None:
                question["exam_point"] = ""
            if question.get("type") is None:
                # 根據選項數量和答案格式推斷題目類型
                options = question.get("options", {})
                answer = question.get("answer", "")
                
                if len(options) == 0:
                    # 沒有選項，可能是簡答題或申論題
                    question["type"] = "簡答題"
                elif "," in answer:
                    # 多個答案，可能是多選題
                    question["type"] = "多選題"
                else:
                    # 默認為單選題
                    question["type"] = "單選題"
                
                progress.append(f"為題目 {question.get('id', '未知')} 自動設置題型: {question['type']}")
                
        # 檢查是否有重複ID，並做相應處理
        seen_ids = set()
        duplicate_count = 0
        for q in questions:
            if q["id"] in seen_ids:
                duplicate_count += 1
            seen_ids.add(q["id"])
        
        if duplicate_count > 0:
            progress.append(f"檢測到 {duplicate_count} 個重複ID，已自動處理以確保ID唯一性")
        
        # 存儲到數據庫
        try:
            # 將題目保存到 PostgreSQL
            for question in questions:
                db_question = Question(
                    exam_name=exam_name,
                    question_number=question["id"],
                    content=question["content"],
                    options=question["options"],
                    answer=question.get("answer", ""),
                    explanation=question.get("explanation", ""),
                    exam_point=question.get("exam_point", ""),
                    keywords=question.get("keywords", []),
                    law_references=question.get("law_references", []),
                    type=question.get("type", "單選題")
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
            vector_store.add_questions(questions, exam_name)
            progress.append("題目已添加到向量資料庫")
        except Exception as json_error:
            print(f"生成JSON檔案錯誤: {str(json_error)}")
            # 提供一個安全的後備方案
            result = json.dumps({"filename": filename, "questions": [], "error": "數據處理錯誤"})
        
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
async def generate_exam(exam_name: str = Form(...), num_questions: int = Form(...), keyword: str = Form(None)):
    """
    生成模擬考題：
      1. 如果有關鍵字，使用 RAG 檢索相關題目
      2. 如果檢索到的題目不足，使用 LLM 生成補充題目
      3. 如果沒有關鍵字，從題庫中隨機選擇題目
    """
    print(f"生成模擬考題: 考試={exam_name}, 數量={num_questions}, 關鍵字={keyword}, 題型=None")
    
    try:
        # 如果有關鍵字，就使用 RAG 檢索
        if keyword:
            # 檢索相關題目
            retrieved_questions = retrieve_questions_by_keyword(keyword, num_questions)
            
            # 檢查是否有足夠的題目
            if len(retrieved_questions) < num_questions:
                # 不夠題目，需要用 LLM 生成
                existing_count = len(retrieved_questions)
                needed_count = num_questions - existing_count
                print(f"檢索到的題目不足，需要生成 {needed_count} 道補充題目")
                
                # 準備一些示例題目作為 LLM 的參考
                examples = []
                if retrieved_questions:
                    examples = [format_question_for_llm(q) for q in retrieved_questions[:min(3, len(retrieved_questions))]]
                
                # 呼叫 LLM 生成其餘題目
                generated_questions = generate_questions_with_llm(
                    keyword=keyword,
                    count=needed_count,
                    examples=examples
                )
                
                # 標記生成的題目來源
                for q in generated_questions:
                    q["source"] = "由 Gemini 出題"  # 或其他 LLM 名稱
                    # 確保生成的題目 ID 不與已有題目衝突
                    q["id"] = existing_count + generated_questions.index(q) + 1
                
                # 合併檢索到的和生成的題目
                final_questions = retrieved_questions + generated_questions
                print(f"最終題目數量: 檢索 {len(retrieved_questions)} + 生成 {len(generated_questions)} = {len(final_questions)}")
            else:
                # 有足夠題目，如果超過需求數量就隨機選擇
                if len(retrieved_questions) > num_questions:
                    import random
                    final_questions = random.sample(retrieved_questions, num_questions)
                else:
                    final_questions = retrieved_questions
                print(f"檢索到 {len(final_questions)} 道相關題目")
        else:
            # 無關鍵字，使用原有邏輯生成考題
            final_questions = generate_regular_exam(exam_name, num_questions)
        
        return {"adapted_exam": final_questions}
    
    except Exception as e:
        print(f"生成模擬考題時發生錯誤：{str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"生成模擬考題失敗：{str(e)}")

def generate_regular_exam(exam_name, num_questions):
    """
    生成常規考題（無關鍵字檢索時使用）
    
    Args:
        exam_name: 考試名稱
        num_questions: 題目數量
    
    Returns:
        list: 題目列表
    """
    # 從資料庫中獲取所有該考試的題目
    from vector_store import vector_store
    
    # 首先嘗試從資料庫中獲取該考試的題目
    results = vector_store.search_by_exam_name(exam_name, 100)  # 先取100題作為候選
    
    # 如果找不到或數量不足，則取所有可用題目
    if len(results) < num_questions:
        all_data = vector_store.collection.get()
        if all_data["metadatas"]:
            results = []
            for i, (doc, metadata) in enumerate(zip(all_data["documents"], all_data["metadatas"])):
                results.append({"content": doc, "metadata": metadata})
    
    # 轉換格式
    questions = []
    for idx, result in enumerate(results, 1):
        metadata = result["metadata"]
        content = result["content"]
        
        # 處理選項，確保格式正確
        options = metadata.get("options", {})
        if isinstance(options, str):
            try:
                import json
                options = json.loads(options)
            except:
                options = {}
        
        # 處理其他元數據字段
        keywords = metadata.get("keywords", [])
        if isinstance(keywords, str):
            try:
                import json
                keywords = json.loads(keywords)
            except:
                keywords = []
        
        law_references = metadata.get("law_references", [])
        if isinstance(law_references, str):
            try:
                import json
                law_references = json.loads(law_references)
            except:
                law_references = []
        
        question = {
            "id": idx,
            "content": content,
            "options": options,
            "answer": metadata.get("answer", ""),
            "explanation": metadata.get("explanation", ""),
            "exam_point": metadata.get("exam_point", ""),
            "keywords": keywords,
            "law_references": law_references,
            "type": metadata.get("type", "單選題"),
            "source": f"來自 {metadata.get('exam_name', '未知來源')}"
        }
        questions.append(question)
    
    # 如果題目數量超過需求，則隨機取出指定數量
    if len(questions) > num_questions:
        import random
        questions = random.sample(questions, num_questions)
    
    print(f"從題庫中選取了 {len(questions)} 道題目")
    return questions

# --------------------------------------------
# API 2: 提交答案並評分
@app.post("/submit_answers")
async def submit_answers(payload: Dict[str, Any]):
    """
    接收使用者作答數據並評分：
      payload 結構：
      {
         "adapted_exam": [ { "id": 1, "answer": "A", "explanation": "...", "type": "單選題", ...}, ... ],
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
        correct_answer = question.get("answer", "").strip().upper()
        user_ans = user_answers.get(qid, "").strip().upper()
        q_type = question.get("type", "單選題")
        
        # 根據題目類型進行不同的答案比對
        is_correct = False
        if q_type == "單選題":
            # 單選題比對
            is_correct = (user_ans == correct_answer)
        elif q_type == "多選題":
            # 多選題比對（需要嚴格匹配所有選項）
            correct_options = set(correct_answer.split(","))
            user_options = set(user_ans.split(","))
            is_correct = (correct_options == user_options)
        else:
            # 其他題型
            is_correct = (user_ans == correct_answer)
            
        if is_correct:
            score += 1
            
        results.append({
            "question_id": qid,
            "user_answer": user_ans,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
            "type": q_type,
            "source_info": f" {question.get('source', '未知')} 試題",
            "exam_point": question.get("exam_point", ""),
            "keywords": question.get("keywords", []),
            "law_references": question.get("law_references", [])
        })
    
    return {
        "score": score,
        "total": len(adapted_exam),
        "results": results
    }

@app.get("/legal/search")
async def search_legal_questions(
    query: str,
    search_type: str = "semantic",  # semantic, keyword, exam_point, law_reference, question_type
    n_results: int = 10,
    include_context: bool = False  # 是否包含題目相關上下文信息
):
    """
    專業法律搜索API，支持語義搜索、關鍵字搜索、考點搜索、法條搜索和題型搜索
    
    Args:
        query: 搜索關鍵詞
        search_type: 搜索類型（semantic=語義搜索, keyword=關鍵字搜索, 
                  exam_point=考點搜索, law_reference=法條搜索, question_type=題型搜索）
        n_results: 返回結果數量
        include_context: 是否包含相關上下文信息（如相似的題目）
    """
    try:
        print(f"執行搜索: 類型={search_type}, 查詢詞='{query}', 結果數量={n_results}")
        
        if search_type == "semantic":
            # 語義搜索
            results = vector_store.search_similar_questions(query, n_results)
        elif search_type == "keyword":
            # 關鍵字搜索
            results = vector_store.search_by_keyword(query, n_results)
        elif search_type == "exam_point":
            # 考點搜索
            results = vector_store.search_by_exam_point(query, n_results)
        elif search_type == "law_reference":
            # 法條搜索
            results = vector_store.search_by_law_reference(query, n_results)
        elif search_type == "question_type":
            # 題型搜索
            results = vector_store.search_by_question_type(query, n_results)
        else:
            raise HTTPException(status_code=400, detail="無效的搜索類型")
        
        # 處理返回結果，確保法律元數據正確顯示
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            
            # 解析JSON字符串為Python對象
            try:
                if isinstance(metadata.get("keywords"), str):
                    metadata["keywords"] = json.loads(metadata["keywords"])
                if isinstance(metadata.get("law_references"), str):
                    metadata["law_references"] = json.loads(metadata["law_references"])
                if isinstance(metadata.get("options"), str):
                    metadata["options"] = json.loads(metadata["options"])
            except:
                # 如果解析失敗，確保字段有默認值
                if "keywords" not in metadata or metadata["keywords"] is None:
                    metadata["keywords"] = []
                if "law_references" not in metadata or metadata["law_references"] is None:
                    metadata["law_references"] = []
                if "options" not in metadata or metadata["options"] is None:
                    metadata["options"] = {}
                
            formatted_result = {
                "content": result["content"],
                "metadata": metadata,
                "distance": result.get("distance", 0)
            }
            
            # 如果請求包含上下文，找到與該題目相關的其他題目
            if include_context:
                # 使用題目內容進行語義搜索，找到相似題目
                similar_content = result["content"]
                if "exam_point" in metadata and metadata["exam_point"]:
                    similar_content += f" 考點: {metadata['exam_point']}"
                    
                context_results = vector_store.search_similar_questions(
                    query=similar_content, 
                    n_results=3  # 限制上下文結果數量
                )
                
                # 過濾掉原題目
                context_items = []
                current_id = metadata.get("id")
                for ctx in context_results:
                    ctx_id = ctx["metadata"].get("id")
                    if ctx_id != current_id:  # 排除當前題目
                        context_items.append({
                            "content": ctx["content"],
                            "metadata": ctx["metadata"],
                            "distance": ctx.get("distance", 0)
                        })
                
                formatted_result["related_questions"] = context_items
            
            formatted_results.append(formatted_result)
        
        return {
            "results": formatted_results, 
            "total": len(formatted_results),
            "search_info": {
                "type": search_type,
                "query": query,
                "include_context": include_context
            }
        }
    
    except Exception as e:
        print(f"搜索法律題目時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索法律題目時發生錯誤: {str(e)}")

# 還要更新 /vector-search 端點以支持新增的法律元數據字段
@app.get("/vector-search")
async def vector_search(query: str, n_results: int = 5):
    """
    使用向量檢索搜索相似題目
    """
    try:
        results = vector_store.search_similar_questions(query, n_results)
        
        # 處理結果，確保法律元數據正確顯示
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            
            # 解析JSON字符串為Python對象
            try:
                if isinstance(metadata.get("keywords"), str):
                    metadata["keywords"] = json.loads(metadata["keywords"])
                if isinstance(metadata.get("law_references"), str):
                    metadata["law_references"] = json.loads(metadata["law_references"])
                if isinstance(metadata.get("options"), str):
                    metadata["options"] = json.loads(metadata["options"])
            except:
                pass  # 如果解析失敗，保持原樣
                
            formatted_results.append({
                "content": result["content"],
                "metadata": metadata,
                "distance": result.get("distance", 0)
            })
        
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量搜索失敗: {str(e)}")

@app.get("/legal/exam_points")
async def get_all_exam_points():
    """
    獲取所有可用的考點列表
    """
    try:
        all_questions = vector_store.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return {"exam_points": [], "total": 0}
            
        # 提取所有考點並去重
        exam_points = set()
        for metadata in all_questions["metadatas"]:
            exam_point = metadata.get("exam_point", "")
            if exam_point and len(exam_point) > 0:
                exam_points.add(exam_point)
        
        # 將集合轉為列表並排序
        sorted_exam_points = sorted(list(exam_points))
        
        return {
            "exam_points": sorted_exam_points,
            "total": len(sorted_exam_points)
        }
    except Exception as e:
        print(f"獲取考點列表時發生錯誤: {str(e)}")
        # 如果是數據庫結構問題，返回空列表而不是錯誤
        if "no such column" in str(e):
            print("檢測到數據庫結構問題，返回空考點列表")
            return {"exam_points": [], "total": 0, "note": "數據庫結構不兼容，請重置向量庫"}
        raise HTTPException(status_code=500, detail=f"獲取考點列表時發生錯誤: {str(e)}")

@app.get("/legal/law_references")
async def get_all_law_references():
    """
    獲取所有可用的法條參考列表
    """
    try:
        all_questions = vector_store.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return {"law_references": [], "total": 0}
            
        # 提取所有法條並去重
        law_refs = set()
        for metadata in all_questions["metadatas"]:
            try:
                refs = json.loads(metadata.get("law_references", "[]"))
                if isinstance(refs, list):
                    for ref in refs:
                        if ref and len(ref) > 0:
                            law_refs.add(ref)
            except:
                continue
        
        # 將集合轉為列表並排序
        sorted_law_refs = sorted(list(law_refs))
        
        return {
            "law_references": sorted_law_refs,
            "total": len(sorted_law_refs)
        }
    except Exception as e:
        print(f"獲取法條列表時發生錯誤: {str(e)}")
        # 如果是數據庫結構問題，返回空列表而不是錯誤
        if "no such column" in str(e):
            print("檢測到數據庫結構問題，返回空法條列表")
            return {"law_references": [], "total": 0, "note": "數據庫結構不兼容，請重置向量庫"}
        raise HTTPException(status_code=500, detail=f"獲取法條列表時發生錯誤: {str(e)}")

@app.get("/legal/keywords")
async def get_all_keywords():
    """
    獲取所有可用的關鍵字列表
    """
    try:
        all_questions = vector_store.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return {"keywords": [], "total": 0}
            
        # 提取所有關鍵字並去重
        all_keywords = set()
        for metadata in all_questions["metadatas"]:
            try:
                keywords = json.loads(metadata.get("keywords", "[]"))
                if isinstance(keywords, list):
                    for kw in keywords:
                        if kw and len(kw) > 0:
                            all_keywords.add(kw)
            except:
                continue
        
        # 將集合轉為列表並排序
        sorted_keywords = sorted(list(all_keywords))
        
        return {
            "keywords": sorted_keywords,
            "total": len(sorted_keywords)
        }
    except Exception as e:
        print(f"獲取關鍵字列表時發生錯誤: {str(e)}")
        # 如果是數據庫結構問題，返回空列表而不是錯誤
        if "no such column" in str(e):
            print("檢測到數據庫結構問題，返回空關鍵字列表")
            return {"keywords": [], "total": 0, "note": "數據庫結構不兼容，請重置向量庫"}
        raise HTTPException(status_code=500, detail=f"獲取關鍵字列表時發生錯誤: {str(e)}")

@app.get("/legal/question_types")
async def get_all_question_types():
    """
    獲取所有可用的題目類型列表
    """
    try:
        question_types = vector_store.get_all_question_types()
        
        return {
            "question_types": question_types,
            "total": len(question_types)
        }
    except Exception as e:
        print(f"獲取題目類型列表時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取題目類型列表時發生錯誤: {str(e)}")

@app.get("/legal/questions_by_type/{question_type}")
async def get_questions_by_type(
    question_type: str,
    limit: int = 10
):
    """
    獲取指定類型的題目
    
    Args:
        question_type: 題目類型（如：單選題、多選題、簡答題等）
        limit: 返回結果數量
    """
    try:
        results = vector_store.search_by_question_type(question_type, limit)
        
        # 處理返回結果，確保數據正確顯示
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            
            # 解析JSON字符串為Python對象
            try:
                if isinstance(metadata.get("keywords"), str):
                    metadata["keywords"] = json.loads(metadata["keywords"])
                if isinstance(metadata.get("law_references"), str):
                    metadata["law_references"] = json.loads(metadata["law_references"])
                if isinstance(metadata.get("options"), str):
                    metadata["options"] = json.loads(metadata["options"])
            except:
                # 如果解析失敗，確保字段有默認值
                if "keywords" not in metadata or metadata["keywords"] is None:
                    metadata["keywords"] = []
                if "law_references" not in metadata or metadata["law_references"] is None:
                    metadata["law_references"] = []
                if "options" not in metadata or metadata["options"] is None:
                    metadata["options"] = {}
                
            formatted_results.append({
                "content": result["content"],
                "metadata": metadata,
                "distance": result.get("distance", 0)
            })
        
        return {
            "question_type": question_type,
            "results": formatted_results,
            "total": len(formatted_results)
        }
    except Exception as e:
        print(f"獲取{question_type}題目時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取{question_type}題目時發生錯誤: {str(e)}")

# 新增API端點：重置向量庫
@app.post("/admin/reset-vector-db")
async def reset_vector_database():
    """
    重置向量數據庫（危險操作，會清空所有向量數據）
    只能由管理員調用
    """
    try:
        success = vector_store.reset_database()
        if success:
            return {"message": "向量庫已成功重置"}
        else:
            raise HTTPException(status_code=500, detail="重置向量庫失敗")
    except Exception as e:
        print(f"重置向量庫時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置向量庫時發生錯誤: {str(e)}")

@app.get("/statistics/metadata")
async def get_metadata_statistics(
    metadata_type: str = Query(..., description="要統計的元數據類型: exam_point, keyword, law_reference"),
    limit: int = Query(20, description="返回結果數量上限"),
    min_count: int = Query(1, description="最小出現次數")
):
    """
    獲取題庫中元數據的統計信息，按出現頻率排序
    
    Args:
        metadata_type: 元數據類型（exam_point=考點, keyword=關鍵詞, law_reference=法條引用）
        limit: 返回的結果數量上限
        min_count: 最小出現次數
    """
    try:
        result = {}
        
        # 獲取所有題目
        all_questions = vector_store.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return {"items": [], "total": 0}
        
        # 統計各元數據出現頻率
        counter = {}
        total_questions = len(all_questions["metadatas"])
        
        for metadata in all_questions["metadatas"]:
            if metadata_type == "exam_point":
                item = metadata.get("exam_point", "")
                if item and len(item) > 0:
                    counter[item] = counter.get(item, 0) + 1
            
            elif metadata_type == "keyword":
                try:
                    keywords = json.loads(metadata.get("keywords", "[]"))
                    if isinstance(keywords, list):
                        for kw in keywords:
                            if kw and len(kw) > 0:
                                counter[kw] = counter.get(kw, 0) + 1
                except:
                    continue
            
            elif metadata_type == "law_reference":
                try:
                    refs = json.loads(metadata.get("law_references", "[]"))
                    if isinstance(refs, list):
                        for ref in refs:
                            if ref and len(ref) > 0:
                                counter[ref] = counter.get(ref, 0) + 1
                except:
                    continue
        
        # 過濾低於最小出現次數的項目
        filtered_items = {k: v for k, v in counter.items() if v >= min_count}
        
        # 按出現頻率排序
        sorted_items = sorted(filtered_items.items(), key=lambda x: x[1], reverse=True)
        
        # 限制返回數量
        sorted_items = sorted_items[:limit]
        
        # 計算百分比
        stats = []
        for item, count in sorted_items:
            stats.append({
                "item": item,
                "count": count,
                "percentage": round(count / total_questions * 100, 2)
            })
        
        return {
            "items": stats,
            "total_distinct": len(filtered_items),
            "total_questions": total_questions,
            "metadata_type": metadata_type
        }
    
    except Exception as e:
        print(f"獲取元數據統計時發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"獲取元數據統計時發生錯誤: {str(e)}")

@app.post("/admin/process-exam-points")
async def admin_process_exam_points():
    """
    管理員API：處理並優化向量庫中的考點，包括：
    1. 拆分復合考點為多個獨立考點
    2. 標準化考點格式
    3. 更新題目的考點和關鍵詞
    4. 建立考點層級結構
    
    警告：此操作將修改向量庫中的數據，建議先備份
    """
    try:
        process_and_update_vector_store()
        return {"status": "success", "message": "考點處理成功完成"}
    except Exception as e:
        error_message = f"處理考點時發生錯誤：{str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/admin/split-exam-point")
async def admin_split_exam_point(exam_point: str = Form(...)):
    """
    拆分單個考點為多個獨立考點
    
    Args:
        exam_point: 要拆分的考點
    
    Returns:
        拆分後的考點列表
    """
    try:
        formatted = format_exam_point(exam_point)
        result = split_exam_point(formatted)
        return {"original": exam_point, "split_results": result}
    except Exception as e:
        error_message = f"拆分考點時發生錯誤：{str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/upload")
async def upload_form():
    return {"message": "請使用POST方法上傳文件"}

@app.get("/statistics/metadata_by_year")
async def get_metadata_statistics_by_year(
    metadata_type: str = Query(..., description="要統計的元數據類型: exam_point, keyword, law_reference"),
    limit: int = Query(20, description="返回結果數量上限"),
    min_count: int = Query(1, description="最小出現次數")
):
    """
    獲取元數據按年份的統計信息
    """
    try:
        # 從向量存儲中獲取所有題目
        all_questions = vector_store.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return {"stats": [], "total_questions": 0, "metadata_type": metadata_type}
            
        total_questions = len(all_questions["metadatas"])
        
        # 初始化計數器
        year_counter = {}
        
        for metadata in all_questions["metadatas"]:
            # 從考試名稱中提取年份
            exam_name = metadata.get("exam_name", "")
            year = exam_name[:3] if exam_name else "未知"  # 假設年份是考試名稱的前3個字符
            
            if metadata_type == "exam_point":
                item = metadata.get("exam_point", "")
                if item and len(item) > 0:
                    if year not in year_counter:
                        year_counter[year] = {}
                    year_counter[year][item] = year_counter[year].get(item, 0) + 1
            
            elif metadata_type == "keyword":
                try:
                    keywords = json.loads(metadata.get("keywords", "[]"))
                    if isinstance(keywords, list):
                        for kw in keywords:
                            if kw and len(kw) > 0:
                                if year not in year_counter:
                                    year_counter[year] = {}
                                year_counter[year][kw] = year_counter[year].get(kw, 0) + 1
                except:
                    continue
            
            elif metadata_type == "law_reference":
                try:
                    refs = json.loads(metadata.get("law_references", "[]"))
                    if isinstance(refs, list):
                        for ref in refs:
                            if ref and len(ref) > 0:
                                if year not in year_counter:
                                    year_counter[year] = {}
                                year_counter[year][ref] = year_counter[year].get(ref, 0) + 1
                except:
                    continue
        
        # 處理統計結果
        stats = []
        for year, items in year_counter.items():
            # 過濾低於最小出現次數的項目
            filtered_items = {k: v for k, v in items.items() if v >= min_count}
            
            # 按出現頻率排序
            sorted_items = sorted(filtered_items.items(), key=lambda x: x[1], reverse=True)
            
            # 限制返回數量
            sorted_items = sorted_items[:limit]
            
            # 計算百分比
            year_stats = []
            for item, count in sorted_items:
                year_stats.append({
                    "item": item,
                    "count": count,
                    "percentage": round(count / total_questions * 100, 2)
                })
            
            stats.append({
                "year": year,
                "items": year_stats,
                "total_items": len(filtered_items)
            })
        
        return {
            "stats": stats,
            "total_questions": total_questions,
            "metadata_type": metadata_type
        }
        
    except Exception as e:
        print(f"獲取按年份統計數據時發生錯誤: {str(e)}")
        raise HTTPException(status_code=500, detail=f"獲取按年份統計數據時發生錯誤: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)