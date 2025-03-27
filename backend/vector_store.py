import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json
from typing import List, Dict
import random
import uuid

class VectorStore:
    def __init__(self, force_reset=False):
        # 使用不同的目錄路徑解決版本不兼容問題
        home_dir = os.path.expanduser("~")
        # 使用帶有版本號的新目錄名稱，避免與舊版本衝突
        self.persist_path = os.path.join(home_dir, "test_generator_vector_db_v2")

        print(f"向量庫存儲路徑: {self.persist_path}")

        # 檢查是否需要重置向量庫
        if force_reset and os.path.exists(self.persist_path):
            try:
                shutil.rmtree(self.persist_path)
                print(f"已刪除現有的向量庫目錄：{self.persist_path}")
            except Exception as e:
                print(f"刪除向量庫目錄時出錯：{str(e)}")
                # 嘗試使用系統命令強制刪除
                try:
                    os.system(f"rm -rf {self.persist_path}")
                    print(f"已使用系統命令強制刪除向量庫目錄：{self.persist_path}")
                except Exception as cmd_error:
                    print(f"強制刪除失敗：{str(cmd_error)}")

        # 如果目錄不存在，則建立；否則保留舊資料
        if not os.path.exists(self.persist_path):
            try:
                os.makedirs(self.persist_path, exist_ok=True, mode=0o755)  # 設置明確的權限
                print(f"創建新的持久化目錄：{self.persist_path}")
            except Exception as e:
                print(f"創建持久化目錄時出錯：{str(e)}")
                # 嘗試使用系統命令創建
                os.system(f"mkdir -p {self.persist_path} && chmod 755 {self.persist_path}")
        else:
            print(f"使用現有的持久化目錄：{self.persist_path}")

        # 嘗試多種初始化策略
        success = False
        error_messages = []
        
        # 嘗試使用常規方式初始化
        try:
            self.client = chromadb.PersistentClient(path=self.persist_path)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.collection = self.client.get_or_create_collection(
                name="exam_questions",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100, "hnsw:M": 16}
            )
            success = True
            print("向量庫初始化成功")
        except Exception as e:
            error_message = f"標準初始化失敗: {str(e)}"
            print(error_message)
            error_messages.append(error_message)
            
            # 如果是數據庫結構問題，嘗試完全重置
            if "no such column" in str(e) and os.path.exists(self.persist_path):
                try:
                    print("檢測到數據庫結構變更，嘗試完全重置...")
                    shutil.rmtree(self.persist_path)
                    os.makedirs(self.persist_path, exist_ok=True, mode=0o755)
                    print(f"已重新創建向量庫目錄：{self.persist_path}")
                    
                    # 重新嘗試初始化
                    self.client = chromadb.PersistentClient(path=self.persist_path)
                    self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    )
                    self.collection = self.client.get_or_create_collection(
                        name="exam_questions",
                        embedding_function=self.embedding_function,
                        metadata={"hnsw:space": "cosine"}  # 使用簡化的元數據
                    )
                    success = True
                    print("向量庫重置後成功初始化")
                except Exception as reset_error:
                    error_message = f"重置後初始化失敗: {str(reset_error)}"
                    print(error_message)
                    error_messages.append(error_message)

        # 如果前面的方法都失敗，嘗試使用內存模式
        if not success:
            try:
                print("嘗試使用內存模式作為備用選項...")
                self.client = chromadb.Client()  # 內存模式
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
                self.collection = self.client.get_or_create_collection(
                    name="exam_questions",
                    embedding_function=self.embedding_function
                )
                print("成功使用內存模式初始化向量庫（注意：數據不會持久化）")
                success = True
            except Exception as mem_error:
                error_message = f"內存模式初始化失敗: {str(mem_error)}"
                print(error_message)
                error_messages.append(error_message)
        
        # 如果所有嘗試都失敗，則拋出異常
        if not success:
            error_details = "\n".join(error_messages)
            raise RuntimeError(f"無法初始化向量庫，嘗試了多種方法但都失敗:\n{error_details}")

    def add_questions(self, questions: List[Dict], exam_name: str):
        documents = []
        metadatas = []
        ids = []
        
        # 生成時間戳作為批次處理標識符，避免ID重複
        batch_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        for index, question in enumerate(questions):
            # 建立增強文檔，包含考點和關鍵字以提升檢索相關性
            enhanced_document = question["content"]
            
            # 添加考點和關鍵字信息到文檔中，提升檢索相關性
            if "exam_point" in question and question["exam_point"]:
                enhanced_document += f" 考點: {question['exam_point']}"
            
            if "keywords" in question and question["keywords"]:
                keywords_text = ", ".join(question["keywords"])
                enhanced_document += f" 關鍵字: {keywords_text}"
                
            if "law_references" in question and question["law_references"]:
                law_refs_text = ", ".join(question["law_references"])
                enhanced_document += f" 法條: {law_refs_text}"
            
            # 添加題目類型資訊
            if "type" in question and question["type"]:
                enhanced_document += f" 題型: {question['type']}"
            
            documents.append(enhanced_document)
            
            metadata = {
                "exam_name": exam_name,
                "question_number": question["id"],
                "answer": question.get("answer", ""),
                "explanation": question.get("explanation", ""),
                "options": json.dumps(question.get("options", {})),
                "exam_point": question.get("exam_point", ""),
                "keywords": json.dumps(question.get("keywords", [])),
                "law_references": json.dumps(question.get("law_references", [])),
                "type": question.get("type", "單選題"),  # 新增題目類型字段，默認為單選題
                "added_at": datetime.now().isoformat()
            }
            
            # 生成全局唯一ID，結合批次ID、索引和原始題號
            question_id = f"{exam_name}_{batch_id}_{index+1}_{question['id']}"
            metadatas.append(metadata)
            ids.append(question_id)
            
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def search_similar_questions(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        if not results["documents"]:
            return []
        similar_questions = []
        for i in range(len(results["documents"][0])):
            similar_questions.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        return similar_questions

    def generate_exam(self, num_questions: int = 10) -> List[Dict]:
        all_questions = self.collection.get()
        if not all_questions or "documents" not in all_questions or not all_questions["documents"]:
            return []
        num_available = len(all_questions["documents"])
        num_to_sample = min(num_questions, num_available)
        sampled_indices = random.sample(range(num_available), num_to_sample)
        exam_questions = []
        for idx in sampled_indices:
            exam_questions.append({
                "content": all_questions["documents"][idx],
                "metadata": all_questions["metadatas"][idx]
            })
        return exam_questions

    def delete_exam_questions(self, exam_name: str):
        results = self.collection.get(where={"exam_name": exam_name})
        if results and results.get("ids"):
            self.collection.delete(ids=results["ids"])

    # 增強：根據關鍵字搜索題目
    def search_by_keyword(self, keyword: str, n_results: int = 10) -> List[Dict]:
        """
        根據關鍵字搜索題目，支援多種搜索方式
        
        參數:
            keyword: 要搜索的關鍵字
            n_results: 返回結果數量
            
        返回值:
            符合條件的題目列表
        """
        all_questions = self.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return []
            
        matching_questions = []
        
        # 第一步：檢查題目的關鍵字欄位
        for i, metadata in enumerate(all_questions["metadatas"]):
            # 解析關鍵字列表
            try:
                keywords = json.loads(metadata.get("keywords", "[]"))
                if not isinstance(keywords, list):
                    keywords = []
            except:
                keywords = []
            
            # 檢查是否有完全匹配的關鍵字
            exact_match = False
            for kw in keywords:
                if keyword.lower() == kw.lower():
                    exact_match = True
                    break
                    
            # 檢查是否有部分匹配的關鍵字
            partial_match = False
            if not exact_match:
                for kw in keywords:
                    if keyword.lower() in kw.lower() or kw.lower() in keyword.lower():
                        partial_match = True
                        break
            
            # 檢查題目內容是否包含關鍵字
            content_match = False
            content = all_questions["documents"][i]
            if keyword.lower() in content.lower():
                content_match = True
            
            # 根據匹配程度分配距離分數
            if exact_match:
                matching_questions.append({
                    "content": content,
                    "metadata": metadata,
                    "distance": 0  # 完全匹配，距離為0
                })
            elif partial_match:
                matching_questions.append({
                    "content": content,
                    "metadata": metadata,
                    "distance": 0.3  # 部分匹配，距離較小
                })
            elif content_match:
                matching_questions.append({
                    "content": content,
                    "metadata": metadata,
                    "distance": 0.6  # 內容匹配，距離適中
                })
        
        # 如果直接匹配找到了足夠的結果，按匹配程度排序並返回
        if len(matching_questions) >= n_results:
            sorted_questions = sorted(matching_questions, key=lambda q: q["distance"])
            return sorted_questions[:n_results]
            
        # 否則使用語義搜索補充結果
        semantic_results = self.search_similar_questions(keyword, n_results=n_results-len(matching_questions))
        
        # 合併結果（去重）
        result_ids = {q["metadata"].get("id") for q in matching_questions}
        for q in semantic_results:
            if q["metadata"].get("id") not in result_ids:
                matching_questions.append(q)
                if len(matching_questions) >= n_results:
                    break
        
        # 按照匹配程度排序
        sorted_results = sorted(matching_questions, key=lambda q: q["distance"])
        return sorted_results

    # 增強：根據考點搜索題目
    def search_by_exam_point(self, exam_point: str, n_results: int = 10) -> List[Dict]:
        """
        根據考點搜索題目，支援模糊匹配和語義搜索
        
        參數:
            exam_point: 要搜索的考點關鍵字
            n_results: 返回結果數量
            
        返回值:
            符合條件的題目列表
        """
        all_questions = self.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return []
            
        matching_questions = []
        
        # 進行考點匹配（支持模糊匹配）
        for i, metadata in enumerate(all_questions["metadatas"]):
            exam_point_value = metadata.get("exam_point", "")
            
            # 完全匹配
            if exam_point_value.lower() == exam_point.lower():
                matching_questions.append({
                    "content": all_questions["documents"][i],
                    "metadata": metadata,
                    "distance": 0  # 完全匹配，距離為0
                })
            # 部分匹配
            elif exam_point.lower() in exam_point_value.lower():
                matching_questions.append({
                    "content": all_questions["documents"][i],
                    "metadata": metadata,
                    "distance": 0.3  # 部分匹配，距離較小
                })
        
        # 如果直接匹配找到了足夠的結果，按匹配程度排序並返回
        if len(matching_questions) >= n_results:
            sorted_questions = sorted(matching_questions, key=lambda q: q["distance"])
            return sorted_questions[:n_results]
        
        # 否則使用語義搜索補充結果
        semantic_results = self.search_similar_questions(f"考點: {exam_point}", n_results=n_results-len(matching_questions))
        
        # 合併結果（去重）
        result_ids = {q["metadata"].get("id") for q in matching_questions}
        for q in semantic_results:
            if q["metadata"].get("id") not in result_ids:
                matching_questions.append(q)
                if len(matching_questions) >= n_results:
                    break
                    
        # 按照匹配程度排序
        sorted_results = sorted(matching_questions, key=lambda q: q["distance"])
        return sorted_results

    # 增強：根據法條搜索題目
    def search_by_law_reference(self, law_ref: str, n_results: int = 10) -> List[Dict]:
        """
        根據法條搜索題目，支援多種搜索方式
        
        參數:
            law_ref: 要搜索的法條關鍵字，例如「憲法第8條」或「刑法」
            n_results: 返回結果數量
            
        返回值:
            符合條件的題目列表
        """
        all_questions = self.collection.get()
        if not all_questions or "metadatas" not in all_questions:
            return []
            
        # 進行精確匹配
        matching_questions = []
        for i, metadata in enumerate(all_questions["metadatas"]):
            law_refs = json.loads(metadata.get("law_references", "[]"))
            
            # 檢查是否有完全匹配的法條
            exact_match = False
            for ref in law_refs:
                if law_ref.lower() == ref.lower():
                    exact_match = True
                    break
                    
            # 檢查是否有部分匹配的法條
            partial_match = False
            if not exact_match:
                for ref in law_refs:
                    if law_ref.lower() in ref.lower():
                        partial_match = True
                        break
            
            # 根據匹配程度分配距離分數
            if exact_match:
                matching_questions.append({
                    "content": all_questions["documents"][i],
                    "metadata": metadata,
                    "distance": 0  # 完全匹配，距離為0
                })
            elif partial_match:
                matching_questions.append({
                    "content": all_questions["documents"][i],
                    "metadata": metadata,
                    "distance": 0.3  # 部分匹配，距離較小
                })
        
        # 如果精確匹配找到了足夠的結果，按照匹配程度排序並返回
        if len(matching_questions) >= n_results:
            sorted_questions = sorted(matching_questions, key=lambda q: q["distance"])
            return sorted_questions[:n_results]
            
        # 否則使用語義搜索補充結果
        semantic_results = self.search_similar_questions(f"法條: {law_ref}", n_results=n_results-len(matching_questions))
        
        # 合併結果（去重）
        result_ids = {q["metadata"].get("id") for q in matching_questions}
        for q in semantic_results:
            if q["metadata"].get("id") not in result_ids:
                matching_questions.append(q)
                if len(matching_questions) >= n_results:
                    break
        
        # 按照匹配程度排序
        sorted_results = sorted(matching_questions, key=lambda q: q["distance"])
        return sorted_results

    # 新增功能：根據題目類型搜索
    def search_by_question_type(self, question_type: str, n_results: int = 10) -> List[Dict]:
        """
        根據題目類型搜索題目
        
        參數:
            question_type: 要搜索的題目類型（如「單選題」、「多選題」等）
            n_results: 返回結果數量
            
        返回值:
            符合條件的題目列表
        """
        # 標準化題目類型，避免因為小寫大寫或空格造成匹配問題
        normalized_type = question_type.strip().lower()
        
        # 嘗試按類型進行精確匹配
        try:
            results = self.collection.get(
                where={"type": {"$eq": question_type}},
                limit=n_results
            )
        except Exception as e:
            print(f"按題型精確匹配搜索失敗: {str(e)}，嘗試獲取全部題目後過濾")
            results = {"ids": [], "documents": [], "metadatas": []}
        
        # 如果沒有找到結果或發生錯誤，獲取所有題目後手動過濾
        if not results["ids"]:
            all_questions = self.collection.get()
            if not all_questions or "metadatas" not in all_questions:
                return []
                
            matching_questions = []
            
            for i, metadata in enumerate(all_questions["metadatas"]):
                q_type = metadata.get("type", "")
                if q_type:  # 确保q_type不是None，避免空值异常
                    q_type = q_type.strip().lower()
                else:
                    q_type = ""
                
                # 精確匹配
                if q_type == normalized_type:
                    matching_questions.append({
                        "content": all_questions["documents"][i],
                        "metadata": metadata,
                        "distance": 0  # 完全匹配，距離為0
                    })
                # 包含匹配（如搜尋"選題"可匹配"單選題"和"多選題"）
                elif normalized_type in q_type or q_type in normalized_type:
                    matching_questions.append({
                        "content": all_questions["documents"][i],
                        "metadata": metadata,
                        "distance": 0.3  # 部分匹配，距離較小
                    })
            
            # 根據匹配度排序
            sorted_questions = sorted(matching_questions, key=lambda q: q["distance"])
            
            # 返回前N個結果
            return sorted_questions[:n_results]
        else:
            # 轉換原始結果為標準格式
            matching_questions = []
            for i in range(len(results["ids"])):
                matching_questions.append({
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i],
                    "distance": 0  # 直接查詢的結果，距離為0
                })
            return matching_questions

    # 獲取所有可用的題目類型
    def get_all_question_types(self) -> List[str]:
        """
        獲取向量庫中所有可用的題目類型
        
        返回值:
            排序後的題目類型列表
        """
        try:
            all_questions = self.collection.get()
            if not all_questions or "metadatas" not in all_questions:
                return []
            
            # 提取所有題型並去重
            question_types = set()
            for metadata in all_questions["metadatas"]:
                q_type = metadata.get("type", "")
                if q_type:
                    question_types.add(q_type)
                
            # 將集合轉為列表並排序
            sorted_types = sorted(list(question_types))
            return sorted_types
        except Exception as e:
            print(f"獲取題目類型時發生錯誤: {str(e)}")
            # 數據庫結構不兼容情況下的備用方案
            if "no such column" in str(e):
                print("檢測到數據庫結構不兼容，使用內存模式獲取題型")
                # 返回預設題型列表
                return ["單選題", "多選題", "簡答題", "申論題", "案例分析題"]
            # 其他錯誤則返回空列表
            return []

    # 增強版的 get_all_keywords 函數，添加錯誤處理
    def get_all_keywords(self) -> List[str]:
        """
        獲取向量庫中所有可用的關鍵字
        
        返回值:
            排序後的關鍵字列表
        """
        try:
            all_questions = self.collection.get()
            if not all_questions or "metadatas" not in all_questions:
                return []
            
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
            return sorted_keywords
        except Exception as e:
            print(f"獲取關鍵字列表時發生錯誤: {str(e)}")
            # 數據庫結構不兼容情況的備用方案
            if "no such column" in str(e):
                print("檢測到數據庫結構不兼容，無法獲取關鍵字列表")
            return []

    # 增強版的 get_all_exam_points 函數
    def get_all_exam_points(self) -> List[str]:
        """
        獲取向量庫中所有可用的考點
        
        返回值:
            排序後的考點列表
        """
        try:
            all_questions = self.collection.get()
            if not all_questions or "metadatas" not in all_questions:
                return []
            
            # 提取所有考點並去重
            exam_points = set()
            for metadata in all_questions["metadatas"]:
                exam_point = metadata.get("exam_point", "")
                if exam_point and len(exam_point) > 0:
                    exam_points.add(exam_point)
            
            # 將集合轉為列表並排序
            sorted_exam_points = sorted(list(exam_points))
            return sorted_exam_points
        except Exception as e:
            print(f"獲取考點列表時發生錯誤: {str(e)}")
            # 數據庫結構不兼容情況的備用方案
            if "no such column" in str(e):
                print("檢測到數據庫結構不兼容，無法獲取考點列表")
            return []

    # 增強版的 get_all_law_references 函數
    def get_all_law_references(self) -> List[str]:
        """
        獲取向量庫中所有可用的法條參考
        
        返回值:
            排序後的法條列表
        """
        try:
            all_questions = self.collection.get()
            if not all_questions or "metadatas" not in all_questions:
                return []
            
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
            return sorted_law_refs
        except Exception as e:
            print(f"獲取法條列表時發生錯誤: {str(e)}")
            # 數據庫結構不兼容情況的備用方案
            if "no such column" in str(e):
                print("檢測到數據庫結構不兼容，無法獲取法條列表")
            return []

# 嘗試初始化向量庫，如果失敗則嘗試重置後重新初始化
try:
    vector_store = VectorStore()
except Exception as e:
    print(f"初始化向量庫時發生錯誤: {str(e)}")
    print("嘗試重置向量庫後重新初始化...")
    try:
        vector_store = VectorStore(force_reset=True)
        print("向量庫重置並成功初始化")
    except Exception as reset_error:
        print(f"重置向量庫後仍然發生錯誤: {str(reset_error)}")
        raise