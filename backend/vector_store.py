import os
import shutil
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json
from typing import List, Dict
import random
import uuid
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

# 載入環境變數
load_dotenv()

# 模型和配額配置
EMBEDDING_CONFIG = {
    "default_model": "gemini-embedding-exp",  # 默認模型
    "fallback_model": "all-MiniLM-L6-v2",     # 本地備用模型
    "batch_size": 8,                          # 批量大小
    "embedding_dimension": 768                # 嵌入維度
}

# 自定義Gemini嵌入函數類
class GeminiEmbeddingFunction:
    def __init__(self, model_name="gemini-embedding-exp", batch_size=8):
        """
        初始化Gemini嵌入函數，支持配置和自動降級
        
        Args:
            model_name: Gemini嵌入模型名稱
            batch_size: 批處理大小，每批處理的文本數量
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 獲取API密鑰
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        
        genai.configure(api_key=api_key)
        
        # 退避策略配置
        self.backoff_config = {
            "initial_wait": 1.0,   # 初始等待時間(秒)
            "max_retries": 3,      # 最大重試次數
            "backoff_factor": 2.0  # 退避倍數
        }
        
        # 初始化本地模型作為備用
        self.local_model = None
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # 檢查CUDA可用性
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            print(f"已加載本地嵌入模型 all-MiniLM-L6-v2 (作為備用) 在 {device} 設備上")
        except Exception as e:
            print(f"無法加載本地嵌入模型: {str(e)}")
        
        # 初始化計數器
        self.api_success = 0
        self.api_failure = 0
        
        print(f"Gemini嵌入模型 '{model_name}' 初始化成功，批次大小：{batch_size}")
    
    def __call__(self, input):
        """
        將文本轉換為嵌入向量
        
        Args:
            input: 要嵌入的文本或文本列表
        
        Returns:
            嵌入向量列表
        """
        if not input:
            return []
        
        # 確保文本是有效的字符串列表
        texts = input if isinstance(input, list) else [input]
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return []
        
        # 使用批處理來獲取嵌入
        return self._process_in_batches(texts)
    
    def _process_in_batches(self, texts):
        """
        批次處理文本嵌入，支持自動重試和降級
        """
        import time
        import numpy as np
        
        # 分批處理
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        
        for i, batch in enumerate(batches):
            print(f"嵌入處理：第 {i+1}/{len(batches)} 批 ({len(batch)} 條文本)")
            
            # 嘗試使用Gemini API
            batch_embeddings = None
            retry_count = 0
            last_error = None
            
            # 重試邏輯
            while retry_count <= self.backoff_config["max_retries"]:
                try:
                    # 嘗試使用Gemini API獲取嵌入
                    batch_embeddings = []
                    for text in batch:
                        # 文本預處理：裁剪長文本
                        if len(text) > 8192:
                            text = text[:8192]
                            
                        # 獲取嵌入
                        result = genai.embed_content(
                            model=self.model_name,
                            content=text,
                            task_type="retrieval_document"  # 使用retrieval_document而非semantic_similarity
                        )
                        
                        # 提取嵌入向量
                        embedding = np.array(result["embedding"])
                        batch_embeddings.append(embedding)
                    
                    # 如果成功，跳出重試循環
                    self.api_success += 1
                    break
                
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    # 檢查是否應該重試
                    if retry_count <= self.backoff_config["max_retries"]:
                        wait_time = self.backoff_config["initial_wait"] * (
                            self.backoff_config["backoff_factor"] ** (retry_count - 1)
                        )
                        print(f"嵌入API錯誤: {str(e)}，等待 {wait_time:.1f} 秒後重試 ({retry_count}/{self.backoff_config['max_retries']})")
                        time.sleep(wait_time)
                    else:
                        print(f"達到最大重試次數，嵌入API錯誤: {str(e)}")
                        self.api_failure += 1
            
            # 如果API調用失敗，使用本地模型
            if batch_embeddings is None:
                print("API調用失敗，使用本地嵌入模型作為備用")
                if self.local_model:
                    try:
                        batch_embeddings = self.local_model.encode(batch)
                    except Exception as local_error:
                        print(f"本地模型嵌入失敗: {str(local_error)}")
                        # 使用零向量作為最後的備用選項
                        batch_embeddings = [np.zeros(768) for _ in batch]
                else:
                    print("無本地模型可用，使用零向量")
                    # 無本地模型時，使用零向量
                    batch_embeddings = [np.zeros(768) for _ in batch]
            
            # 添加批次結果到總結果
            all_embeddings.extend(batch_embeddings)
            
            # 批次間添加短暫延遲，避免超過API速率限制
            if i < len(batches) - 1:
                time.sleep(1.5)
        
        return all_embeddings

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

        # 初始化策略
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """優化的向量存儲初始化方法，支持多種初始化策略和降級方案"""
        success = False
        error_messages = []
        
        # 檢查資料庫文件是否存在，如果存在且為空目錄但初始化失敗，可能是結構問題
        db_file = os.path.join(self.persist_path, "chroma.sqlite3")
        if not os.path.exists(db_file) and os.path.exists(self.persist_path) and len(os.listdir(self.persist_path)) == 0:
            print(f"檢測到向量庫目錄存在但無數據庫文件，嘗試強制重置...")
            try:
                shutil.rmtree(self.persist_path)
                os.makedirs(self.persist_path, exist_ok=True, mode=0o777)
                print(f"已強制重置向量庫目錄: {self.persist_path}")
            except Exception as e:
                print(f"強制重置向量庫時出錯: {str(e)}")
        
        # 第一階段：嘗試使用Gemini嵌入模型初始化
        try:
            print("嘗試使用Gemini嵌入模型初始化向量庫...")
            self.client = chromadb.PersistentClient(path=self.persist_path)
            
            # 使用優化過的Gemini嵌入函數
            self.embedding_function = GeminiEmbeddingFunction(
                model_name=EMBEDDING_CONFIG["default_model"],
                batch_size=EMBEDDING_CONFIG["batch_size"]
            )
            
            # 創建或獲取集合，使用改進的索引設置
            self.collection = self.client.get_or_create_collection(
                name="exam_questions",
                embedding_function=self.embedding_function,
                metadata={
                    "hnsw:space": "cosine",          # 使用余弦相似度
                    "hnsw:construction_ef": 100,     # 建立索引時的擴展因子
                    "hnsw:search_ef": 100,           # 搜索時的擴展因子
                    "hnsw:M": 16,                    # 每個節點的最大連接數
                    "hnsw:num_threads": 4            # 使用多線程構建索引
                }
            )
            success = True
            print("使用Gemini嵌入模型成功初始化向量庫")
        except Exception as e:
            error_message = f"使用Gemini嵌入模型初始化失敗: {str(e)}"
            print(error_message)
            error_messages.append(error_message)
            
            # 如果發現是數據庫結構錯誤，立即嘗試重置
            if "no such column" in str(e) or "readonly database" in str(e):
                print("檢測到數據庫結構錯誤，嘗試強制刪除並重建...")
                try:
                    # 徹底刪除目錄
                    if os.path.exists(self.persist_path):
                        shutil.rmtree(self.persist_path)
                    # 使用系統命令確保完全刪除    
                    os.system(f"rm -rf {self.persist_path}")
                    # 重新創建目錄
                    os.makedirs(self.persist_path, exist_ok=True, mode=0o777)
                    # 設置寬鬆權限確保可寫
                    os.system(f"chmod -R 777 {self.persist_path}")
                    print(f"已強制刪除並重建向量庫目錄: {self.persist_path}")
                    
                    # 重新嘗試初始化
                    try:
                        self.client = chromadb.PersistentClient(path=self.persist_path)
                        self.embedding_function = GeminiEmbeddingFunction(
                            model_name=EMBEDDING_CONFIG["default_model"],
                            batch_size=EMBEDDING_CONFIG["batch_size"]
                        )
                        self.collection = self.client.get_or_create_collection(
                            name="exam_questions",
                            embedding_function=self.embedding_function,
                            metadata={"hnsw:space": "cosine"}
                        )
                        success = True
                        print("強制重置後成功初始化向量庫")
                        return  # 成功初始化後直接返回
                    except Exception as reset_e:
                        print(f"重置後仍無法初始化: {str(reset_e)}")
                except Exception as del_e:
                    print(f"刪除向量庫目錄時出錯: {str(del_e)}")
        
        # 第二階段：如果使用Gemini失敗，嘗試使用SentenceTransformer
        if not success:
            try:
                print("嘗試使用SentenceTransformer嵌入模型初始化向量庫...")
                self.client = chromadb.PersistentClient(path=self.persist_path)
                
                # 使用SentenceTransformer嵌入函數，添加相容層以適應新接口
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(EMBEDDING_CONFIG["fallback_model"])
                
                # 創建一個包裝器，確保接口一致性
                class STWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def __call__(self, input):
                        # 保持與新的ChromaDB接口兼容
                        texts = input if isinstance(input, list) else [input]
                        return self.model.encode(texts)
                
                self.embedding_function = STWrapper(model)
                
                # 創建或獲取集合，使用基本設置
                self.collection = self.client.get_or_create_collection(
                    name="exam_questions",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}  # 使用簡化設置
                )
                success = True
                print("使用SentenceTransformer嵌入模型成功初始化向量庫")
            except Exception as e:
                error_message = f"使用SentenceTransformer初始化失敗: {str(e)}"
                print(error_message)
                error_messages.append(error_message)
        
        # 第三階段：如果持久化存儲有問題，嘗試重置數據庫
        if not success and any(x in str(error_messages) for x in ["no such column", "readonly database", "permission denied"]):
            try:
                print("檢測到數據庫結構變更或權限問題，嘗試完全重置...")
                # 強制刪除並重新創建目錄
                if os.path.exists(self.persist_path):
                    shutil.rmtree(self.persist_path)
                # 使用系統命令確保完全刪除    
                os.system(f"rm -rf {self.persist_path}")
                # 重新創建目錄
                os.makedirs(self.persist_path, exist_ok=True, mode=0o777)
                # 設置寬鬆權限確保可寫
                os.system(f"chmod -R 777 {self.persist_path}")
                
                # 重新嘗試初始化
                self.client = chromadb.PersistentClient(path=self.persist_path)
                
                # 優先使用SentenceTransformer作為重置後的模型（更穩定）
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(EMBEDDING_CONFIG["fallback_model"])
                
                # 創建相同的包裝器
                class STWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def __call__(self, input):
                        texts = input if isinstance(input, list) else [input]
                        return self.model.encode(texts)
                
                self.embedding_function = STWrapper(model)
                
                # 簡化設置
                self.collection = self.client.get_or_create_collection(
                    name="exam_questions",
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                success = True
                print("向量庫重置後成功初始化")
            except Exception as reset_error:
                error_message = f"重置後初始化失敗: {str(reset_error)}"
                print(error_message)
                error_messages.append(error_message)
        
        # 第四階段：最終備用選項 - 使用內存模式
        if not success:
            try:
                print("嘗試使用內存模式作為最終備用選項...")
                self.client = chromadb.Client()  # 內存模式
                
                # 使用SentenceTransformer（最穩定的選項）
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(EMBEDDING_CONFIG["fallback_model"])
                
                # 使用相同的包裝器
                class STWrapper:
                    def __init__(self, model):
                        self.model = model
                    
                    def __call__(self, input):
                        texts = input if isinstance(input, list) else [input]
                        return self.model.encode(texts)
                
                self.embedding_function = STWrapper(model)
                
                # 使用基本配置
                self.collection = self.client.get_or_create_collection(
                    name="exam_questions",
                    embedding_function=self.embedding_function
                )
                success = True
                print("⚠️ 成功使用內存模式初始化向量庫（注意：數據不會持久化）")
            except Exception as mem_error:
                error_message = f"內存模式初始化失敗: {str(mem_error)}"
                print(error_message)
                error_messages.append(error_message)
        
        # 如果所有嘗試都失敗，拋出異常
        if not success:
            error_details = "\n".join(error_messages)
            raise RuntimeError(f"無法初始化向量庫，所有嘗試均失敗：\n{error_details}")
        
        # 輸出集合大小
        try:
            count = self.collection.count()
            print(f"向量庫初始化完成，當前共有 {count} 個文檔")
        except:
            print("向量庫初始化完成，但無法獲取文檔數量")

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
        """刪除特定考試名稱的所有題目
        
        支持新版和舊版ChromaDB的API結構，
        並增加錯誤處理以提高穩定性
        
        Args:
            exam_name: 要刪除的考試名稱
        """
        try:
            # 方法1: 使用where參數查詢（新版本ChromaDB方式）
            try:
                results = self.collection.get(where={"exam_name": exam_name})
                if results and results.get("ids") and len(results["ids"]) > 0:
                    print(f"找到 {len(results['ids'])} 個題目將被刪除")
                    self.collection.delete(ids=results["ids"])
                    print(f"已成功刪除考試 '{exam_name}' 的所有題目")
                    return
            except Exception as e:
                if "no such column: collections.topic" in str(e):
                    print(f"使用新版API刪除失敗，嘗試使用替代方法: {str(e)}")
                else:
                    raise
            
            # 方法2: 獲取所有題目並篩選（舊版本兼容方式）
            try:
                all_results = self.collection.get()
                if not all_results or not all_results.get("ids"):
                    print("向量庫為空或無法獲取內容")
                    return
                    
                # 找出要刪除的ID
                ids_to_delete = []
                for i, metadata in enumerate(all_results["metadatas"]):
                    if metadata.get("exam_name") == exam_name:
                        ids_to_delete.append(all_results["ids"][i])
                
                if ids_to_delete:
                    print(f"找到 {len(ids_to_delete)} 個題目將被刪除 (替代方法)")
                    self.collection.delete(ids=ids_to_delete)
                    print(f"已成功刪除考試 '{exam_name}' 的所有題目")
                else:
                    print(f"未找到考試 '{exam_name}' 的題目")
            except Exception as e:
                print(f"替代刪除方法也失敗: {str(e)}")
                raise
            
        except Exception as e:
            print(f"刪除題目時發生錯誤: {str(e)}")
            # 如果是ChromaDB結構錯誤，提供更明確的錯誤信息
            if "no such column: collections.topic" in str(e):
                print("檢測到ChromaDB結構變更，建議重置向量庫: python reset_vector_db.py")
            raise

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
        """根據題目類型搜索（完全匹配）"""
        try:
            # 使用元數據過濾搜索
            results = self.collection.query(
                query_texts=[""],  # 空查詢文本
                where={"type": question_type},
                n_results=n_results
            )
            
            return self._format_search_results(results)
        except Exception as e:
            print(f"根據題目類型搜索失敗: {str(e)}")
            return []

    def search_by_exam_name(self, exam_name: str, n_results: int = 100) -> List[Dict]:
        """根據考試名稱搜索題目"""
        try:
            # 使用元數據過濾搜索
            results = self.collection.query(
                query_texts=[""],  # 空查詢文本
                where={"exam_name": exam_name},
                n_results=n_results
            )
            
            return self._format_search_results(results)
        except Exception as e:
            print(f"根據考試名稱搜索失敗: {str(e)}")
            return []

    def get_all_question_types(self) -> List[str]:
        """獲取所有可用的題目類型"""
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

    def reset_database(self):
        """
        重置向量數據庫 - 完全刪除並重建集合
        警告：此操作將刪除所有向量數據
        """
        try:
            print("開始重置向量數據庫...")
            # 嘗試刪除舊集合
            try:
                self.client.delete_collection("exam_questions")
                print("已刪除現有集合")
            except Exception as e:
                print(f"無法刪除集合(可能不存在): {str(e)}")
            
            # 建立新集合
            self.collection = self.client.create_collection(
                name="exam_questions",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            # 嘗試清理持久化目錄中的數據文件
            if os.path.exists(self.persist_path):
                try:
                    chroma_db_path = os.path.join(self.persist_path, "chroma.sqlite3")
                    if os.path.exists(chroma_db_path):
                        os.remove(chroma_db_path)
                        print(f"已刪除數據庫文件: {chroma_db_path}")
                except Exception as e:
                    print(f"清理數據庫文件失敗: {str(e)}")
            
            print("向量數據庫重置完成")
            return True
        except Exception as e:
            print(f"重置向量數據庫失敗: {str(e)}")
            return False

    def _format_search_results(self, results):
        if not results["documents"]:
            return []
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        return formatted_results

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