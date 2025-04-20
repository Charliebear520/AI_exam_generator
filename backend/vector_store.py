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
import logging
import time

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

# 模型和配額配置
EMBEDDING_CONFIG = {
    "default_model": "gemini-2.0-flash",  # 默認模型
    "fallback_model": "text-embedding-3-large",  # OpenAI 備用模型
    "batch_size": 8,  # 批量大小
    "embedding_dimension": 768  # 嵌入維度
}

class GeminiEmbeddingFunction:
    def __init__(self, model_name="gemini-2.0-flash", batch_size=8):
        """
        初始化Gemini嵌入函數，支持配置和自動降級
        
        Args:
            model_name: Gemini嵌入模型名稱
            batch_size: 批處理大小，每批處理的文本數量
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # 獲取API密鑰
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.gemini_api_key and not self.openai_api_key:
            logger.error("未設置API密鑰，請在 .env 檔案中設定 GEMINI_API_KEY 或 OPENAI_API_KEY")
            raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY 或 OPENAI_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            logger.info(f"已配置Gemini API密鑰，前5位: {self.gemini_api_key[:5]}...")
        
        # 退避策略配置
        self.backoff_config = {
            "initial_wait": 1.0,   # 初始等待時間(秒)
            "max_retries": 3,      # 最大重試次數
            "backoff_factor": 2.0  # 退避倍數
        }
        
        logger.info(f"嵌入模型 '{model_name}' 初始化成功，批次大小：{batch_size}")
    
    def __call__(self, input):
        """
        將文本轉換為嵌入向量，支持自動降級到 OpenAI
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
        批次處理文本嵌入，支持自動重試和降級到 OpenAI
        """
        import time
        import numpy as np
        import openai
        
        # 分批處理
        batches = [texts[i:i+self.batch_size] for i in range(0, len(texts), self.batch_size)]
        all_embeddings = []
        
        for i, batch in enumerate(batches):
            logger.info(f"正在處理第 {i+1}/{len(batches)} 批文本，本批包含 {len(batch)} 條文本")
            
            # 嘗試使用Gemini API
            batch_embeddings = None
            retry_count = 0
            last_error = None
            
            # 重試邏輯
            while retry_count <= self.backoff_config["max_retries"]:
                try:
                    if self.gemini_api_key:
                        # 嘗試使用Gemini API獲取嵌入
                        batch_embeddings = []
                        for text in batch:
                            if len(text) > 8192:
                                logger.warning(f"文本長度超過8192字符，將被截斷")
                                text = text[:8192]
                                
                            result = genai.embed_content(
                                model=self.model_name,
                                content=text,
                                task_type="retrieval_document"
                            )
                            embedding = np.array(result["embedding"])
                            batch_embeddings.append(embedding)
                        
                        logger.info(f"第 {i+1} 批文本嵌入成功")
                        break
                    else:
                        raise Exception("Gemini API key not available")
                
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if retry_count <= self.backoff_config["max_retries"]:
                        wait_time = self.backoff_config["initial_wait"] * (
                            self.backoff_config["backoff_factor"] ** (retry_count - 1)
                        )
                        logger.warning(f"Gemini嵌入API錯誤: {str(e)}，將在 {wait_time:.1f} 秒後重試 (第 {retry_count} 次)")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"達到最大重試次數 ({self.backoff_config['max_retries']})，將切換到OpenAI嵌入")
            
            # 如果Gemini API調用失敗，嘗試使用OpenAI
            if batch_embeddings is None and self.openai_api_key:
                try:
                    logger.info("正在使用OpenAI嵌入模型作為備用")
                    client = openai.OpenAI(api_key=self.openai_api_key)
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=batch
                    )
                    batch_embeddings = [np.array(item.embedding) for item in response.data]
                    logger.info("OpenAI嵌入模型處理成功")
                except Exception as e:
                    logger.error(f"OpenAI嵌入失敗: {str(e)}")
                    # 使用零向量作為最後的備用選項
                    batch_embeddings = [np.zeros(768) for _ in batch]
                    logger.warning("使用零向量作為最後的備用選項")
            
            all_embeddings.extend(batch_embeddings)
            
            # 批次間添加短暫延遲
            if i < len(batches) - 1:
                time.sleep(1.5)
        
        return all_embeddings

class VectorStore:
    def __init__(self, force_reset=False):
        # 使用不同的目錄路徑解決版本不兼容問題
        home_dir = os.path.expanduser("~")
        self.persist_path = os.path.join(home_dir, "test_generator_vector_db_v2")
        
        logger.info(f"向量庫存儲路徑: {self.persist_path}")
        
        # 檢查是否需要重置向量庫
        if force_reset:
            self._reset_database()
        
        # 初始化向量庫
        if not self._initialize_vector_store():
            logger.error("向量庫初始化失敗，請檢查錯誤日誌")
            raise Exception("向量庫初始化失敗")
    
    def _reset_database(self):
        """重置向量庫"""
        try:
            if os.path.exists(self.persist_path):
                # 先嘗試正常刪除
                try:
                    shutil.rmtree(self.persist_path)
                    logger.info(f"已刪除現有的向量庫目錄：{self.persist_path}")
                except Exception as e:
                    logger.warning(f"正常刪除失敗，嘗試強制刪除：{str(e)}")
                    # 如果正常刪除失敗，嘗試使用系統命令強制刪除
                    os.system(f"rm -rf {self.persist_path}")
                    logger.info("已使用系統命令強制刪除向量庫目錄")
                
                # 確保目錄被完全刪除
                if os.path.exists(self.persist_path):
                    raise Exception("無法完全刪除向量庫目錄")
                
                # 等待一小段時間確保文件系統更新
                time.sleep(1)
        except Exception as e:
            logger.error(f"重置向量庫時發生錯誤：{str(e)}")
            raise
    
    def _initialize_vector_store(self):
        """初始化向量庫"""
        try:
            # 確保目錄存在
            os.makedirs(self.persist_path, exist_ok=True)
            
            # 初始化 ChromaDB 客戶端
            self.client = chromadb.PersistentClient(path=self.persist_path)
            
            # 檢查集合是否已存在
            try:
                existing_collections = self.client.list_collections()
                for collection in existing_collections:
                    if collection.name == "legal_questions":
                        logger.info("發現已存在的集合，正在刪除...")
                        self.client.delete_collection("legal_questions")
                        logger.info("已刪除現有集合")
                        break
            except Exception as e:
                logger.warning(f"檢查現有集合時發生錯誤：{str(e)}")
            
            # 創建嵌入函數
            embedding_fn = GeminiEmbeddingFunction(
                model_name=EMBEDDING_CONFIG["default_model"],
                batch_size=EMBEDDING_CONFIG["batch_size"]
            )
            
            # 創建集合
            self.collection = self.client.create_collection(
                name="legal_questions",
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("向量庫初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"初始化向量庫失敗: {str(e)}")
            return False

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