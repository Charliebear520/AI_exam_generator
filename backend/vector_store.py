import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import json
from typing import List, Dict
import random

class VectorStore:
    def __init__(self):
        # 使用新路徑，避免舊數據干擾
        self.client = chromadb.PersistentClient(path="vector_db_new_model")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"  # 確認使用新模型
        )
        self.collection = self.client.get_or_create_collection(
            name="exam_questions",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100, "hnsw:M": 16}
        )

    def add_questions(self, questions: List[Dict], exam_name: str):
        documents = [q["content"] for q in questions]  # 只嵌入題目主體
        metadatas = []
        ids = []
        for question in questions:
            metadata = {
                "exam_name": exam_name,
                "question_number": question["id"],
                "answer": question.get("answer", ""),
                "explanation": question.get("explanation", ""),
                "options": json.dumps(question.get("options", {})),
                "added_at": datetime.now().isoformat()
            }
            question_id = f"{exam_name}_{question['id']}"
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

vector_store = VectorStore()