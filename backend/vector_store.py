import os
from typing import List, Dict
import random
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime

class VectorStore:
    def __init__(self):
        # 使用 PersistentClient 进行持久化存储
        self.client = chromadb.PersistentClient(
            path="vector_db"
        )
        
        # 使用多语言模型作为嵌入函数
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-mpnet-base-v2"
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name="exam_questions",
            embedding_function=self.embedding_function
        )

    def add_questions(self, questions: List[Dict], exam_name: str):
        """
        将题目添加到向量数据库
        """
        documents = []  # 题目内容
        metadatas = []  # 题目元数据
        ids = []        # 唯一标识符

        for question in questions:
            # 组合题目内容（包括题目和选项）
            content = question["content"]
            if question.get("options"):
                for key, value in question["options"].items():
                    content += f"\n{key}. {value}"

            # 创建元数据
            metadata = {
                "exam_name": exam_name,
                "question_number": question["id"],
                "answer": question.get("answer", ""),
                "explanation": question.get("explanation", ""),
                "added_at": datetime.now().isoformat()
            }

            # 生成唯一ID
            question_id = f"{exam_name}_{question['id']}"

            documents.append(content)
            metadatas.append(metadata)
            ids.append(question_id)

        # 批量添加到向量数据库
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search_similar_questions(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        搜索相似题目
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        if not results["documents"]:  # 避免 IndexError
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
        """
        随机从向量数据库中选择题目生成模拟考试
        """
        # 获取所有题目
        all_questions = self.collection.get()

        if not all_questions or "documents" not in all_questions or not all_questions["documents"]:
            return []

        # 抽取随机题目
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
        """
        删除指定考试的所有题目
        """
        results = self.collection.get(
            where={"exam_name": exam_name}
        )

        if results and results.get("ids"):  # 避免 KeyError 或 NoneType 错误
            self.collection.delete(
                ids=results["ids"]
            )

# 创建全局实例
vector_store = VectorStore()