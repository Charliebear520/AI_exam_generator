# 測試嵌入函數與ChromaDB的兼容性
import os
import sys
import chromadb
from dotenv import load_dotenv

# 導入向量庫模塊
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vector_store import GeminiEmbeddingFunction, EMBEDDING_CONFIG

# 載入環境變數
load_dotenv()

def test_gemini_embedding():
    """測試Gemini嵌入函數是否符合ChromaDB接口要求"""
    print("=== 開始測試Gemini嵌入函數 ===")
    
    # 測試文本
    test_texts = [
        "民法中的債權讓與",
        "刑法中的正當防衛",
        "行政法中的比例原則"
    ]
    
    # 初始化嵌入函數
    try:
        embedding_function = GeminiEmbeddingFunction(
            model_name=EMBEDDING_CONFIG["default_model"],
            batch_size=EMBEDDING_CONFIG["batch_size"]
        )
        print("✓ 嵌入函數初始化成功")
        
        # 測試單個文本嵌入
        try:
            single_result = embedding_function(test_texts[0])
            print(f"✓ 單個文本嵌入成功，向量維度: {len(single_result)}")
        except Exception as e:
            print(f"× 單個文本嵌入失敗: {str(e)}")
        
        # 測試批量文本嵌入
        try:
            batch_results = embedding_function(test_texts)
            print(f"✓ 批量文本嵌入成功，結果數量: {len(batch_results)}")
        except Exception as e:
            print(f"× 批量文本嵌入失敗: {str(e)}")
        
        # 與ChromaDB集成測試
        try:
            # 使用內存模式客戶端進行測試
            client = chromadb.Client()
            collection = client.create_collection(
                name="test_collection",
                embedding_function=embedding_function
            )
            
            # 添加文檔
            collection.add(
                documents=test_texts,
                metadatas=[{"source": f"test{i}"} for i in range(len(test_texts))],
                ids=[f"id{i}" for i in range(len(test_texts))]
            )
            
            # 查詢測試
            results = collection.query(
                query_texts=["刑法概念"],
                n_results=2
            )
            
            print(f"✓ ChromaDB集成測試成功，查詢結果數: {len(results['documents'][0])}")
            print(f"  查詢結果: {results['documents'][0]}")
            
        except Exception as e:
            print(f"× ChromaDB集成測試失敗: {str(e)}")
            
    except Exception as e:
        print(f"× 嵌入函數初始化失敗: {str(e)}")
    
    print("=== 嵌入函數測試完成 ===")

if __name__ == "__main__":
    test_gemini_embedding() 