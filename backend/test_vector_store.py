# filepath: /Users/charliebear/Desktop/code/test_generator/backend/test_vector_store.py
import json
from vector_store import vector_store

def test_vector_store():
    """
    測試向量資料庫的基本功能
    """
    print("開始測試向量資料庫...")

    # 1. 初始化向量資料庫
    try:
        vs = vector_store
        print("✓ 成功初始化向量資料庫")
    except Exception as e:
        print(f"✗ 初始化失敗: {str(e)}")
        return

    # 2. 從 JSON 檔案讀取測試數據
    json_file_path = "/Users/charliebear/Desktop/code/test_generator/backend/downloads/questions_20250314002256.json"  # 替換為你的實際檔案路徑
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            test_questions = data["questions"]  # 假設 JSON 結構是 {"questions": [...]}
        print("✓ 成功讀取 JSON 檔案")
    except Exception as e:
        print(f"✗ 讀取 JSON 檔案失敗: {str(e)}")
        return

    exam_name = data.get("exam", "測試考試")  # 從 JSON 讀取考試名稱，如果沒有則使用預設值

    # 3. 測試添加題目
    try:
        vs.delete_exam_questions(exam_name)  # 刪除現有資料
        vs.add_questions(test_questions, exam_name)
        print("✓ 成功添加測試題目")
    except Exception as e:
        print(f"✗ 添加題目失敗: {str(e)}")
        return

    # 4. 測試搜索相似題目
    try:
        # 使用第一題的部分內容作為搜索查詢
        search_query = test_questions[0]["content"][:50]  # 使用前 50 個字元作為查詢
        results = vs.search_similar_questions(search_query, n_results=2)

        print("\n搜索結果:")
        for i, result in enumerate(results, 1):
            print(f"\n結果 {i}:")
            print(f"相似度距離: {result['distance']}")
            print(f"題目內容: {result['content'][:100]}...")
            print(f"來源考試: {result['metadata']['exam_name']}")
            print(f"題號: {result['metadata']['question_number']}")

        if results:
            print("✓ 成功執行相似題目搜索")
        else:
            print("! 搜索未返回結果")
    except Exception as e:
        print(f"✗ 搜索失敗: {str(e)}")

    # 5. 測試刪除功能
    try:
        vs.delete_exam_questions(exam_name)
        # 驗證刪除是否成功
        results = vs.search_similar_questions("法人", n_results=1)
        if not results:
            print("✓ 成功刪除測試題目")
        else:
            print("! 題目可能未完全刪除")
    except Exception as e:
        print(f"✗ 刪除失敗: {str(e)}")

    print("\n測試完成!")

if __name__ == "__main__":
    test_vector_store()