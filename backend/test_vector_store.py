import os
import json
from vector_store import vector_store

DOWNLOADS_DIR = os.path.join(os.path.dirname(__file__), "downloads")

def process_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exam_name = data.get("exam", os.path.splitext(os.path.basename(file_path))[0])
    test_questions = data.get("questions", [])
    return exam_name, test_questions

def test_vector_store():
    print("開始測試向量資料庫...")
    try:
        vs = vector_store
        print("✓ 成功初始化向量資料庫")
    except Exception as e:
        print(f"✗ 初始化失敗: {str(e)}")
        return

    json_files = [os.path.join(DOWNLOADS_DIR, file)
                  for file in os.listdir(DOWNLOADS_DIR)
                  if file.endswith('.json')]
    
    if not json_files:
        print("未找到任何 JSON 測試檔案")
        return

    for json_file in json_files:
        print(f"\n處理檔案：{json_file}")
        try:
            exam_name, test_questions = process_json_file(json_file)
            print(f"✓ 成功讀取 JSON 檔案，考試名稱：{exam_name}，題目數：{len(test_questions)}")
        except Exception as e:
            print(f"✗ 讀取 JSON 檔案失敗（{json_file}）：{str(e)}")
            continue

        try:
            vs.delete_exam_questions(exam_name)
            vs.add_questions(test_questions, exam_name)
            print(f"✓ 成功添加測試題目，資料庫總數: {vs.collection.count()}")
        except Exception as e:
            print(f"✗ 添加題目失敗: {str(e)}")
            continue

        try:
            if test_questions:
                search_query = test_questions[0]["content"]
                print(f"查詢內容: {search_query}")  # 調試查詢內容
                results = vs.search_similar_questions(search_query, n_results=5)
                print("\n搜索結果（使用完整題目查詢）:")
                for i, result in enumerate(results, 1):
                    print(f"\n結果 {i}:")
                    print(f"相似度距離: {result['distance']}")
                    print(f"題目內容: {result['content'][:100]}...")
                    print(f"來源考試: {result['metadata']['exam_name']}")
                    print(f"題號: {result['metadata']['question_number']}")
                if results:
                    print(f"✓ 成功執行相似題目搜索，返回 {len(results)} 個結果")
                else:
                    print("! 搜索未返回結果")
            else:
                print("無題目數據，跳過搜索測試")
        except Exception as e:
            print(f"✗ 搜索失敗: {str(e)}")

        try:
            vs.delete_exam_questions(exam_name)
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