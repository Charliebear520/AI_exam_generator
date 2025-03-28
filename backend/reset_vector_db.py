"""
重置向量數據庫腳本 - 用於修復 'no such column: segments.topic' 錯誤

使用方法：
python reset_vector_db.py

警告：此腳本將刪除所有現有的向量數據！
"""

import os
import shutil
import sys

def reset_vector_db():
    # 使用不同的目錄路徑解決版本不兼容問題
    home_dir = os.path.expanduser("~")
    # 使用帶有版本號的新目錄名稱
    persist_path = os.path.join(home_dir, "test_generator_vector_db_v2")
    
    try:
        if os.path.exists(persist_path):
            print(f"正在刪除向量庫目錄: {persist_path}")
            shutil.rmtree(persist_path)
            print("成功刪除向量庫目錄")
        else:
            print(f"向量庫目錄不存在: {persist_path}")
        
        # 建立新的目錄
        os.makedirs(persist_path, exist_ok=True)
        print(f"已創建新的向量庫目錄: {persist_path}")
        return True
    except Exception as e:
        print(f"刪除向量庫時發生錯誤: {str(e)}")
        try:
            # 嘗試使用系統命令強制刪除
            os.system(f"rm -rf {persist_path}")
            os.makedirs(persist_path, exist_ok=True)
            print("已使用系統命令強制重置向量庫目錄")
            return True
        except Exception as e2:
            print(f"強制重置失敗: {str(e2)}")
            return False

if __name__ == "__main__":
    print("=" * 50)
    print("警告：此操作將刪除所有向量庫數據！")
    print("=" * 50)
    
    confirm = input("輸入 'RESET' 確認重置操作: ")
    
    if confirm == "RESET":
        if reset_vector_db():
            print("向量庫重置成功，請重新啟動後端服務")
        else:
            print("向量庫重置失敗")
    else:
        print("操作已取消") 