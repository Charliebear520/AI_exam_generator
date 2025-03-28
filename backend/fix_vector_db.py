#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修復向量數據庫腳本 - 用於解決 'no such column: collections.topic' 錯誤
此腳本會更新ChromaDB並重置數據庫結構

使用方法：
python fix_vector_db.py
"""

import os
import shutil
import sys
import subprocess

def fix_vector_db():
    print("=" * 50)
    print("開始修復向量數據庫...")
    print("=" * 50)
    
    # 步驟1: 嘗試升級ChromaDB
    try:
        print("正在升級ChromaDB...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chromadb", "--upgrade"])
        print("ChromaDB升級完成")
    except Exception as e:
        print(f"升級ChromaDB時發生錯誤: {str(e)}")
        print("繼續執行其他修復步驟...")
    
    # 步驟2: 重置向量庫目錄
    home_dir = os.path.expanduser("~")
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
        # 設置寬鬆權限確保可寫
        os.system(f"chmod -R 777 {persist_path}")
        print(f"已創建新的向量庫目錄: {persist_path}")
        
        # 嘗試修復系統級權限問題
        print("正在檢查系統權限...")
        os.system(f"chown -R $(whoami) {persist_path}")
        print("權限設置完成")
        
        print("=" * 50)
        print("修復完成，請重新啟動後端服務")
        print("=" * 50)
        return True
    except Exception as e:
        print(f"修復向量庫時發生錯誤: {str(e)}")
        try:
            # 嘗試使用系統命令強制刪除
            os.system(f"rm -rf {persist_path}")
            os.makedirs(persist_path, exist_ok=True)
            os.system(f"chmod -R 777 {persist_path}")
            print("已使用系統命令強制重置向量庫目錄")
            print("修復完成，請重新啟動後端服務")
            return True
        except Exception as e2:
            print(f"強制修復失敗: {str(e2)}")
            return False

if __name__ == "__main__":
    fix_vector_db() 