#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PostgreSQL 數據庫遷移腳本
用於向 questions 表添加 type 欄位
"""

import os
import psycopg2
from psycopg2 import sql
from database import DATABASE_URL

def migrate_database():
    """執行數據庫遷移，添加缺失的欄位"""
    print("開始數據庫結構更新...")
    
    conn = None
    try:
        # 連接數據庫
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # 檢查 type 欄位是否已存在
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='questions' AND column_name='type'
        """)
        if cur.fetchone() is None:
            print("正在添加 type 欄位...")
            # 如果不存在，添加 type 欄位，並設置默認值為"單選題"
            cur.execute(sql.SQL("""
                ALTER TABLE questions 
                ADD COLUMN type VARCHAR(50) DEFAULT '單選題'
            """))
            conn.commit()
            print("type 欄位添加成功！")
        else:
            print("type 欄位已存在，無需添加")
        
        # 提交事務
        conn.commit()
        print("數據庫結構更新完成")
        
    except Exception as e:
        print(f"數據庫遷移過程中發生錯誤: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate_database() 