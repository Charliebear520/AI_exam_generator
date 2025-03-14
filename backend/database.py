# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker

# 資料庫連線設定
DATABASE_URL = "postgresql://postgres:11111111@localhost:5432/exam_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依賴注入
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()