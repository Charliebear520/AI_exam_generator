# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from database import Base

Base = declarative_base()

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    exam_name = Column(String(255), nullable=False)
    question_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)
    answer = Column(String, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    
    def to_dict(self):
        return {
            "id": self.id,
            "exam_name": self.exam_name,
            "question_number": self.question_number,
            "content": self.content,
            "options": self.options,
            "answer": self.answer,
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }