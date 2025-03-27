# models.py
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from database import Base

class Question(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    exam_name = Column(String(255), nullable=False, index=True)
    question_number = Column(Integer, nullable=False, index=True)
    content = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)
    answer = Column(String, nullable=True)
    explanation = Column(Text, nullable=True)
    exam_point = Column(String(255), nullable=True)
    keywords = Column(JSON, nullable=True)
    law_references = Column(JSON, nullable=True)
    type = Column(String(50), default="單選題", nullable=True)
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
            "exam_point": self.exam_point,
            "keywords": self.keywords,
            "law_references": self.law_references,
            "type": self.type,
            "created_at": self.created_at.isoformat()
        }