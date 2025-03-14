import React, { useState, useEffect } from "react";
import { getQuestions, deleteQuestions } from "../services/api";
import "./QuestionsList.css";

const QuestionsList = ({ onRefresh }) => {
  const [exams, setExams] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchQuestions = async () => {
    try {
      setLoading(true);
      const data = await getQuestions();

      if (data.exams) {
        setExams(
          data.exams.map((exam) => ({
            name: exam.exam_name,
            questionCount: exam.total_questions,
          }))
        );
      } else {
        setError("無法獲取考試列表");
        setExams([]);
      }
    } catch (err) {
      console.error("獲取考試列表失敗:", err);
      setError("獲取考試列表失敗");
      setExams([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, []);

  const handleDelete = async (examName) => {
    try {
      if (window.confirm(`確定要刪除 ${examName} 的所有題目嗎？`)) {
        await deleteQuestions(examName);
        fetchQuestions();
        if (onRefresh) {
          onRefresh();
        }
      }
    } catch (err) {
      console.error("刪除考試失敗:", err);
      setError("刪除考試失敗");
    }
  };

  if (loading) {
    return <div className="loading">載入中...</div>;
  }

  if (error) {
    return <div className="error-message">{error}</div>;
  }

  return (
    <div className="exams-list">
      <h2>考試列表</h2>
      {exams.length > 0 ? (
        <ul>
          {exams.map((exam) => (
            <li key={exam.name} className="exam-item">
              <div className="exam-info">
                <span className="exam-name">{exam.name}</span>
                <span className="question-count">
                  ({exam.questionCount} 題)
                </span>
              </div>
              <button
                onClick={() => handleDelete(exam.name)}
                className="delete-btn"
              >
                刪除
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="no-exams">尚未上傳任何考試</p>
      )}
    </div>
  );
};

export default QuestionsList;
