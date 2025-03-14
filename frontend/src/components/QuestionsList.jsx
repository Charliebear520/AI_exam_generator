import React, { useState, useEffect } from "react";
import { getQuestions, deleteQuestions } from "../services/api";
import "./QuestionsList.css";

const QuestionsList = ({ onRefresh }) => {
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [deletingExam, setDeletingExam] = useState(null);

  // 获取题目列表
  const fetchQuestions = async () => {
    try {
      setLoading(true);
      const data = await getQuestions();
      console.log("API返回的题目数据:", data); // 添加日志

      if (!data || !data.questions || !Array.isArray(data.questions)) {
        console.error("API返回的数据格式不正确:", data);
        setError("获取题目失败: 数据格式不正确");
        setLoading(false);
        return;
      }

      // 按考试名称分组
      const groupedQuestions = {};
      data.questions.forEach((q) => {
        // 确保exam_name存在且有效
        if (q && q.exam_name && typeof q.exam_name === "string") {
          if (!groupedQuestions[q.exam_name]) {
            groupedQuestions[q.exam_name] = {
              exam_name: q.exam_name,
              count: 0,
              created_at: q.created_at || new Date().toISOString(),
            };
          }
          groupedQuestions[q.exam_name].count++;
        } else {
          console.warn("跳过无效的题目数据:", q);
        }
      });

      console.log("分组后的考试数据:", groupedQuestions); // 添加日志

      // 转换为数组并按创建时间排序
      const exams = Object.values(groupedQuestions).sort(
        (a, b) => new Date(b.created_at) - new Date(a.created_at)
      );

      console.log("排序后的考试列表:", exams); // 添加日志
      setQuestions(exams);
      setError(null);
    } catch (err) {
      setError("获取题目失败: " + err.message);
      console.error("获取题目失败:", err); // 添加错误日志
    } finally {
      setLoading(false);
    }
  };

  // 删除题目
  const handleDelete = async (examName) => {
    console.log("尝试删除考试，名称:", examName, "类型:", typeof examName); // 添加类型检查

    if (!examName || examName === "undefined" || typeof examName !== "string") {
      setError(`考试名称无效，无法删除: ${String(examName)}`);
      return;
    }

    if (
      window.confirm(`确定要删除"${examName}"的所有题目吗？此操作不可撤销。`)
    ) {
      try {
        setDeletingExam(examName);
        console.log("正在刪除考試:", examName); // 添加日志
        await deleteQuestions(examName);
        console.log("刪除成功，刷新列表");
        fetchQuestions(); // 重新加载题目列表
        if (onRefresh) onRefresh(); // 通知父组件刷新
      } catch (err) {
        setError("刪除題目失敗: " + err.message);
        console.error("刪除失敗:", err); // 添加错误日志
      } finally {
        setDeletingExam(null);
      }
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, []);

  if (loading && questions.length === 0) {
    return <div className="loading">加載中...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="questions-list">
      <h2>最近上傳的题目</h2>
      {questions.length === 0 ? (
        <p>暫無题目，請上傳PDF文件</p>
      ) : (
        <ul>
          {questions.map((exam, index) => {
            // 确保exam和exam.exam_name有效
            if (
              !exam ||
              !exam.exam_name ||
              typeof exam.exam_name !== "string"
            ) {
              console.warn("跳過無效的考試項:", exam);
              return null;
            }

            console.log(`渲染考試項 ${index}:`, exam); // 添加日志
            return (
              <li key={exam.exam_name || index} className="exam-item">
                <div className="exam-info">
                  <span className="exam-name">{exam.exam_name}</span>
                  <span className="exam-count">{exam.count} 題</span>
                </div>
                <div className="exam-actions">
                  <button
                    className="delete-btn"
                    onClick={(e) => {
                      e.preventDefault(); // 防止事件冒泡
                      const name = exam.exam_name;
                      console.log("點擊刪除按鈕，考試名稱:", name); // 添加日志
                      if (name && typeof name === "string") {
                        handleDelete(name);
                      } else {
                        console.error("無效的考試名稱:", name);
                        setError(`無效的考試名稱: ${String(name)}`);
                      }
                    }}
                    disabled={deletingExam === exam.exam_name}
                  >
                    {deletingExam === exam.exam_name ? "刪除中..." : "🗑️ 刪除"}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      )}
      <div className="refresh-container">
        <button className="refresh-btn" onClick={fetchQuestions}>
          刷新列表
        </button>
      </div>
    </div>
  );
};

export default QuestionsList;
