// src/App.js
import React, { useState, useEffect, useCallback } from "react";
import "./App.css";
import QuestionsList from "./components/QuestionsList";
import { uploadPdf, getQuestions, downloadJSON } from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [examName, setExamName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [recentQuestions, setRecentQuestions] = useState([]);
  const [filterExamName, setFilterExamName] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [visibleQuestions, setVisibleQuestions] = useState(10); // 默认显示10道题目

  const fetchRecentQuestions = useCallback(async () => {
    try {
      const data = await getQuestions(filterExamName || null, 0, 100);

      // 處理新的數據格式
      if (data.exams) {
        // 將所有考試的題目合併成一個數組
        let allQuestions = [];
        data.exams.forEach((exam) => {
          // 為每個題目添加考試名稱
          const questionsWithExamName = exam.questions.map((q) => ({
            ...q,
            exam_name: exam.exam_name,
          }));
          allQuestions = [...allQuestions, ...questionsWithExamName];
        });

        // 按ID排序
        let filteredQuestions = allQuestions.sort((a, b) => b.id - a.id);

        // 應用搜索過濾
        if (searchTerm) {
          filteredQuestions = filteredQuestions.filter(
            (q) =>
              q.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
              q.explanation?.toLowerCase().includes(searchTerm.toLowerCase())
          );
        }

        setRecentQuestions(filteredQuestions);
      } else {
        console.error("API返回的數據格式不正確:", data);
        setRecentQuestions([]);
      }
    } catch (err) {
      console.error("無法載入最近題目:", err);
      setRecentQuestions([]);
    }
  }, [filterExamName, searchTerm]);

  useEffect(() => {
    fetchRecentQuestions();
    const interval = setInterval(fetchRecentQuestions, 900000);
    return () => clearInterval(interval);
  }, [fetchRecentQuestions]);

  useEffect(() => {
    fetchRecentQuestions();
  }, [result, fetchRecentQuestions]);

  const loadMoreQuestions = () => {
    setVisibleQuestions((prev) => prev + 10); // 每次加载10道题目
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.type === "application/pdf") {
      setFile(selectedFile);
      setError(null);
    } else {
      setFile(null);
      setError("請選擇 PDF 檔案");
    }
  };

  const handleExamNameChange = (e) => {
    setExamName(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("請選擇檔案");
      return;
    }
    if (!examName) {
      setError("請輸入考試名稱");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await uploadPdf(file, examName);
      setResult(data);
      setFile(null);
      setExamName("");
      setRefreshKey((prev) => prev + 1);
    } catch (err) {
      setError(err.detail || "上傳失敗，請稍後再試");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (result && result.download_url) {
      const filename = result.download_url.split("/").pop();
      downloadJSON(filename);
    }
  };

  const handleRefresh = () => {
    setRefreshKey((prev) => prev + 1);
    fetchRecentQuestions();
  };

  return (
    <div className="app-container">
      <header>
        <h1>考試題目 PDF 轉 JSON 工具</h1>
        <p>
          上傳 PDF 考試試題，自動轉換為結構化 JSON 格式，作為 RAG 檢索資料來源
        </p>
      </header>

      <main className="main-content">
        <div className="left-column">
          <div className="upload-section">
            <h2>上傳PDF文件</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="examName">考試名稱：</label>
                <input
                  type="text"
                  id="examName"
                  value={examName}
                  onChange={handleExamNameChange}
                  placeholder="例如：111年司法官考試"
                  required
                />
              </div>
              <div className="form-group">
                <label htmlFor="pdfFile">PDF 檔案：</label>
                <input
                  type="file"
                  id="pdfFile"
                  onChange={handleFileChange}
                  accept=".pdf"
                  required
                />
                <small>僅支援 PDF 格式</small>
              </div>
              <button type="submit" disabled={loading} className="submit-btn">
                {loading ? "處理中..." : "上傳並解析"}
              </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="result-section">
                <h2>處理結果</h2>
                <p>成功解析 {result.questions_count} 道題目</p>
                <button onClick={handleDownload} className="download-btn">
                  下載 JSON 檔案
                </button>
              </div>
            )}
          </div>

          <section className="questions-section">
            <QuestionsList key={refreshKey} onRefresh={handleRefresh} />
          </section>
        </div>

        <div className="right-column">
          <div className="recent-questions">
            <h2>最近上傳的題目</h2>
            <div className="filter-section">
              <input
                type="text"
                placeholder="按考試名稱篩選"
                value={filterExamName}
                onChange={(e) => setFilterExamName(e.target.value)}
              />
              <input
                type="text"
                placeholder="搜尋題目內容或解析"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="questions-container">
              {recentQuestions.length > 0 ? (
                <ul>
                  {recentQuestions.slice(0, visibleQuestions).map((q) => (
                    <li key={q.id + q.exam_name}>
                      <strong>#{q.id}</strong>: {q.content.substring(0, 50)}...
                      <details>
                        <summary>查看詳情</summary>
                        <p>
                          <strong>選項:</strong>
                        </p>
                        <ul>
                          {Object.entries(q.options).map(([key, value]) => (
                            <li key={key}>
                              {key}: {value}
                            </li>
                          ))}
                        </ul>
                        <p>
                          <strong>答案:</strong> {q.answer}
                        </p>
                        <p>
                          <strong>解析:</strong>{" "}
                          {q.explanation.substring(0, 100)}
                          ...
                        </p>
                      </details>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>無符合條件的題目</p>
              )}
              {recentQuestions.length > visibleQuestions && (
                <div className="load-more">
                  <button onClick={loadMoreQuestions} className="load-more-btn">
                    加載更多題目
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer>
        <p>&copy; {new Date().getFullYear()} 考試題目 PDF 轉 JSON 工具</p>
      </footer>
    </div>
  );
}

export default App;
