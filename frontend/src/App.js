import React, { useState, useEffect, useCallback } from "react";
import "./App.css";
import QuestionsList from "./components/QuestionsList";
import ExamGenerator from "./components/ExamGenerator";
import { uploadPdf, getQuestions, downloadJSON } from "./services/api";

function App() {
  const [file, setFile] = useState(null);
  const [examName, setExamName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

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
            <QuestionsList key={refreshKey} onRefresh={() => setRefreshKey((prev) => prev + 1)} />
          </section>
        </div>

        {/* 右側改為顯示模擬考題生成區塊 */}
        <div className="right-column">
          <ExamGenerator />
        </div>
      </main>

      <footer>
        <p>&copy; {new Date().getFullYear()} 考試題目 PDF 轉 JSON 工具</p>
      </footer>
    </div>
  );
}

export default App;