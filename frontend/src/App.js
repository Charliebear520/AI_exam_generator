import React, { useState } from "react";
import "./App.css";
import QuestionsList from "./components/QuestionsList";
import ExamGenerator from "./components/ExamGenerator";
import { uploadPdf, downloadJSON } from "./services/api";
import { Upload, Button, message, Spin } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import {
  SignedIn,
  SignedOut,
  SignInButton,
  UserButton,
} from "@clerk/clerk-react";

function App() {
  const [file, setFile] = useState(null);
  const [examName, setExamName] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Upload.Dragger 配置，只接受 PDF 並禁止自動上傳
  const uploadProps = {
    name: "file",
    accept: ".pdf",
    multiple: false,
    showUploadList: true, // 顯示上傳清單
    beforeUpload: (uploadedFile) => {
      if (uploadedFile.type !== "application/pdf") {
        message.error("請上傳 PDF 檔案");
        return Upload.LIST_IGNORE;
      }
      // 返回 false 阻止自動上傳
      return false;
    },
    onChange: (info) => {
      // info.fileList 為當前上傳的文件清單
      if (info.fileList.length > 0) {
        // 取第一個文件的原始檔案
        setFile(info.fileList[0].originFileObj);
      } else {
        setFile(null);
      }
    },
    onRemove: () => {
      setFile(null);
    },
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
        <div className="header-content">
          <div className="auth-section">
            <SignedIn>
              <UserButton afterSignOutUrl="/" />
            </SignedIn>
          </div>
          <h1>AI 生成模擬考題 工具</h1>
          <p>上傳想要練習的考試題庫，AI 會自動解析 PDF 檔案，並生成模擬考題</p>
        </div>
      </header>

      <SignedIn>
        <main className="main-content">
          <div className="left-column">
            <div className="upload-section">
              <h2>上傳題庫</h2>
              <form onSubmit={handleSubmit}>
                <div className="form-group">
                  <label htmlFor="examName">題庫名稱：</label>
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
                  <Upload.Dragger {...uploadProps}>
                    <p className="ant-upload-drag-icon">
                      <UploadOutlined />
                    </p>
                    <p className="ant-upload-text">點擊或拖曳上傳 PDF 檔案</p>
                    <p className="ant-upload-hint">僅支援 PDF 格式</p>
                  </Upload.Dragger>
                </div>
                <div className="form-group">
                  <Button type="primary" htmlType="submit" disabled={loading}>
                    {loading ? <Spin /> : "上傳並解析"}
                  </Button>
                </div>
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
              <QuestionsList
                key={refreshKey}
                onRefresh={() => setRefreshKey((prev) => prev + 1)}
              />
            </section>
          </div>

          <div className="right-column">
            <ExamGenerator />
          </div>
        </main>
      </SignedIn>

      <SignedOut>
        <div className="signin-container">
          <h2>請先登入以使用系統</h2>
          <SignInButton mode="modal" redirectUrl="/">
            <Button type="primary" size="large">
              點擊此處登入
            </Button>
          </SignInButton>
        </div>
      </SignedOut>

      <footer>
        <p>&copy; {new Date().getFullYear()} AI 生成模擬考題 工具 BY黃塏峻</p>
      </footer>
    </div>
  );
}

export default App;
