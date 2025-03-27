import React, { useState, useEffect } from "react";
import { submitAnswers } from "../services/api";
import { message, Button, Input, Form, Tabs, Modal, Spin, Tag } from "antd";
import "./ExamGenerator.css";

const { TabPane } = Tabs;

const ExamGenerator = () => {
  const [examName, setExamName] = useState("");
  const [keyword, setKeyword] = useState("");
  const [numQuestions, setNumQuestions] = useState(10);
  const [generatedExam, setGeneratedExam] = useState([]);
  const [userAnswers, setUserAnswers] = useState({}); // { questionId: "A", ... }
  const [scoreResult, setScoreResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("default");
  const [modalVisible, setModalVisible] = useState(false);
  const [modalContent, setModalContent] = useState({ type: "", data: [] });
  const [metadataLoading, setMetadataLoading] = useState(false);

  const handleGenerateExam = async () => {
    if (!examName.trim()) {
      message.error("請輸入考試名稱");
      return;
    }
    if (activeTab === "keyword" && !keyword.trim()) {
      message.error("請輸入關鍵字以進行 RAG 檢索");
      return;
    }
    setLoading(true);
    try {
      // 建立 FormData，附上 exam_name 與 num_questions
      const formData = new FormData();
      formData.append("exam_name", examName);
      formData.append("num_questions", numQuestions);
      if (activeTab === "keyword") {
        formData.append("keyword", keyword);
      }
      const response = await fetch("http://localhost:8000/generate_exam", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error((await response.json()).detail || response.statusText);
      }
      const data = await response.json();
      setGeneratedExam(data.adapted_exam);
      message.success("生成模擬考題成功！");
    } catch (err) {
      console.error(err);
      const errorMsg =
        err.response?.data?.detail || err.message || "生成模擬考題失敗！";
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleOptionClick = (questionId, optionKey) => {
    setUserAnswers((prev) => ({
      ...prev,
      [questionId]: optionKey,
    }));
  };

  const handleSubmitAnswers = async () => {
    if (generatedExam.length === 0) {
      message.error("尚未生成考題");
      return;
    }
    setLoading(true);
    try {
      const payload = {
        adapted_exam: generatedExam,
        answers: userAnswers,
      };
      const result = await submitAnswers(payload);
      setScoreResult(result);
      message.success("提交答案成功！");
    } catch (err) {
      console.error(err);
      const errorMsg =
        err.response?.data?.detail || err.message || "提交答案失敗！";
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const showMetadata = async (type) => {
    setMetadataLoading(true);
    setModalVisible(true);

    try {
      let url = "";
      let title = "";

      if (type === "examPoints") {
        url = "http://localhost:8000/legal/exam_points";
        title = "考點列表";
      } else if (type === "keywords") {
        url = "http://localhost:8000/legal/keywords";
        title = "關鍵詞列表";
      } else if (type === "lawReferences") {
        url = "http://localhost:8000/legal/law_references";
        title = "法條引用列表";
      }

      const response = await fetch(url);
      const data = await response.json();

      setModalContent({
        type: title,
        data:
          data[type.replace("Points", "_points")] ||
          data.keywords ||
          data.law_references ||
          [],
      });
    } catch (error) {
      console.error("獲取元數據失敗:", error);
      message.error("載入數據失敗");
    } finally {
      setMetadataLoading(false);
    }
  };

  return (
    <div className="exam-generator-container">
      <h2>模擬考題生成測試</h2>
      <Tabs activeKey={activeTab} onChange={(key) => setActiveTab(key)}>
        <TabPane tab="預設生成" key="default">
          <Form layout="vertical">
            <Form.Item label="考試名稱">
              <Input
                value={examName}
                onChange={(e) => setExamName(e.target.value)}
                placeholder="例如：111年司法官考試"
              />
            </Form.Item>
            <Form.Item label="題目數量">
              <Input
                type="number"
                value={numQuestions}
                onChange={(e) =>
                  setNumQuestions(parseInt(e.target.value) || 10)
                }
              />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                onClick={handleGenerateExam}
                loading={loading}
              >
                {loading ? "生成中..." : "生成模擬考題"}
              </Button>
            </Form.Item>
          </Form>
        </TabPane>
        <TabPane tab="關鍵字檢索" key="keyword">
          <Form layout="vertical">
            <Form.Item label="考試名稱">
              <Input
                value={examName}
                onChange={(e) => setExamName(e.target.value)}
                placeholder="例如：111年司法官考試"
              />
            </Form.Item>
            <Form.Item label="關鍵字 (RAG 檢索)">
              <Input
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="輸入關鍵字，例如：刑法"
              />
            </Form.Item>
            <Form.Item label="題目數量">
              <Input
                type="number"
                value={numQuestions}
                onChange={(e) =>
                  setNumQuestions(parseInt(e.target.value) || 10)
                }
              />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                onClick={handleGenerateExam}
                loading={loading}
              >
                {loading ? "生成中..." : "生成模擬考題"}
              </Button>
            </Form.Item>
          </Form>
        </TabPane>
      </Tabs>

      {generatedExam.length > 0 && (
        <div>
          <h3>生成的考題</h3>
          {generatedExam.map((q) => (
            <div key={q.id} className="question-item">
              <p>
                <strong>題目 #{q.id}:</strong> {q.content}
              </p>
              <p>
                <strong>選項:</strong>
              </p>
              <div>
                {["A", "B", "C", "D"].map((letter) => (
                  <Button
                    key={letter}
                    type={userAnswers[q.id] === letter ? "primary" : "default"}
                    onClick={() => handleOptionClick(q.id, letter)}
                    style={{ marginRight: "8px" }}
                  >
                    {letter}: {q.options[letter] || "未生成"}
                  </Button>
                ))}
              </div>
              <p className="question-source">改編: {q.source}</p>
            </div>
          ))}
          <Button
            type="primary"
            onClick={handleSubmitAnswers}
            loading={loading}
          >
            提交答案
          </Button>
        </div>
      )}

      {scoreResult && (
        <div className="exam-result-container">
          <h3>作答結果</h3>
          <p>
            得分: {scoreResult.score} / {scoreResult.total}
          </p>
          {scoreResult.results.map((res) => (
            <div key={res.question_id} className="question-result">
              <p>
                <strong>題目 {res.question_id}:</strong>{" "}
                {res.is_correct ? "正確" : "錯誤"}
              </p>
              <p>
                你的答案: {res.user_answer}，正確答案: {res.correct_answer}
              </p>
              <p>解析: {res.explanation}</p>
              <p className="question-source">{res.source_info}</p>
              {res.exam_point && (
                <div className="metadata-section">
                  <p className="metadata-title">考點：</p>
                  <div className="exam-point-content">{res.exam_point}</div>
                </div>
              )}
              {res.keywords && res.keywords.length > 0 && (
                <div className="metadata-section">
                  <p className="metadata-title">關鍵詞：</p>
                  <div>
                    {res.keywords.map((keyword, index) => (
                      <span key={index} className="keyword-tag">
                        {keyword}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {res.law_references && res.law_references.length > 0 && (
                <div className="metadata-section">
                  <p className="metadata-title">法律依據：</p>
                  <div>
                    {res.law_references.map((ref, index) => (
                      <div key={index} className="law-reference-item">
                        {ref}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div style={{ marginBottom: 16 }}>
        <Button
          onClick={() => showMetadata("examPoints")}
          style={{ marginRight: 8 }}
        >
          查看考點
        </Button>
        <Button
          onClick={() => showMetadata("keywords")}
          style={{ marginRight: 8 }}
        >
          查看關鍵詞
        </Button>
        <Button onClick={() => showMetadata("lawReferences")}>
          查看法條引用
        </Button>
      </div>

      <Modal
        title={modalContent.type}
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        <Spin spinning={metadataLoading}>
          <div style={{ maxHeight: "500px", overflow: "auto" }}>
            {modalContent.data.map((item, index) => (
              <Tag
                color={
                  modalContent.type === "考點列表"
                    ? "blue"
                    : modalContent.type === "關鍵詞列表"
                    ? "green"
                    : "purple"
                }
                key={index}
                style={{ margin: "5px" }}
              >
                {item}
              </Tag>
            ))}
          </div>
        </Spin>
      </Modal>
    </div>
  );
};

export default ExamGenerator;
