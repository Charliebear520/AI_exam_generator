import React, { useState } from 'react';
import { generateExam, submitAnswers } from '../services/api';
import { message, Button, Input, Form } from 'antd';

const ExamGenerator = () => {
  const [examName, setExamName] = useState('');
  const [numQuestions, setNumQuestions] = useState(10);
  const [generatedExam, setGeneratedExam] = useState([]);
  const [userAnswers, setUserAnswers] = useState({}); // { questionId: "A", ... }
  const [scoreResult, setScoreResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleGenerateExam = async () => {
    if (!examName.trim()) {
      message.error("請輸入考試名稱");
      return;
    }
    setLoading(true);
    try {
      const data = await generateExam(examName, numQuestions);
      setGeneratedExam(data.adapted_exam);
      message.success("生成模擬考題成功！");
    } catch (err) {
      console.error(err);
      // 嘗試從 error.response.data.detail 讀取後端返回的完整錯誤訊息
      const errorMsg = err.response?.data?.detail || err.message || "生成模擬考題失敗！";
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  const handleOptionClick = (questionId, optionKey) => {
    setUserAnswers(prev => ({
      ...prev,
      [questionId]: optionKey
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
        answers: userAnswers
      };
      const result = await submitAnswers(payload);
      setScoreResult(result);
      message.success("提交答案成功！");
    } catch (err) {
      console.error(err);
      const errorMsg = err.response?.data?.detail || err.message || "提交答案失敗！";
      message.error(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ddd', borderRadius: '8px' }}>
      <h2>模擬考題生成測試</h2>
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
            value={numQuestions} 
            type="number" 
            onChange={(e) => setNumQuestions(parseInt(e.target.value) || 10)}
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={handleGenerateExam} loading={loading}>
            {loading ? '生成中...' : '生成模擬考題'}
          </Button>
        </Form.Item>
      </Form>

      {generatedExam.length > 0 && (
        <div>
          <h3>生成的考題</h3>
          {generatedExam.map((q) => (
            <div key={q.id} style={{ marginBottom: '1rem', padding: '0.5rem', border: '1px solid #eee', borderRadius: '4px' }}>
              <p><strong>題目 #{q.id}:</strong> {q.content}</p>
              <p><strong>選項:</strong></p>
              <div>
                {['A', 'B', 'C', 'D'].map(letter => (
                  <Button 
                    key={letter}
                    type={userAnswers[q.id] === letter ? "primary" : "default"}
                    onClick={() => handleOptionClick(q.id, letter)}
                    style={{ marginRight: '8px' }}
                  >
                    {letter}: {q.options[letter] || "未生成"}
                  </Button>
                ))}
              </div>
              <p style={{ fontSize: '0.8rem', color: '#888' }}>改編: {q.source}</p>
            </div>
          ))}
          <Button type="primary" onClick={handleSubmitAnswers} loading={loading}>
            提交答案
          </Button>
        </div>
      )}

      {scoreResult && (
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f8f8f8', borderRadius: '8px' }}>
          <h3>作答結果</h3>
          <p>得分: {scoreResult.score} / {scoreResult.total}</p>
          {scoreResult.results.map((res) => (
            <div key={res.question_id} style={{ marginBottom: '0.5rem' }}>
              <p><strong>題目 {res.question_id}:</strong> {res.is_correct ? '正確' : '錯誤'}</p>
              <p>你的答案: {res.user_answer}，正確答案: {res.correct_answer}</p>
              <p>解析: {res.explanation}</p>
              <p style={{ fontSize: '0.8rem', color: '#888' }}>{res.source_info}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ExamGenerator;