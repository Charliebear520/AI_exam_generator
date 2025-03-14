import React, { useState } from "react";
import axios from "axios";
import { Button, Form, Input, Upload, message, Progress } from "antd";
import {
  UploadOutlined,
  LockOutlined,
  FileTextOutlined,
} from "@ant-design/icons";

const UploadForm = ({ onUploadSuccess }) => {
  const [fileList, setFileList] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [password, setPassword] = useState("");
  const [examName, setExamName] = useState("");
  const [showPasswordInput, setShowPasswordInput] = useState(false);

  const handleUpload = async () => {
    const formData = new FormData();
    const file = fileList[0];

    if (!file) {
      message.error("请先选择PDF文件");
      return;
    }

    if (!examName.trim()) {
      message.error("请输入考试名称");
      return;
    }

    formData.append("file", file);
    formData.append("exam_name", examName);

    // 如果提供了密码，添加到表单数据
    if (password) {
      formData.append("password", password);
    }

    setUploading(true);
    setProgress(0);

    try {
      const response = await axios.post(
        "http://localhost:8000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          },
        }
      );

      setFileList([]);
      setPassword("");
      setExamName("");
      setShowPasswordInput(false);
      message.success("上传成功！");

      // 如果提供了回调函数，调用它并传递响应数据
      if (onUploadSuccess) {
        onUploadSuccess(response.data);
      }
    } catch (error) {
      console.error("上传失败:", error);

      // 检查是否是PDF加密错误
      if (
        error.response &&
        error.response.status === 422 &&
        error.response.data.detail &&
        error.response.data.detail.includes("已加密")
      ) {
        message.error("PDF文件已加密，请提供密码");
        setShowPasswordInput(true);
      } else {
        message.error(
          `上传失败: ${error.response?.data?.detail || "未知错误"}`
        );
      }
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const props = {
    onRemove: () => {
      setFileList([]);
      setShowPasswordInput(false);
    },
    beforeUpload: (file) => {
      if (file.type !== "application/pdf") {
        message.error("只能上传PDF文件！");
        return Upload.LIST_IGNORE;
      }

      setFileList([file]);
      return false;
    },
    fileList,
  };

  return (
    <Form layout="vertical">
      <Form.Item
        label="考试名称"
        required
        tooltip="请输入一个有意义的名称，用于区分不同的考试"
      >
        <Input
          prefix={<FileTextOutlined />}
          placeholder="输入考试名称"
          value={examName}
          onChange={(e) => setExamName(e.target.value)}
        />
      </Form.Item>

      <Form.Item label="上传PDF文件">
        <Upload {...props} maxCount={1}>
          <Button icon={<UploadOutlined />}>选择文件</Button>
        </Upload>
      </Form.Item>

      {showPasswordInput && (
        <Form.Item label="PDF密码（如果有）">
          <Input.Password
            prefix={<LockOutlined />}
            placeholder="输入PDF密码"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </Form.Item>
      )}

      <Form.Item>
        <Button
          type="primary"
          onClick={handleUpload}
          disabled={fileList.length === 0 || !examName.trim()}
          loading={uploading}
        >
          {uploading ? "上传中..." : "上传"}
        </Button>
      </Form.Item>

      {uploading && progress > 0 && (
        <Form.Item>
          <Progress percent={progress} />
        </Form.Item>
      )}
    </Form>
  );
};

export default UploadForm;
