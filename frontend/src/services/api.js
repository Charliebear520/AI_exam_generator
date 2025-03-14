// API服务，用于与后端通信

const API_URL = "http://localhost:8000";

// 获取题目列表
export const getQuestions = async (examName = null, skip = 0, limit = 100) => {
  let url = `${API_URL}/questions?skip=${skip}&limit=${limit}`;
  if (examName) {
    url += `&exam_name=${encodeURIComponent(examName)}`;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`获取题目失败: ${response.statusText}`);
  }
  return await response.json();
};

// 删除题目
export const deleteQuestions = async (examName) => {
  // 检查examName是否有效
  if (!examName || examName === "undefined") {
    throw new Error("考试名称无效，无法删除");
  }

  console.log("发送删除请求，考试名称:", examName);

  const response = await fetch(
    `${API_URL}/questions/${encodeURIComponent(examName)}`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `删除题目失败: ${response.statusText}`);
  }
  return await response.json();
};

// 上传PDF文件
export const uploadPdf = async (file, examName) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("exam_name", examName);

  const response = await fetch(`${API_URL}/upload`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `上传失败: ${response.statusText}`);
  }
  return await response.json();
};

// 下载JSON文件
export const getDownloadUrl = (filename) => {
  return `${API_URL}/download/${filename}`;
};

// 直接下载JSON文件（兼容旧版本）
export const downloadJSON = (filename) => {
  window.open(getDownloadUrl(filename), "_blank");
};

// 以下是为了兼容旧代码的别名
export const uploadPDF = uploadPdf;
