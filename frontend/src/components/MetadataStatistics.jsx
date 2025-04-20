import React, { useState, useEffect } from "react";
import {
  Card,
  Tabs,
  Table,
  Spin,
  Progress,
  Radio,
  InputNumber,
  Button,
  Alert,
  Select,
  Tooltip,
  Collapse,
} from "antd";
import {
  BarChartOutlined,
  CloseOutlined,
  InfoCircleOutlined,
} from "@ant-design/icons";
import "./MetadataStatistics.css";

const { TabPane } = Tabs;
const { Option } = Select;
const { Panel } = Collapse;

const MetadataStatistics = () => {
  const [activeTab, setActiveTab] = useState("exam_point");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState({ stats: [], total_questions: 0 });
  const [limit, setLimit] = useState(20);
  const [minCount, setMinCount] = useState(2);
  const [error, setError] = useState(null);

  const fetchStatistics = async (type = activeTab) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `http://localhost:8000/statistics/metadata_by_year?metadata_type=${type}&limit=${limit}&min_count=${minCount}`
      );

      if (!response.ok) {
        throw new Error(
          `獲取統計數據失敗: ${response.status} ${response.statusText}`
        );
      }

      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error("獲取統計數據時出錯:", error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatistics();
  }, []);

  const handleTabChange = (key) => {
    setActiveTab(key);
    fetchStatistics(key);
  };

  const applyFilters = () => {
    fetchStatistics();
  };

  const getTypeLabel = (type) => {
    switch (type) {
      case "exam_point":
        return "考點";
      case "keyword":
        return "關鍵詞";
      case "law_reference":
        return "法條引用";
      default:
        return "";
    }
  };

  const columns = [
    {
      title: "序號",
      dataIndex: "index",
      key: "index",
      render: (_, __, index) => index + 1,
      width: 70,
    },
    {
      title: getTypeLabel(activeTab),
      dataIndex: "item",
      key: "item",
    },
    {
      title: "出現次數",
      dataIndex: "count",
      key: "count",
      sorter: (a, b) => a.count - b.count,
      width: 110,
    },
    {
      title: "佔比",
      dataIndex: "percentage",
      key: "percentage",
      render: (_, record) => {
        // 直接使用該項目的出現次數
        const percentage = (
          (record.count / data.total_questions) *
          100
        ).toFixed(2);
        return <span>{percentage}%</span>;
      },
      width: 90,
    },
    {
      title: "出現年份",
      dataIndex: "years",
      key: "years",
      render: (_, record) => {
        // 從所有題目中提取該項目出現的年份
        const years = new Set();
        data.stats.forEach((yearData) => {
          yearData.items.forEach((item) => {
            if (item.item === record.item) {
              years.add(yearData.year);
            }
          });
        });

        return (
          <div className="year-tags">
            {Array.from(years)
              .sort()
              .map((year) => (
                <span key={year} className="year-tag">
                  {year}年
                </span>
              ))}
          </div>
        );
      },
    },
  ];

  return (
    <div className="metadata-statistics">
      <Card title="考題元數據統計" className="statistics-card">
        <Tabs activeKey={activeTab} onChange={handleTabChange}>
          <TabPane tab="考點統計" key="exam_point">
            <div className="statistics-description">
              顯示題庫中出現頻率最高的考點，幫助您了解重點考察內容
            </div>
          </TabPane>
          <TabPane tab="關鍵詞統計" key="keyword">
            <div className="statistics-description">
              顯示題庫中出現頻率最高的關鍵詞，幫助您把握核心概念
            </div>
          </TabPane>
          <TabPane tab="法條引用統計" key="law_reference">
            <div className="statistics-description">
              顯示題庫中引用最多的法條，幫助您重點掌握重要法條
            </div>
          </TabPane>
        </Tabs>

        <div className="filters">
          <div className="filter-item">
            <span>顯示數量:</span>
            <InputNumber
              min={1}
              max={100}
              value={limit}
              onChange={(value) => setLimit(value)}
            />
          </div>
          <div className="filter-item">
            <span>最小出現次數:</span>
            <InputNumber
              min={1}
              max={50}
              value={minCount}
              onChange={(value) => setMinCount(value)}
            />
          </div>
          <Button type="primary" onClick={applyFilters}>
            應用篩選
          </Button>
        </div>

        {error && (
          <Alert
            message="獲取數據錯誤"
            description={error}
            type="error"
            showIcon
            style={{ marginBottom: 16 }}
          />
        )}

        <Spin spinning={loading}>
          {data.stats && data.stats.length > 0 ? (
            <>
              <div className="statistics-summary">
                總共分析 {data.total_questions} 道題目
              </div>
              <Table
                dataSource={data.stats.flatMap((yearData) => yearData.items)}
                columns={columns}
                rowKey={(record) => record.item}
                pagination={{ pageSize: 10 }}
                size="middle"
              />
            </>
          ) : (
            !loading && (
              <div className="no-data">
                暫無數據，請嘗試減少最小出現次數或上傳更多題目
              </div>
            )
          )}
        </Spin>
      </Card>
    </div>
  );
};

const MetadataStatisticsSidebar = ({ visible, onClose }) => {
  const [activeTab, setActiveTab] = useState("exam_point");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState({ items: [], total_questions: 0 });
  const [limit, setLimit] = useState(10);
  const [minCount, setMinCount] = useState(2);

  useEffect(() => {
    if (visible) {
      fetchStatistics();
    }
  }, [visible, activeTab]);

  const fetchStatistics = async () => {
    if (!visible) return;

    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8000/statistics/metadata?metadata_type=${activeTab}&limit=${limit}&min_count=${minCount}`
      );

      if (response.ok) {
        const result = await response.json();
        setData(result);
      } else {
        console.error("獲取統計數據失敗:", response.statusText);
      }
    } catch (error) {
      console.error("獲取統計數據時出錯:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (key) => {
    setActiveTab(key);
  };

  const applyFilters = () => {
    fetchStatistics();
  };

  // 側邊欄樣式，基於可見性狀態
  const sidebarClass = visible
    ? "metadata-sidebar sidebar-visible"
    : "metadata-sidebar";

  const getTypeLabel = (type) => {
    switch (type) {
      case "exam_point":
        return "考點";
      case "keyword":
        return "關鍵詞";
      case "law_reference":
        return "法條引用";
      default:
        return "";
    }
  };

  const columns = [
    {
      title: "排名",
      dataIndex: "index",
      key: "index",
      render: (_, __, index) => index + 1,
      width: 50,
    },
    {
      title: getTypeLabel(activeTab),
      dataIndex: "item",
      key: "item",
      ellipsis: {
        showTitle: false,
      },
      render: (text) => (
        <Tooltip placement="topLeft" title={text}>
          {text}
        </Tooltip>
      ),
    },
    {
      title: "次數",
      dataIndex: "count",
      key: "count",
      width: 60,
    },
    {
      title: "百分比",
      dataIndex: "percentage",
      key: "percentage",
      render: (percentage) => (
        <Progress
          percent={percentage}
          size="small"
          format={(percent) => `${percent}%`}
          status="active"
        />
      ),
      width: 100,
    },
  ];

  if (!visible) {
    return (
      <div className="sidebar-toggle-container">
        <Button
          type="primary"
          icon={<BarChartOutlined />}
          className="toggle-sidebar-button"
          onClick={() => onClose(true)}
          title="顯示統計數據"
        />
      </div>
    );
  }

  return (
    <div className={sidebarClass}>
      <div className="sidebar-header">
        <h3>
          <BarChartOutlined /> 考題元數據統計
        </h3>
        <Button
          type="text"
          icon={<CloseOutlined />}
          onClick={() => onClose(false)}
          title="關閉統計面板"
        />
      </div>

      <Tabs activeKey={activeTab} onChange={handleTabChange} size="small">
        <TabPane tab="考點" key="exam_point" />
        <TabPane tab="關鍵詞" key="keyword" />
        <TabPane tab="法條引用" key="law_reference" />
      </Tabs>

      <div className="sidebar-filters">
        <div className="filter-row">
          <span>顯示數量:</span>
          <InputNumber
            min={1}
            max={50}
            value={limit}
            onChange={(value) => setLimit(value)}
            size="small"
          />
        </div>
        <div className="filter-row">
          <span>最小出現次數:</span>
          <InputNumber
            min={1}
            max={20}
            value={minCount}
            onChange={(value) => setMinCount(value)}
            size="small"
          />
        </div>
        <Button type="primary" size="small" onClick={applyFilters}>
          更新統計
        </Button>
      </div>

      <div className="statistics-info">
        <InfoCircleOutlined /> 分析 {data.total_questions || 0} 道題目， 找到{" "}
        {data.total_distinct || 0} 個不同{getTypeLabel(activeTab)}
      </div>

      <div className="statistics-table-container">
        <Spin spinning={loading}>
          {data.items && data.items.length > 0 ? (
            <Table
              dataSource={data.items}
              columns={columns}
              rowKey={(record) => record.item}
              pagination={{ pageSize: 5, size: "small" }}
              size="small"
              scroll={{ y: 300 }}
            />
          ) : (
            <div className="no-data">暫無數據</div>
          )}
        </Spin>
      </div>

      <div className="sidebar-footer">
        <Button
          type="link"
          size="small"
          onClick={() => window.open("/statistics", "_blank")}
        >
          查看詳細統計 →
        </Button>
      </div>
    </div>
  );
};

export default MetadataStatistics;
