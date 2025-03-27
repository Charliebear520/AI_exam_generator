{
  res.exam_point && (
    <div className="metadata-section">
      <p className="metadata-title">考點：</p>
      <div className="exam-point-content">{res.exam_point}</div>
    </div>
  );
}
{
  res.keywords && res.keywords.length > 0 && (
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
  );
}
{
  res.law_references && res.law_references.length > 0 && (
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
  );
}
