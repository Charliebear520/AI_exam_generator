#!/usr/bin/env python
# test_dependencies.py - 測試依賴項導入

def test_imports():
    """測試所有必需的庫導入是否正常"""
    import_results = {}
    
    # 測試 PyPDF2 導入
    try:
        from PyPDF2 import PdfReader
        import_results["PyPDF2"] = "成功"
    except ImportError as e:
        import_results["PyPDF2"] = f"失敗: {str(e)}"
    
    # 測試 pdfplumber 導入
    try:
        import pdfplumber
        import_results["pdfplumber"] = "成功"
    except ImportError as e:
        import_results["pdfplumber"] = f"失敗: {str(e)}"
    
    # 測試 Crypto 導入
    try:
        from Crypto.Cipher import AES
        import_results["pycryptodome (Crypto)"] = "成功"
    except ImportError as e:
        import_results["pycryptodome (Crypto)"] = f"失敗: {str(e)}"
    
    # 測試 google.generativeai 導入
    try:
        import google.generativeai as genai
        import_results["google.generativeai"] = "成功"
    except ImportError as e:
        import_results["google.generativeai"] = f"失敗: {str(e)}"
    
    # 打印結果
    print("依賴項測試結果:")
    print("-" * 40)
    for lib, result in import_results.items():
        print(f"{lib}: {result}")
    print("-" * 40)
    
    # 檢查是否所有導入都成功
    all_success = all(result == "成功" for result in import_results.values())
    print(f"全部依賴項測試: {'通過' if all_success else '失敗'}")
    
    return all_success

if __name__ == "__main__":
    test_imports() 