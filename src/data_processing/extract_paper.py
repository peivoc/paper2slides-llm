import os
import re
from pathlib import Path
from pypdf import PdfReader
import json
from datetime import datetime

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """從PDF提取文本，並處理常見的PDF問題"""
    try:
        reader = PdfReader(str(pdf_path))
        text = ""
        total_pages = len(reader.pages)
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            
            # 進度顯示
            if i % 10 == 0:
                print(f"   處理進度: {i+1}/{total_pages} 頁")
        
        return text
    except Exception as e:
        print(f"❌ PDF讀取失敗: {e}")
        return ""


def clean_text(text):
    """清理文本，移除PDF常見的格式問題"""
    if not text:
        return ""
    
    # 移除頁碼和頁眉頁腳（常見模式）
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # 單獨的數字行
    text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text)  # Page X of Y
    
    # 處理連字符號斷行
    text = re.sub(r'([a-z])-\n([a-z])', r'\1\2', text)
    
    # 移除多餘的空白和換行
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 多個空行變成兩個
    text = re.sub(r'[ \t]+', ' ', text)  # 多個空格變成一個
    
    # 移除首尾空白
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def extract_paper_metadata(text):
    """嘗試提取論文基本資訊"""
    metadata = {}
    
    # 提取標題（通常在前幾行）
    lines = text.split('\n')[:20]  # 只看前20行
    title_candidates = []
    
    for line in lines:
        if len(line) > 20 and len(line) < 200:  # 標題長度合理
            if not re.search(r'^\d+\.|abstract|introduction|arxiv', line.lower()):
                title_candidates.append(line)
    
    if title_candidates:
        metadata['title'] = title_candidates[0]
    
    # 提取摘要
    abstract_match = re.search(r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n|\nintroduction|\n1\.|\nkeywords)', 
                              text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        metadata['abstract'] = abstract_match.group(1).strip()
    
    # 提取關鍵詞
    keywords_match = re.search(r'keywords?\s*[:\-]?\s*(.*?)(?=\n\s*\n|\nintroduction|\n1\.)', 
                              text, re.IGNORECASE | re.DOTALL)
    if keywords_match:
        metadata['keywords'] = keywords_match.group(1).strip()
    
    return metadata


def split_into_sections(text):
    """將文本分割為章節"""
    sections = []
    
    # 常見的章節標題模式
    section_patterns = [
        r'\n\s*(\d+\.?\s+[A-Z][^.\n]*)\n',  # 1. Introduction
        r'\n\s*([A-Z][A-Z\s]{2,})\n',      # ABSTRACT, INTRODUCTION
        r'\n\s*(Abstract|Introduction|Related Work|Methodology|Experiments|Results|Conclusion|References)\s*\n',
    ]
    
    # 找到所有章節標題
    section_breaks = [(0, "開始")]
    
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in matches:
            section_title = match.group(1).strip()
            if len(section_title) < 100:  # 避免誤判
                section_breaks.append((match.start(), section_title))
    
    # 按位置排序
    section_breaks.sort(key=lambda x: x[0])
    
    # 提取章節內容
    for i in range(len(section_breaks) - 1):
        start_pos = section_breaks[i][0]
        end_pos = section_breaks[i + 1][0]
        section_title = section_breaks[i][1]
        section_content = text[start_pos:end_pos].strip()
        
        if len(section_content) > 100:  # 過短的章節可能是誤判
            sections.append({
                'title': section_title,
                'content': section_content
            })
    
    # 處理最後一個章節
    if section_breaks:
        last_section = text[section_breaks[-1][0]:].strip()
        if len(last_section) > 100:
            sections.append({
                'title': section_breaks[-1][1],
                'content': last_section
            })
    
    return sections


def split_into_paragraphs(text, min_length=50):
    """將文本分割為段落，改進版本"""
    # 先按雙換行分割
    paragraphs = text.split('\n\n')
    
    # 進一步處理
    processed_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        
        # 過濾條件
        if len(para) < min_length:
            continue
        
        # 跳過可能的頁碼、頁眉頁腳
        if re.match(r'^\d+$', para) or re.match(r'^Page \d+', para):
            continue
        
        # 跳過純數字或符號
        if re.match(r'^[\d\s\-\.\(\)]+$', para):
            continue
        
        processed_paragraphs.append(para)
    
    return processed_paragraphs


def save_processed_data(data, output_path):
    """儲存處理後的資料，支援多種格式"""
    # 儲存為JSON格式（結構化資料）
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # 儲存為純文本格式（易讀）
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"論文處理結果\n")
        f.write(f"處理時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # 寫入元資料
        if 'metadata' in data:
            f.write("📋 論文資訊\n")
            f.write("-" * 20 + "\n")
            for key, value in data['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        # 寫入章節
        if 'sections' in data:
            f.write("📚 章節內容\n")
            f.write("-" * 20 + "\n")
            for i, section in enumerate(data['sections'], 1):
                f.write(f"[章節 {i}] {section['title']}\n")
                f.write(f"{section['content']}\n\n")
        
        # 寫入段落
        if 'paragraphs' in data:
            f.write("📄 段落內容\n")
            f.write("-" * 20 + "\n")
            for i, para in enumerate(data['paragraphs'], 1):
                f.write(f"[段落 {i}]\n{para}\n\n")


def process_single_pdf(pdf_path):
    """處理單個PDF文件"""
    print(f"📄 正在處理: {pdf_path.name}")
    
    # 提取原始文本
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        print(f"❌ 無法提取文本: {pdf_path.name}")
        return
    
    # 清理文本
    cleaned_text = clean_text(raw_text)
    
    # 提取資料
    metadata = extract_paper_metadata(cleaned_text)
    sections = split_into_sections(cleaned_text)
    paragraphs = split_into_paragraphs(cleaned_text)
    
    # 組織資料
    processed_data = {
        'source_file': pdf_path.name,
        'processed_time': datetime.now().isoformat(),
        'metadata': metadata,
        'sections': sections,
        'paragraphs': paragraphs,
        'statistics': {
            'total_text_length': len(cleaned_text),
            'section_count': len(sections),
            'paragraph_count': len(paragraphs)
        }
    }
    
    # 使用檔名的第一部分（例如 '2104' from '2104.05740v1.pdf'）作為檔名
    filename_base = pdf_path.name.split('.')[0]
    output_file = OUTPUT_DIR / filename_base
    save_processed_data(processed_data, output_file)
    
    print(f"✅ 處理完成: {pdf_path.name}")
    print(f"   - 章節數: {len(sections)}")
    print(f"   - 段落數: {len(paragraphs)}")
    print(f"   - 文本長度: {len(cleaned_text):,} 字元")
    
    return processed_data


def process_all_pdfs():
    """處理所有PDF文件"""
    pdf_files = list(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❗ 找不到任何 PDF 論文，請確認 data/raw/ 資料夾")
        return
    
    print(f"🚀 開始處理 {len(pdf_files)} 個PDF文件...")
    
    results = []
    for pdf_path in pdf_files:
        try:
            result = process_single_pdf(pdf_path)
            if result:
                results.append(result)
        except Exception as e:
            print(f"❌ 處理失敗 {pdf_path.name}: {e}")
    
    print(f"\n📊 處理完成！成功處理 {len(results)} 個文件")
    
    # 儲存處理摘要
    summary_path = OUTPUT_DIR / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(pdf_files),
            'successful_files': len(results),
            'processing_time': datetime.now().isoformat(),
            'file_statistics': [
                {
                    'filename': r['source_file'],
                    'sections': r['statistics']['section_count'],
                    'paragraphs': r['statistics']['paragraph_count'],
                    'text_length': r['statistics']['total_text_length']
                }
                for r in results
            ]
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_all_pdfs()