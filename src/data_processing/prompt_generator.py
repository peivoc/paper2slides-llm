import json
from pathlib import Path

def convert_to_prompt(processed_data: dict) -> str:
    """
    將處理過的論文資料轉換為一個清晰的、給予大型語言模型(LLM)的Prompt。

    Args:
        processed_data (dict): 從 extract_paper.py 處理後得到的結構化資料。

    Returns:
        str: 格式化後的Prompt字串。
    """
    
    title = processed_data.get('metadata', {}).get('title', 'N/A')
    abstract = processed_data.get('metadata', {}).get('abstract', 'N/A')
    
    # 為了讓Prompt更精簡，我們可以選擇性地加入章節
    # 這裡我們先簡單地將所有章節內容串接起來
    full_text = f"Title: {title}\n\nAbstract: {abstract}\n\n"
    
    sections = processed_data.get('sections', [])
    if sections:
        full_text += "--- Main Content ---\n"
        for section in sections:
            # 移除章節標題中的換行符，讓格式更乾淨
            section_title_cleaned = ' '.join(section.get('title', 'Unnamed Section').split())
            full_text += f"\n## {section_title_cleaned}\n\n"
            full_text += section.get('content', '') + "\n"

    # --- Prompt 模板 ---
    prompt = f"""
# 指令：將學術論文轉換為簡報

## 你的角色
你是一位頂尖的學術研究助理，專長是將複雜的科學論文提煉成清晰、專業且易於理解的簡報投影片。

## 你的任務
根據下方提供的論文內容，生成一份完整的簡報。請嚴格遵循以下要求：

1.  **輸出格式**: 使用 Markdown 格式。
2.  **簡報結構**:
    *   總投影片數量應在 8 到 12 張之間。
    *   第一張必須是 **標題頁**，包含論文標題和作者（如果有的話）。
    *   必須包含 **摘要/研究動機**、**研究方法**、**主要發現/結果**、**結論與未來工作** 等核心章節的投影片。
    *   最後一張應為 **Q&A** 或 **感謝聆聽**。
3.  **內容要求**:
    *   每張投影片的內容應以 **條列式 (bullet points)**呈現，力求簡潔有力。
    *   避免直接複製貼上原文，而是要進行 **總結和提煉**。
    *   保留關鍵術語、數據和圖表引用（例如 "如圖1所示..."）。
4.  **投影片標題**: 每張投影片都應以 `## Slide X: [投影片標題]` 的格式開始。

---
## 論文內容

{full_text}

---
## 輸出結果

請在這裡開始生成你的 Markdown 格式簡報：
"""
    
    return prompt.strip()

if __name__ == '__main__':
    # 這是一個測試範例，實際使用時會從其他腳本呼叫
    # 假設我們有一個已處理好的 JSON 檔案
    processed_file = Path("../../data/processed/2407.json") # 假設這個檔案存在
    
    if processed_file.exists():
        with open(processed_file, 'r', encoding='utf-8') as f:
            paper_data = json.load(f)
        
        # 產生 prompt
        final_prompt = convert_to_prompt(paper_data)
        
        # 儲存 prompt 以便檢查
        prompt_output_path = Path("../../data/processed/prompt_example.txt")
        with open(prompt_output_path, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
            
        print(f"✅ Prompt 已成功生成並儲存於: {prompt_output_path}")
        print("\n--- Prompt 預覽 ---")
        print(final_prompt[:1000] + "...") # 預覽前1000個字元
    else:
        print(f"❌ 測試失敗：找不到範例檔案 {processed_file}")
        # 創建一個假的資料來演示
        fake_data = {
            'metadata': {
                'title': '這是一個範例論文標題',
                'abstract': '這是一段論文摘要，說明了研究的核心貢獻。'
            },
            'sections': [
                {'title': '1. Introduction', 'content': '這是引言的內容...'},
                {'title': '2. Methodology', 'content': '這是研究方法的內容...'},
                {'title': '3. Conclusion', 'content': '這是結論的內容...'}
            ]
        }
        final_prompt = convert_to_prompt(fake_data)
        print("\n--- 使用假資料的 Prompt 預覽 ---")
        print(final_prompt)
