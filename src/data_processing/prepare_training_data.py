import json
from pathlib import Path
import re
import sys

# 將專案根目錄添加到 Python 的搜索路徑中
# 這樣即使從子目錄執行，也能正確導入模組
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_processing.prompt_generator import convert_to_prompt

# 定義資料夾路徑
PROCESSED_DIR = Path("data/processed")
SLIDES_DIR = Path("data/slides")
TRAINING_DIR = Path("data/training")

TRAINING_DIR.mkdir(parents=True, exist_ok=True)

def prepare_training_dataset(output_filename: str = "training_data.jsonl"):
    """
    準備用於模型微調的訓練資料集。
    它會將處理過的論文資料 (Prompt) 與人工修改後的簡報 (Completion) 配對。
    """
    print("\n🚀 開始準備訓練資料集...")
    
    training_examples = []
    
    # 遍歷所有已處理的論文 JSON 檔案
    processed_files = list(PROCESSED_DIR.glob("*.json"))
    # 排除 processing_summary.json
    processed_files = [f for f in processed_files if f.name != 'processing_summary.json']

    if not processed_files:
        print("❗ 在 data/processed/ 中找不到任何已處理的論文 JSON 檔案。請先執行 main.py 處理論文。")
        return

    for processed_file in processed_files:
        try:
            with open(processed_file, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
            
            # 從處理過的檔案名中提取基礎名稱 (例如 '2104.05740v1' from '2104.05740v1.json')
            # 或者如果我們採用了截斷邏輯，就是 '2104' from '2104.json'
            # 這裡我們使用 processed_file.stem，它會是完整的檔名（不含.json）
            # 為了兼容之前的檔名截斷邏輯，我們需要更靈活地匹配
            base_name = processed_file.stem # 例如 '2104.05740v1' 或 '2104'

            # 尋找對應的簡報 Markdown 檔案
            # 簡報檔案名可能包含時間戳，所以我們用 glob 模式匹配開頭
            # 這裡假設人工修改後的簡報檔案名仍然以原始論文的 base_name 開頭
            # 例如：2104.05740v1_slides_20240717_123456.md 或 2104_my_edited_slides.md
            slide_files = list(SLIDES_DIR.glob(f"{base_name}*.md"))
            
            if not slide_files:
                print(f"⚠️ 找不到與 {processed_file.name} 對應的簡報 Markdown 檔案。請確保您已修改並儲存了簡報。")
                continue
            
            # 如果有多個匹配的簡報檔案，我們選擇最新的那一個
            slide_file = max(slide_files, key=lambda f: f.stat().st_mtime)
            
            with open(slide_file, 'r', encoding='utf-8') as f:
                completion_text = f.read()
            
            # 生成 Prompt
            prompt_text = convert_to_prompt(processed_data)
            
            # 將 Prompt 和 Completion 合併為 SFTTrainer 需要的格式
            # 格式為: {"text": "<s>[INST] prompt內容 [/INST] completion內容 </s>"}
            formatted_text = f"<s>[INST] {prompt_text} [/INST] {completion_text} </s>"
            training_examples.append({"text": formatted_text})
            print(f"✅ 已配對並轉換 {processed_file.name} 和 {slide_file.name}")

        except Exception as e:
            print(f"❌ 處理 {processed_file.name} 時發生錯誤: {e}")
            continue

    if not training_examples:
        print("❌ 未生成任何訓練範例。請檢查 data/processed/ 和 data/slides/ 中的檔案。")
        return

    # 將訓練範例寫入 JSONL 檔案
    output_path = TRAINING_DIR / output_filename
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"\n🎉 訓練資料集已成功生成至: {output_path}")
    print(f"   共生成 {len(training_examples)} 個訓練範例。")

if __name__ == "__main__":
    prepare_training_dataset()
