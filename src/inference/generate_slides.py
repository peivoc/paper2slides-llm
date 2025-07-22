import json
import os
from pathlib import Path
import datetime
import yaml
import google.generativeai as genai

# 這是我們在 data_processing 中建立的模組
from src.data_processing.prompt_generator import convert_to_prompt

# 定義資料夾路徑
PROCESSED_DIR = Path("data/processed")
SLIDES_DIR = Path("data/slides")
CONFIG_PATH = Path("configs/model_config.yaml")
SLIDES_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    """載入模型設定檔"""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到設定檔 {CONFIG_PATH}")
        return None
    except yaml.YAMLError as e:
        print(f"❌ 錯誤：解析 YAML 設定檔失敗: {e}")
        return None

def load_processed_paper(file_path: Path) -> dict:
    """載入已處理的論文 JSON 檔案。"""
    print(f"\n📂 正在載入已處理的論文: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到檔案 {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ 錯誤：檔案 {file_path} 不是有效的 JSON 格式。")
        return None

def generate_presentation_content(prompt: str) -> str:
    """
    呼叫 Google Gemini API 來生成簡報內容。
    
    Args:
        prompt (str): 完整的 Prompt 字串。

    Returns:
        str: 由模型生成的 Markdown 簡報內容，或在失敗時返回 None。
    """
    print("🤖 正在呼叫 Gemini API 生成簡報...")
    
    config = load_config()
    if not config:
        return None

    api_key = config.get('gemini_api_key')
    if not api_key or api_key == "YOUR_NEW_API_KEY_HERE":
        return "❌ 錯誤：Gemini API 金鑰未設定。請在 configs/model_config.yaml 中設定您的金鑰。"

    try:
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": config.get('temperature', 0.7),
            "top_p": config.get('top_p', 1.0),
            "top_k": config.get('top_k', 32),
            "max_output_tokens": config.get('max_output_tokens', 4096),
        }
        
        model = genai.GenerativeModel(
            model_name=config.get('model_name', 'gemini-1.5-flash-latest'),
            generation_config=generation_config
        )
        
        response = model.generate_content(prompt)
        
        print("✅ 簡報內容已成功生成！")
        return response.text
        
    except Exception as e:
        print(f"❌ 呼叫 Gemini API 時發生錯誤: {e}")
        return f"# 錯誤\n\n呼叫 Gemini API 時發生錯誤:\n`{str(e)}`"

def save_slides(content: str, source_filename: str):
    """將生成的簡報內容儲存為 Markdown 檔案。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{Path(source_filename).stem}_slides_{timestamp}.md"
    output_path = SLIDES_DIR / output_filename
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 簡報已成功儲存至: {output_path}")
        return output_path
    except IOError as e:
        print(f"❌ 錯誤：無法儲存檔案 {output_path}。原因: {e}")
        return None

def paper_to_slides(processed_file_path: str):
    """
    將單一已處理的論文 JSON 檔案轉換為簡報的完整流程。
    """
    # 1. 載入處理好的論文資料
    paper_data = load_processed_paper(Path(processed_file_path))
    if not paper_data:
        return

    # 2. 將論文資料轉換為 Prompt
    prompt = convert_to_prompt(paper_data)

    # 3. 呼叫 LLM 生成簡報內容
    slides_content = generate_presentation_content(prompt)
    if not slides_content:
        print("❌ 流程中止：無法生成簡報內容。")
        return

    # 4. 儲存生成的簡報
    source_filename = paper_data.get('source_file', 'unknown_paper')
    save_slides(slides_content, source_filename)
    
    print("\n✨ 流程執行完畢！")

if __name__ == '__main__':
    # --- 測試流程 ---
    try:
        example_files = list(PROCESSED_DIR.glob("*.json"))
        example_files = [f for f in example_files if f.name != 'processing_summary.json']
        
        if not example_files:
            raise FileNotFoundError("在 data/processed/ 中找不到任何 .json 檔案可以測試。")
            
        latest_file = max(example_files, key=lambda f: f.stat().st_mtime)
        
        print(f"🚀 開始執行 'paper_to_slides' 流程，使用範例檔案: {latest_file.name}")
        paper_to_slides(str(latest_file))

    except FileNotFoundError as e:
        print(f"❌ 測試中止: {e}")
    except Exception as e:
        print(f"❌ 執行時發生未預期的錯誤: {e}")
