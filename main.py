import argparse
import os
from pathlib import Path
import sys

# 將 src 加入 Python 的搜尋路徑，這樣我們才能匯入自己的模組
# 這是常見的專案結構處理方式
sys.path.append(str(Path(__file__).resolve().parent))

from src.data_processing.extract_paper import process_single_pdf, process_all_pdfs
from src.inference.generate_slides import paper_to_slides

# 定義資料夾路徑
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def process_paper_flow(pdf_path: Path):
    """處理單一論文的完整流程"""
    print("="*50)
    print("🚀 開始執行 Paper-to-Slides 流程")
    print(f"📄 目標檔案: {pdf_path}")
    print("="*50)

    # --- 步驟 1: 檢查檔案是否存在 ---
    if not pdf_path.exists():
        print(f"❌ 錯誤：找不到指定的檔案！請確認 '{pdf_path.name}' 是否存在於 '{RAW_DIR}' 資料夾中。")
        # 提示使用者可能的檔案
        available_files = list(RAW_DIR.glob("*.pdf"))
        if available_files:
            print("\n📂 data/raw/ 中可用的檔案有:")
            for f in available_files:
                print(f"  - {f.name}")
        return

    # --- 步驟 2: 處理PDF，提取結構化內容 ---
    print("\n[階段 1/2] 正在處理 PDF 檔案...")
    processed_data = process_single_pdf(pdf_path)
    
    if not processed_data:
        print("❌ 流程中止：PDF 處理失敗，無法繼續生成簡報。")
        return
        
    # 根據 process_single_pdf 的邏輯，它會使用檔名的第一部分（例如 '2104'）作為檔名
    # 我們需要找到這個檔案的路徑，以便傳遞給下一步
    filename_base = pdf_path.name.split('.')[0]
    processed_filename = f"{filename_base}.json"
    processed_file_path = PROCESSED_DIR / processed_filename
    
    if not processed_file_path.exists():
        print(f"❌ 錯誤：預期的處理後檔案 {processed_filename} 不存在於 {PROCESSED_DIR} 中。")
        return
        
    print("✅ PDF 處理完成!")

    # --- 步驟 3: 生成簡報 ---
    print("\n[階段 2/2] 正在生成簡報內容...")
    paper_to_slides(str(processed_file_path))
    
    print("\n" + "="*50)
    print("🎉 恭喜！所有流程已成功執行完畢！")
    print(f"✨ 請查看 data/slides/ 資料夾中的 Markdown 簡報檔案。")
    print("="*50)

def main():
    """專案主執行函式"""
    
    # --- 設定命令列參數 ---
    parser = argparse.ArgumentParser(
        description="Paper-to-Slides: 將學術論文PDF自動轉換為Markdown簡報。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="選擇要執行的命令")

    # 'single' 子命令
    single_parser = subparsers.add_parser(
        "single", 
        help="處理單一 PDF 檔案"
    )
    single_parser.add_argument(
        "filename",
        type=str,
        help="指定要處理的PDF檔案名稱（例如 '2401.09603v2.pdf'）。檔案必須位於 data/raw/ 資料夾中。"
    )

    # 'all' 子命令
    all_parser = subparsers.add_parser(
        "all", 
        help="處理 data/raw/ 資料夾中所有 PDF 檔案"
    )
    
    args = parser.parse_args()
    
    if args.command == "single":
        pdf_filename = args.filename
        pdf_path = RAW_DIR / pdf_filename
        process_paper_flow(pdf_path)
    elif args.command == "all":
        print("="*50)
        print("🚀 開始執行 Paper-to-Slides 流程 (處理所有 PDF)")
        print("="*50)
        
        pdf_files = list(RAW_DIR.glob("*.pdf"))
        if not pdf_files:
            print("❗ 找不到任何 PDF 論文，請確認 data/raw/ 資料夾")
            return

        for pdf_path in pdf_files:
            process_paper_flow(pdf_path)
            print("\n" + "-"*50 + "\n") # 分隔不同論文的處理輸出
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
