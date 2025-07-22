import os
import arxiv

def fetch_and_save(query=None, max_results=5, output_dir="data/raw"):
    """
    爬取RAG相關論文
    """
    os.makedirs(output_dir, exist_ok=True)
    client = arxiv.Client()
    
    # 改進的搜尋查詢，更精確地定位RAG相關論文
    if query is None:
        # 使用更精確的搜尋策略
        query = '(ti:"Retrieval-Augmented Generation" OR ti:"Retrieval Augmented Generation" OR ti:RAG) OR (abs:"Retrieval-Augmented Generation" OR abs:"Retrieval Augmented Generation")'
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    print(f"🔍 搜尋查詢: {query}")
    print(f"📊 預期結果數量: {max_results}")
    print("-" * 50)
    
    downloaded_count = 0
    for result in client.results(search):
        print(f"📄 標題: {result.title}")
        print(f"📅 發表日期: {result.published.strftime('%Y-%m-%d')}")
        print(f"👥 作者: {', '.join([author.name for author in result.authors[:3]])}{'...' if len(result.authors) > 3 else ''}")
        print(f"🔗 分類: {', '.join(result.categories)}")
        
        # 檢查是否真的與RAG相關
        title_lower = result.title.lower()
        abstract_lower = result.summary.lower()
        rag_keywords = ['retrieval-augmented', 'retrieval augmented', 'rag', 'retrieve', 'generation']
        
        is_rag_related = any(keyword in title_lower or keyword in abstract_lower for keyword in rag_keywords)
        
        if is_rag_related:
            print("✅ 確認為RAG相關論文")
            
            fname = result.get_short_id() + ".pdf"
            out_path = os.path.join(output_dir, fname)
            
            if not os.path.exists(out_path):
                try:
                    result.download_pdf(filename=out_path)
                    print(f"   ✅ 已下載至 {out_path}")
                    downloaded_count += 1
                except Exception as e:
                    print(f"   ❌ 下載失敗: {e}")
            else:
                print("   🔁 檔案已存在，跳過下載")
                downloaded_count += 1
        else:
            print("❌ 不是RAG相關論文，跳過")
        
        print("-" * 30)
    
    print(f"\n📈 總共下載了 {downloaded_count} 篇RAG相關論文")

def fetch_rag_papers_advanced(max_results=10, output_dir="data/raw"):
    """
    使用多個搜尋策略來獲取更多RAG相關論文
    """
    # 不同的搜尋策略
    search_queries = [
        # 精確匹配標題
        'ti:"Retrieval-Augmented Generation"',
        'ti:"Retrieval Augmented Generation"',
        'ti:"RAG"',
        
        # 摘要中包含相關詞彙
        'abs:"retrieval-augmented generation"',
        'abs:"retrieval augmented generation"',
        
        # 相關技術詞彙
        'ti:"dense passage retrieval" OR ti:"DPR"',
        'ti:"retrieval-based" AND ti:"generation"',
        'abs:"retrieval-based question answering"',
        
        # 相關模型和方法
        'ti:"FiD" OR ti:"Fusion-in-Decoder"',
        'ti:"REALM" OR ti:"Retrieval-Enhanced"',
        'ti:"T5" AND abs:"retrieval"'
    ]
    
    print("🚀 開始進階RAG論文搜尋...")
    
    for i, query in enumerate(search_queries, 1):
        print(f"\n🔍 搜尋策略 {i}/{len(search_queries)}: {query}")
        try:
            fetch_and_save(query=query, max_results=max_results//len(search_queries), output_dir=output_dir)
        except Exception as e:
            print(f"❌ 搜尋策略 {i} 失敗: {e}")

if __name__ == "__main__":
    # 選擇搜尋方式
    print("請選擇搜尋方式:")
    print("1. 基本搜尋 (推薦)")
    print("2. 進階搜尋 (多策略)")
    
    choice = input("請輸入選擇 (1 或 2): ").strip()
    
    if choice == "2":
        fetch_rag_papers_advanced(max_results=20)
    else:
        fetch_and_save(max_results=90)