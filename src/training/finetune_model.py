import os
import torch
from datasets import load_dataset
from transformers (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from pathlib import Path

# 定義路徑
TRAINING_DATA_PATH = Path("data/training/training_data.jsonl")
MODEL_OUTPUT_DIR = Path("models/gemma-2b-finetuned")

# 確保輸出目錄存在
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def finetune_gemma_2b():
    print("\n🚀 開始 Gemma-2B 模型微調流程...")

    # 1. 載入訓練資料集
    if not TRAINING_DATA_PATH.exists():
        print(f"❌ 錯誤：找不到訓練資料集 {TRAINING_DATA_PATH}。請確保您已執行 prepare_training_data.py 並生成了資料。")
        return
    
    # datasets 函式庫可以從 jsonl 檔案載入資料
    dataset = load_dataset("json", data_files=str(TRAINING_DATA_PATH), split="train")
    print(f"✅ 成功載入 {len(dataset)} 個訓練範例。")
    # print("資料集範例:", dataset[0]) # 可以取消註解查看資料集結構

    # 2. 設定基礎模型和分詞器
    model_name = "google/gemma-2b"
    
    # 4-bit 量化設定，用於節省 GPU 記憶體
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # 載入基礎模型 (Gemma-2B)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Gemma 模型的建議設定

    print("✅ 基礎模型和分詞器載入完成。")

    # 3. 設定 LoRA (Parameter-Efficient Fine-Tuning)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("✅ LoRA 設定完成。")

    # 4. 設定訓練參數
    training_arguments = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=1,  # 為了練習，先設定為 1 個 epoch
        per_device_train_batch_size=2, # 根據 Colab GPU 記憶體調整
        gradient_accumulation_steps=1, # 梯度累積步數
        optim="paged_adamw_8bit", # 8-bit AdamW 優化器
        save_steps=0, # 不在訓練過程中儲存檢查點
        logging_steps=25, # 每 25 步記錄一次日誌
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False, # Gemma 建議使用 bfloat16
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1, # 根據 num_train_epochs 決定步數
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none" # 不上報到任何平台，例如 wandb
    )
    print("✅ 訓練參數設定完成。")

    # 5. 初始化 SFTTrainer
    # 我們需要一個格式化函式來將 dataset 中的 prompt 和 completion 組合起來
    def formatting_prompts_func(examples):
        output_texts = []
        for i in range(len(examples["prompt"])):
            # 這裡的格式需要與您在 prompt_generator.py 中給模型的指令格式一致
            # 並且將 completion 放在 prompt 之後，作為模型學習的目標
            text = examples["prompt"][i] + examples["completion"][i]
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_arguments,
        max_seq_length=1024, # 根據您的資料長度調整，過長會導致 OOM
    )
    print("✅ SFTTrainer 初始化完成。")

    # 6. 開始訓練
    print("\n🔥 開始訓練模型...這可能需要一些時間。")
    trainer.train()
    print("\n🎉 模型訓練完成！")

    # 7. 儲存微調後的模型
    trainer.save_model(MODEL_OUTPUT_DIR)
    print(f"✅ 微調後的模型已儲存至: {MODEL_OUTPUT_DIR}")

    # 清理 GPU 記憶體
    del model
    del tokenizer
    del trainer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    finetune_gemma_2b()
