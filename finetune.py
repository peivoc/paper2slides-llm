import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import os

# --- 1. 設定模型和資料集路徑 ---
# 我們選擇 Mistral-7B 作為基礎模型。它在性能和資源需求之間取得了很好的平衡。
# Gemma-2B 是另一個很好的選擇，如果 Mistral-7B 對您的硬體來說太大的話。
model_name = "google/gemma-2b-it"
dataset_path = "data/finetuning/training_dataset.jsonl" # 我們剛剛建立的資料集檔案
output_dir = "results/finetuned_adapter" # 訓練完成後，儲存模型 adapter 的地方

# --- 2. 設定 QLoRA 量化 ---
# 這是能在 4GB VRAM 上進行微調的關鍵！
# 我們將模型權重以 4-bit 的精度載入，大大減少記憶體佔用。
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # 使用 bfloat16 來進行計算，以保持精度
    bnb_4bit_use_double_quant=False,
)

# --- 3. 載入模型和 Tokenizer ---
print("載入模型和 Tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # 自動將模型分配到可用的硬體上（例如 GPU）
)
model.config.use_cache = False # 在訓練時關閉快取，這是一個推薦的最佳實踐


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# 設定 padding token。如果 tokenizer 沒有 pad_token，我們通常會將其設定為 eos_token。
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # 將 padding 放在右側，以避免 T5 這類模型的問題

# --- 4. 設定 PEFT (LoRA) ---
# 我們只訓練一小部分的 "adapter" 權重，而不是整個模型。
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 5. 設定訓練參數 ---
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,                # 訓練的總輪數。先從 1 開始，之後可以增加。
    per_device_train_batch_size=1,     # 由於 VRAM 有限，我們一次只處理一個樣本
    gradient_accumulation_steps=4,     # 梯度累積。效果等同於 batch_size = 1 * 4 = 4
    optim="paged_adamw_32bit",         # 使用分頁優化器以節省記憶體
    save_steps=25,                     # 每 25 個步驟儲存一次 checkpoint
    logging_steps=5,                   # 每 5 個步驟記錄一次 log
    learning_rate=2e-4,                # 學習率
    weight_decay=0.001,
    fp16=False,                        # 我們使用 bfloat16，所以關閉 fp16
    bf16=True,                         # 啟用 bfloat16 訓練
    max_grad_norm=0.3,
    max_steps=-1,                      # 如果設定了，會覆寫 num_train_epochs
    warmup_ratio=0.03,
    group_by_length=True,              # 將長度相近的樣本分組，以提高效率
    lr_scheduler_type="constant",      # 學習率排程器
)

# --- 6. 建立並開始訓練 ---
print("載入資料集...")
# 我們需要先載入資料集
dataset = load_dataset("json", data_files=dataset_path, split="train")

print("設定 SFT Trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # 告訴 trainer 我們資料集裡的文字欄位叫做 "text"
    max_seq_length=2048,       # 序列的最大長度。可以根據您的 VRAM 和資料進行調整。
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,             # 是否將多個短樣本打包成一個長樣本
)

print("🚀 開始微調！")
trainer.train()

# --- 7. 儲存最終的模型 adapter ---
print("✅ 微調完成，正在儲存 adapter...")
trainer.save_model(output_dir)
print(f"🎉 成功！Adapter 已儲存至 {output_dir}")
