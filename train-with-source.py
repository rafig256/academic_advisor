import os
import json
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq

# پوشه حاوی فایل‌های jsonl
source_folder = "source"
all_data = []

# خواندن فایل‌های .jsonl به صورت خط‌به‌خط
for filename in os.listdir(source_folder):
    if filename.endswith(".jsonl"):
        filepath = os.path.join(source_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    example = json.loads(line.strip())
                    if "input" in example and "target" in example:
                        all_data.append(example)
                except json.JSONDecodeError as e:
                    print(f"خطا در خواندن خطی از {filename}: {e}")

# تقسیم داده‌ها به train و validation
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data)
})

# بارگذاری مدل و tokenizer
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# پیش‌پردازش داده‌ها
def preprocess(examples):
    inputs = ["خلاصه کن: " + inp for inp in examples["input"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess, batched=True)

# تنظیمات آموزش
training_args = TrainingArguments(
    output_dir="./mt5_summarizer",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # <<< داده‌های اعتبارسنجی
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# آموزش
trainer.train()

# ذخیره نهایی
model.save_pretrained("./mt5_summarizer")
tokenizer.save_pretrained("./mt5_summarizer")
