import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np

# بارگیری داده‌ها از فایل اکسل
df = pd.read_excel("data.xlsx")

# حذف ردیف‌هایی که سوال یا جوابشان خالی است
df = df.dropna(subset=["question", "answer"])

# تقسیم داده‌ها به آموزش و ارزیابی
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# ساخت ستون‌های input_text و target_text
train_df["input_text"] = "سوال: " + train_df["question"].astype(str)
train_df["target_text"] = train_df["answer"].astype(str)

val_df["input_text"] = "سوال: " + val_df["question"].astype(str)
val_df["target_text"] = val_df["answer"].astype(str)

# بارگیری توکنایزر و مدل
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# تابع توکنایز کردن
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
    labels = tokenizer(
        examples["target_text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# تبدیل داده‌ها به فرمت Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# اعمال توکنایزر
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# بارگیری متریک ارزیابی
rouge = evaluate.load("rouge")

# تابع محاسبه متریک‌ها
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}

# Data Collator برای پد کردن دسته‌ای
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./results",
    save_steps=500,  # ذخیره مدل هر 500 مرحله
    eval_steps=500,  # ارزیابی هر 500 مرحله
    # evaluation_strategy="steps",  # ارزیابی هر 500 مرحله
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    # load_best_model_at_end=True,  # بارگذاری بهترین مدل در پایان
    logging_dir='./logs',
    logging_steps=10,
    fp16=False
)


# راه‌اندازی Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# اجرای آموزش
trainer.train()

# ذخیره مدل و توکنایزر آموزش‌دیده
trainer.save_model("./trained_mt5")
tokenizer.save_pretrained("./trained_mt5")
