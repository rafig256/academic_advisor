import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset


# بارگیری داده‌ها از فایل اکسل
df = pd.read_excel("data.xlsx")

# حذف ردیف‌هایی که سوال یا جوابشون خالیه (برای اطمینان)
df = df.dropna(subset=["question", "answer"])

# ساخت ستون‌های input و output برای مدل
df["input_text"] = "سوال: " + df["question"].astype(str)
df["target_text"] = df["answer"].astype(str)

# نمایش چند نمونه‌ی اول برای بررسی
# print(df[["input_text", "target_text"]].head())

# تقسیم داده به 80% آموزش و 20% ارزیابی
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# بررسی سایز داده‌ها
# print(f"Train size: {len(train_df)} | Validation size: {len(val_df)}")

# دوباره توکنایزر رو بارگذاری می‌کنیم (اگه قبلاً بارگذاری شده نیازی نیست، ولی بد نیست مجدد تعریف شه)
model_name = "HooshvareLab/bert-fa-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["question"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# تبدیل به فرمت HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# توکنایز هر دو دیتاست
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)