import pandas as pd

# بارگیری داده‌ها از فایل اکسل
df = pd.read_excel("data.xlsx")

# حذف ردیف‌هایی که سوال یا جوابشون خالیه (برای اطمینان)
df = df.dropna(subset=["question", "answer"])

# ساخت ستون‌های input و output برای مدل
df["input_text"] = "سوال: " + df["question"].astype(str)
df["target_text"] = df["answer"].astype(str)

# نمایش چند نمونه‌ی اول برای بررسی
print(df[["input_text", "target_text"]].head())