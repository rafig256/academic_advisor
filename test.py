from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def main():
    model_path = "./trained_mt5"  # مسیر پوشه‌ی مدل ذخیره شده
    
    try:
        # مدل و توکنایزر را بارگذاری می‌کنیم
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)  # چون fast نیاز به protobuf دارد، غیرفعالش کردیم
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    except Exception as e:
        print("\n❌ خطا در بارگذاری مدل یا توکنایزر:")
        print(e)
        return

    # سوال تستی
    question = "انتخاب رشته تئاتر در دبیرستان وجود داره؟"
    
    try:
        # سوال را توکنایز می‌کنیم
        inputs = tokenizer(question, return_tensors="pt")

        # مدل را روی حالت ارزیابی می‌گذاریم
        model.eval()
        
        # پیش‌بینی (با torch.no_grad چون مرحله آموزش نیستیم)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100)

        # خروجی را دی‌کود (decode) می‌کنیم
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        print("\n✅ پاسخ مدل:")
        print("----------------------------------")
        print(answer)
        print("----------------------------------")
        
    except Exception as e:
        print("\n❌ خطا در پردازش سوال:")
        print(e)

if __name__ == "__main__":
    main()
