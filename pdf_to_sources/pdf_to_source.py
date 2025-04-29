import fitz  # PyMuPDF
import json
import os

def normalize_farsi_text(text):
    replacements = {
        '\u0640': '',         # کشیده
        'ي': 'ی',
        'ك': 'ک',
        '‌': ' ',             # نیم‌فاصله به فاصله
        '\u200c': ' ',         # Zero-width non-joiner
        '//' : '',
        '\n': '',            # سطر جدید → فاصله
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text

def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    current_paragraph = ""

    for page in doc:
        lines = page.get_text("text").splitlines()
        for line in lines:
            clean_line = normalize_farsi_text(line.strip())
            if not clean_line:
                # خط خالی: پایان پاراگراف
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                    current_paragraph = ""
            else:
                # ادامه پاراگراف
                if current_paragraph:
                    current_paragraph += " " + clean_line
                else:
                    current_paragraph = clean_line

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())  # آخرین پاراگراف

    # حذف پاراگراف‌های خیلی کوتاه یا بی‌معنا
    paragraphs = [p for p in paragraphs if len(p) > 50]

    print(f"🔍 استخراج متن از {pdf_path}: {len(paragraphs)} پاراگراف واقعی تشخیص داده شد")
    return paragraphs

def create_summarization_dataset(paragraphs, output_path):
    dataset = []
    for para in paragraphs:
        item = {
            "input": f"خلاصه کن: {para}",
            "target": ""  # بعداً می‌تونی خلاصه دستی یا خودکار اضافه کنی
        }
        dataset.append(item)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

    print(f"✅ خروجی ذخیره شد: {output_path} (تعداد نمونه‌ها: {len(dataset)})")

def main():
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf")
    base_dir = os.path.dirname(os.path.abspath(__file__))  # مسیر پوشه‌ی اسکریپت
    output_dir = os.path.join(base_dir, "..", "source")
    output_dir = os.path.abspath(output_dir)  # برای اطمینان، مطلق‌سازی مسیر

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.jsonl")

            print(f"🔄 در حال پردازش: {filename}")
            paragraphs = extract_paragraphs_from_pdf(pdf_path)
            if paragraphs:
                create_summarization_dataset(paragraphs, output_path)
            else:
                print(f"⚠️ هیچ پاراگرافی برای پردازش در فایل {filename} یافت نشد.")

if __name__ == "__main__":
    main()
