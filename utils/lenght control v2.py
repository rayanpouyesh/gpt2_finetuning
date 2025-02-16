import pandas as pd

def extract_words_from_sentences(input_file, output_file):
    # خواندن فایل اکسل ورودی
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except Exception as e:
        print(f"خطا در خواندن فایل اکسل: {e}")
        return

    # بررسی وجود ستون 'sentence'
    if 'sentence' not in df.columns:
        print("ستون 'sentence' در فایل اکسل یافت نشد.")
        return

    # پردازش هر جمله و استخراج کلمات کامل قبل از 256 کاراکتر
    def process_sentence(sentence):
        text = str(sentence)[:256]  # متن تا 256 کاراکتر اول
        words = text.split()  # جداسازی کلمات
        processed_text = ''
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > 256:
                break
            processed_text += word + ' '
            current_length += len(word) + 1

        return processed_text.strip()  # حذف فضای اضافی انتهایی

    # اعمال تابع روی ستون 'sentence'
    df['sentence'] = df['sentence'].apply(process_sentence)

    # ذخیره فایل اکسل خروجی
    try:
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"جملات پردازش شده با موفقیت در فایل {output_file} ذخیره شدند.")
    except Exception as e:
        print(f"خطا در ذخیره فایل اکسل: {e}")

# مثال از استفاده از تابع
input_excel = 'C:\\rayanpoyesh_project\\Ghaemi\\512_sentences.xlsx'  # نام فایل اکسل ورودی
output_excel = 'C:\\rayanpoyesh_project\\Ghaemi\\256_sentences.xlsx'  # نام فایل اکسل خروجی
extract_words_from_sentences(input_excel, output_excel)
