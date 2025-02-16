import os
from docx import Document
from openpyxl import Workbook


# تابع برای خواندن فایل ورد و تقسیم متن به بخش‌های 10 کلمه‌ای
def read_and_split_words(docx_path):
    doc = Document(docx_path)
    full_text = ""

    # خواندن تمام متن از فایل ورد
    for para in doc.paragraphs:
        full_text += para.text + " "

    # جدا کردن متن به کلمات
    words = full_text.strip().split()

    # گروه‌بندی کلمات به بخش‌های 10 کلمه‌ای
    chunks = [" ".join(words[i:i + 30]) for i in range(0, len(words), 30)]
    return chunks


# تابع برای ذخیره داده‌ها در فایل اکسل
def save_to_excel(chunks, excel_path):
    # ایجاد یک فایل اکسل جدید
    wb = Workbook()
    ws = wb.active
    ws.title = "Words"

    # نام ستون
    ws.append(["sentence"])

    # ذخیره هر بخش 10 کلمه‌ای در یک سطر
    for chunk in chunks:
        ws.append([chunk])

    # ذخیره فایل اکسل
    wb.save(excel_path)


# تابع اصلی
def process_docs_to_excel(docx_folder, excel_path):
    all_chunks = []

    # پردازش تمام فایل‌های ورد در پوشه مشخص شده
    for filename in os.listdir(docx_folder):
        if filename.endswith(".docx"):
            docx_path = os.path.join(docx_folder, filename)
            chunks = read_and_split_words(docx_path)
            all_chunks.extend(chunks)

    # ذخیره داده‌ها در فایل اکسل
    save_to_excel(all_chunks, excel_path)


# مسیر پوشه فایل‌های ورد و فایل اکسل مقصد
docx_folder = 'dat/httpsirantypist'
excel_path = 'data.xlsx'

# فراخوانی تابع اصلی
process_docs_to_excel(docx_folder, excel_path)
