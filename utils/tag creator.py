import pandas as pd

# خواندن فایل اکسل
file_path = 'C:\\rayanpoyesh_project\\Ghaemi\\dataset\\delete_noise.xlsx'  # مسیر فایل اکسل خود را اینجا قرار دهید

# خواندن همه شیت‌های فایل اکسل
all_sheets = pd.read_excel(file_path, sheet_name=None)

# فرض می‌کنیم ستون‌ها به این نام‌ها باشند
noisy_column = 'noise'  # نام ستون جمله‌های دارای غلط
clean_column = 'origin'  # نام ستون جمله‌های صحیح

# لیست برای ذخیره داده‌های فرمت‌شده
formatted_data = []

# حلقه روی همه شیت‌ها
for sheet_name, df in all_sheets.items():
    print(f"پردازش شیت: {sheet_name}")

    # بررسی اینکه ستون‌ها در هر شیت وجود دارند
    if noisy_column not in df.columns or clean_column not in df.columns:
        raise ValueError(f"اطمینان حاصل کنید که ستون‌ها به درستی در شیت '{sheet_name}' نامگذاری شده‌اند: {noisy_column}, {clean_column}")

    # حلقه روی ردیف‌های هر شیت
    for index, row in df.iterrows():
        if len(formatted_data) >= 8000:
            break  # توقف در صورت رسیدن به ۸۰۰۰ جمله
        noisy_sentence = row[noisy_column]
        clean_sentence = row[clean_column]

        # فرمت‌دهی به سبک GPT-2
        formatted_line = f"<|input|> {noisy_sentence} <|sep|> {clean_sentence} <|endoftext|>"
        formatted_data.append(formatted_line)

# ذخیره داده‌ها در فایل متنی
output_file = 'summary7_data.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for line in formatted_data[:8000]:
        f.write(line + '\n')

print(f"داده‌ها با موفقیت به {output_file} ذخیره شدند.")