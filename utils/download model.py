from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5ForConditionalGeneration, AutoTokenizer
#
# # بارگذاری مدل و توکن‌ساز
model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
# tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
#
# # مشخص کردن مسیر ذخیره‌سازی
save_model_directory = "C:\\rayanpoyesh_project\\Ghaemi\\byt5-spell-corrector\\byt5_small_model"
# save_toknizer_directory = "C:\\rayanpoyesh_project\\Ghaemi\\byt5-spell-corrector\\gpt2tokenizer"
#
# # ذخیره‌سازی مدل و توکن‌ساز
model.save_pretrained(save_model_directory)
# tokenizer.save_pretrained(save_toknizer_directory)


# import pandas as pd
#
# # مسیر فایل اکسل خود را وارد کنید
# file_path = '.xlsx'
#
# # بارگذاری تمامی شیت‌ها درون یک دیکشنری
# excel_file = pd.ExcelFile(file_path)
#
# # لیستی از نام تمامی شیت‌ها
# sheet_names = excel_file.sheet_names
#
# # لیست برای ذخیره تمامی داده‌ها
# all_data = []
#
# # خواندن داده‌ها از هر شیت و افزودن به لیست
# for sheet in sheet_names:
#     df = pd.read_excel(file_path, sheet_name=sheet)
#     all_data.append(df)
#
# # ترکیب داده‌ها به یک دیتافریم
# combined_data = pd.concat(all_data, ignore_index=True)
#
# # نوشتن دیتافریم ترکیب‌شده به یک شیت جدید در همان فایل
# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     combined_data.to_excel(writer, sheet_name='Combined_Sheet', index=False)
#
# print("داده‌ها با موفقیت ترکیب شدند و در شیت جدید ذخیره شدند.")
