import Levenshtein
import pandas as pd
import numpy as np

df = pd.read_excel("byt5-spell-corrector/corrected_text6.1.xlsx")

predict = df['byt5']
correct = df['origin']

def calculate_levenshtein_distance(predicted, correct):
    """
    محاسبه فاصله Levenshtein بین دو رشته و نسبت شباهت.
    """
    distance = Levenshtein.distance(predicted, correct)
    ratio = Levenshtein.ratio(predicted, correct)
    return distance, ratio

# محاسبه فاصله و نسبت Levenshtein برای هر جفت جمله و ذخیره در لیست‌ها
levenshtein_distances = []
levenshtein_ratios = []
for p, c in zip(predict, correct):
    distance, ratio = calculate_levenshtein_distance(p, c)
    levenshtein_distances.append(distance)
    rounded_ratio = round(ratio, 2)
    levenshtein_ratios.append(rounded_ratio)

# اضافه کردن لیست فواصل و نسبت‌ها به عنوان ستون‌های جدید به DataFrame
df['Levenshtein_Distance'] = levenshtein_distances
df['Levenshtein_Ratio'] = levenshtein_ratios

# محاسبه میانگین فاصله و نسبت Levenshtein
average_distance = round(df['Levenshtein_Distance'].mean() , 2 )
average_ratio = round(df['Levenshtein_Ratio'].mean() , 2)

# ایجاد یک سطر جدید به عنوان دیکشنری
new_row = {
    'byt5': 'میانگین',  # یا هر متن دیگری که می‌خواهید نمایش دهید
    'Levenshtein_Distance': average_distance,
    'Levenshtein_Ratio': average_ratio
}

# اضافه کردن سطر جدید به DataFrame
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# ذخیره DataFrame به فایل اکسل
df.to_excel("byt5-spell-corrector/corrected_text6.1_with_levenshtein.xlsx", index=False)

# چاپ میانگین‌ها در کنسول
print(f"میانگین فاصله Levenshtein: {average_distance:.2f}")
print(f"شباهت Levenshtein: {average_ratio:.2f}")