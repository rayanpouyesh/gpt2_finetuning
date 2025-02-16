import pandas as pd

def shuffle_rows(df):
    """
    این تابع ترتیب سطرهای هر دو ستون 'origin' و 'noise' را به طور تصادفی تغییر می‌دهد
    ولی همچنان داده‌های هر سطر با هم در کنار هم می‌ماند.
    """
    # جابجایی تصادفی سطرها
    shuffled_df = df.sample(frac=1).reset_index(drop=True)
    return shuffled_df

# بارگذاری داده‌ها
file_path = "C:\\rayanpoyesh_project\\Ghaemi\\noise.xlsx"  # مسیر فایل اکسلی شما
df = pd.read_excel(file_path)

# اطمینان از اینکه فقط ستون‌های 'origin' و 'noise' جابجا شوند
df_shuffled = shuffle_rows(df[['origin', 'noise']])

# نمایش نمونه‌ای از داده‌ها
print(df_shuffled.head())

# ذخیره داده‌ها به یک فایل جدید
df_shuffled.to_excel("C:\\rayanpoyesh_project\\Ghaemi\\test2.xlsx", index=False)
