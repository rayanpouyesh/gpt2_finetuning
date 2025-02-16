import pandas as pd


def add_sentence_length_column(file_path, column_name, output_file):
    """
    Adds a new column to the Excel file with the length of sentences in the specified column.

    Args:
        file_path (str): Path to the input Excel file.
        column_name (str): Name of the column containing sentences.
        output_file (str): Path to save the updated Excel file.

    Returns:
        None
    """
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Calculate the length of sentences in the specified column
    if column_name in df.columns:
        df['sentence_length2'] = df[column_name].apply(lambda x: len(str(x)) if pd.notnull(x) else 0)
    else:
        print(f"Column '{column_name}' not found in the file.")
        return

    # Save the updated DataFrame back to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"Updated file saved to {output_file}")


# Example usage
input_file = "C:\\rayanpoyesh_project\\Ghaemi\\byt5-spell-corrector\\corrected_text6.1_with_levenshtein.xlsx"  # Path to the input Excel file
output_file = "C:\\rayanpoyesh_project\\Ghaemi\\test15.xlsx"  # Path to save the updated Excel file
column_name = "noise"  # Name of the column containing sentences

add_sentence_length_column(input_file, column_name, output_file)
