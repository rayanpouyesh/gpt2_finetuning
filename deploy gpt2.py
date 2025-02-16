import torch
import time  # Import the time module
from transformers import Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, AutoTokenizer , GPT2Tokenizer
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import yaml

def display_farsi_text(sentence: str):
    return (get_display(reshape(sentence)))

#----------------------------------------------------------------------------------
# Load configuration from YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config_distilgpt2.yaml')
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Set working directory
import os
new_path = "C:/rayanpoyesh_project/Ghaemi/byt5-spell-corrector"
os.chdir(new_path)
print(os.getcwd())
#----------------------------------------------------------------------------------

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define max length parameter
max_lenght = int(config['model']['max_seq_length'])

# Paths to the model and tokenizer
model_path     = "./results/checkpoint-1600"
tokenizer_path = "./final_tokenizer"

# Load the model and tokenizer
start_time = time.time()
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
end_time = time.time()
model_loading_time = end_time - start_time

# # Setting PAD token
# # tokenizer.pad_token = tokenizer.eos_token  # or
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#
# # Setting PAD token
# tokenizer.pad_token = '[PAD]'

from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print ('----------------------- Model Loaded: '+model_path+' --------------------------------------')
print ('----------------------- Tokenizer Loaded: '+tokenizer_path+' --------------------------------------')
print(f"----------------------- Model Tokenizer Loading Time: {model_loading_time:.2f} seconds -----------------")
print(f'----------------------- Device: {device} --------------------------------------')
txt=""
while txt != "<ext>":
    txt = input(display_farsi_text('Input text:: '))
    txt = f"<|input|> {txt} <|sep|>"
    print('Input length:: ', len(txt))
    start_response_time = time.time()
    inputs = tokenizer(txt, return_tensors="pt", padding=True, truncation=True, max_length=max_lenght).to(device)
    #----------------------------------------------------------------------
    limited_len = int(inputs['input_ids'].shape[1])  # Limit output length to input length
    print('------------------------------ Limited length: ', limited_len ,'--------------------------------------')
    #----------------------------------------------------------------------
    # Generate predictions
    outputs = model.generate(
        **inputs,
        max_length=limited_len*2,   # Set maximum output length
        min_length=limited_len,     # Set minimum output length
        num_beams=1,                # Number of beams for beam search
        early_stopping=True,        # Enable early stopping
        length_penalty=1.0,         # Penalty for output length
        temperature=0.0             # Temperature to control output randomness
    )
    # Decode the predictions
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    end_response_time = time.time()
    response_model_time = end_response_time - start_response_time
    print(f"Response model time: {model_loading_time:.2f} seconds")

    print('Output length:: ', len(corrected_text))
    print('Reference code:')
    print(txt)
    print('Corrected text:')

    def extract_clean_text(model_output):
        # Extract clean text between <|sep|> and <|endoftext|>
        segments = model_output.split('<|sep|>')

        if len(segments) < 2:
            return model_output.strip()  # If <|sep|> tag is not found

        clean_part = segments[-1].split('<|endoftext|>')[0]

        # Remove extra spaces and characters
        clean_text = clean_part.strip()

        # Remove </s> tag if exists
        clean_text = clean_text.replace('</s>', '').strip()

        return clean_text

    corrected_text = extract_clean_text(corrected_text)
    print(corrected_text)

print('End of test.-------------------------------------------------')
