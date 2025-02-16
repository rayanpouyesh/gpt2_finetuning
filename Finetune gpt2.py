import os
import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer , GPT2Tokenizer
import logging
import yaml

# #logging settings
# logging.basicConfig(
#     filename="C:\\rayanpoyesh_project\\Ghaemi\\byt5-spell-corrector\\logs\\training.log",
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# Load configuration from YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config_distilgpt2.yaml')

# Device settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---------------------Using device: {device}----------------------")

# Load data from multi text file
def load_data_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def load_data_from_directory(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            data.extend(load_data_from_txt(file_path))
    return data

# Directory containing text files
input_directory = "C:\\rayanpoyesh_project\\Ghaemi\\dataset\\New folder"

data = load_data_from_directory(input_directory)

print(f"Total lines loaded: {len(data)}")

#-----------------------------------------------------------------------------------

# Load model and tokenizer
model_path = "final_model"
tokenizer_path = "final_tokenizer"

# Load the pretrained model from the local directory
model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

# Add special tokens to the tokenizer and update the model's token embeddings
tokenizer.add_special_tokens({"additional_special_tokens": ["<|input|>", "<|sep|>", "<|endoftext|>"]})
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
#----------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

# Split data into training and evaluation sets
X_train, X_eval = train_test_split(data, test_size=0.2, random_state=42)

# Define max length parameter
max_length = int(config['model']['max_seq_length'])

# Tokenize data with labels
train_inputs = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
train_inputs["labels"] = train_inputs["input_ids"].clone()

eval_inputs = tokenizer(X_eval, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
eval_inputs["labels"] = eval_inputs["input_ids"].clone()

# Define training parameters
output_dir = config['training']['output_dir']
epochs = int(config['training']['epochs'])
batch_size = int(config['training']['batch_size'])
gradient_steps = int(config['training']['gradient_steps'])
lr = float(config['training']['learning_rate'])
logging_steps = int(config['training']['logging_steps'])
save_strategy = config['training']['save_strategy']
evaluation_strategy = config['training']['evaluation_strategy']
# eval_steps = int(config['training']['eval_steps'])
save_total_limit = int(config['training']['save_total_limit'])
save_steps = int(config['training']['save_steps'])

#--------------------------------Extend the TextDataset class to handle evaluation data:
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

train_dataset = TextDataset(train_inputs)
eval_dataset = TextDataset(eval_inputs)

# Use DataCollator to manage masking and improve training process
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is a non-masked language model
)

# Calculate warmup_steps
num_training_steps = (len(train_dataset) // batch_size // gradient_steps) * epochs
warmup_steps = int(0.1 * num_training_steps)

print(f"warmup{warmup_steps}")

#---------------------------------Update the TrainingArguments to include evaluation-related parameters:-------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_steps,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    learning_rate=lr,
    logging_dir='./logs',
    logging_steps=logging_steps,
    save_strategy=save_strategy,
    evaluation_strategy=evaluation_strategy,
    # eval_steps=eval_steps,
    save_total_limit=save_total_limit,
    load_best_model_at_end=True,
    save_steps=save_steps , # You can adjust this to your preferred saving frequency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator
)

# Resume training from the last checkpoint if available
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]  # Get the latest checkpoint

# Train model
print('--------------------------- resume_from_checkpoint -----------------------------------')
print(last_checkpoint)

print('-----------------------------------------------------------------------------------------')

if last_checkpoint:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Save the final model and tokenizer
model_save_path = './final_model'
tokenizer_save_path = './final_tokenizer'
# Save the trained model
model.save_pretrained(model_save_path, from_pt=True)
print(f"Model saved to {model_save_path}")
# Save the tokenizer
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")

# Evaluate the model after training
evaluation_results = trainer.evaluate()
print("Evaluation Results:", evaluation_results)
