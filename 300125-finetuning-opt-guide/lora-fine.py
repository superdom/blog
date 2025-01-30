import os
import torch
import argparse
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES'))

# Argument Parsing
parser = argparse.ArgumentParser(description="Train a model with configurable parameters.")
parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Hugging Face model ID.")
parser.add_argument("--model_type", type=str, choices=["base", "it"], default="base", help="Model type for prompt template.")
parser.add_argument("--bs", type=int, default=1, help="Batch size per device.")
parser.add_argument("--gc", type=int, default=2, help="Gradient accumulation steps.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--lrs", type=str, choices=["linear", "cosine"], default="linear", help="Learning rate scheduler.")
parser.add_argument("--ep", type=int, default=3, help="Number of epochs.")
parser.add_argument("--lorar", type=int, default=16, help="LoRA Rank.")
parser.add_argument("--loraa", type=int, default=32, help="LoRA Alpha.")
args = parser.parse_args()

MODEL_ID = args.model
MODEL_TYPE = args.model_type

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

# Login to Hugging Face
login(token=HF_TOKEN)

# Load Model & Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_eos_token=False)

# Tokenizer settings by model
if 'qwen' in MODEL_ID.lower():
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.pad_token_id = 151645
    tokenizer.padding_side = 'left'
elif 'llama' in MODEL_ID.lower():
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.pad_token_id = 128004
    tokenizer.padding_side = 'right'
else:
    tokenizer.padding_side = 'right'

# Load & Process Dataset 
dataset = load_dataset("beomi/KoAlpaca-v1.1a").shuffle(seed=42)

def generate_prompt(example):
    eos = tokenizer.eos_token  # Always append EOS manually
    if MODEL_TYPE == "it":
        # Chat template for "it"
        messages = [
            {
                "role": "user",
                "content": example['instruction'].strip()
            },
            {
                "role": "assistant",
                "content": example['output'].strip()
            }
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ) + eos
    elif MODEL_TYPE == "base":
        # Default base template
        return f"### Question: {example['instruction']}\n### Answer: {example['output']}{eos}"
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Supported types are 'base' and 'it'.")

# Preprocess dataset
dataset = dataset.map(
    lambda x: {'text': generate_prompt(x)},
    num_proc=min(os.cpu_count(), 8),
    remove_columns=['instruction', 'output']
)

split_data = dataset['train'].train_test_split(test_size=0.05, seed=42)
train_data, val_data = split_data['train'], split_data['test']

# Debugging tokenization and decoding process for verifying data processing
print("======================= Verify Data Processing ===================================")
# Print Origin Text
print("[DEBUG] Original text:", train_data[1]['text'])

# Check tokenized results
encoded = tokenizer(train_data[1]['text'], return_tensors='pt')
print("[DEBUG] Token IDs:", encoded['input_ids'][0].tolist())
print("[DEBUG] Last token ID:", encoded['input_ids'][0][-1].item())
print("[DEBUG] EOS token ID:", tokenizer.eos_token_id)

# Check decoded results
decoded = tokenizer.decode(encoded['input_ids'][0])
print("[DEBUG] Decoded text:", decoded)
print("============================================================================")

# Define LoRA configuration
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=args.lorar,
    lora_alpha=args.loraa,
    lora_dropout=0.1,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
)

# Training Arguments
output_dir = f"./outputs/{MODEL_ID.split('/')[-1]}-{MODEL_TYPE}-{args.bs}-{args.gc}-{args.lr}-{args.lrs}-{args.lorar}-{args.loraa}"

train_args = TrainingArguments(
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.gc,
    gradient_checkpointing=True,
    num_train_epochs=args.ep,
    learning_rate=args.lr,
    lr_scheduler_type=args.lrs,
    weight_decay=0.01,
    bf16=True,
    warmup_ratio=0.1,
    optim="adamw_torch",
    seed=42,
    output_dir=output_dir,
    logging_steps=2,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    report_to="wandb",
    log_level="debug",
)

# Trainer Setup
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    dataset_text_field="text",
    max_seq_length=1024,
    packing=False,
    args=train_args,
)

# Set up Weights & Biases logging
run_name = f"{MODEL_ID.split('/')[-1]}-{MODEL_TYPE}-{args.bs}-{args.gc}-{args.lr}-{args.lrs}-{args.lorar}-{args.loraa}"

wandb.login(key=WANDB_API_KEY)
wandb.init(
    project="finetune-test",
    name=run_name,
    config={
        "learning_rate": args.lr,
        "batch_size": args.bs,
        "gradient_accumulation_steps": args.gc,
        "num_train_epochs": args.ep
    }
)

# Train Model
model.config.use_cache = False
trainer.train()

# Save Model
save_path = f"./model/{MODEL_ID.split('/')[-1]}-{MODEL_TYPE}-{args.bs}-{args.gc}-{args.lr}-{args.lrs}-{args.lorar}-{args.loraa}"
trainer.save_model(save_path)
print(f"Model saved at {save_path}")
