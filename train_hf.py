"""
train_hf.py - QLoRA Fine-tuning script for RegexGPT (HuggingFace version)

Uses standard HuggingFace PEFT + TRL for training.
Alternative to train.py when Unsloth has compatibility issues.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "./regex_gpt_lora"
DATA_DIR = "./data"

# LoRA Configuration
LORA_R = 16                    # LoRA rank
LORA_ALPHA = 32                # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05            # Dropout for regularization
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
BATCH_SIZE = 2                 # Smaller batch for stability
GRADIENT_ACCUMULATION = 8      # Effective batch = 2 * 8 = 16
NUM_EPOCHS = 3                 # Training epochs
LEARNING_RATE = 2e-4           # Learning rate
MAX_SEQ_LENGTH = 512           # Max sequence length
WARMUP_RATIO = 0.03            # Warmup steps ratio

# ============================================================
# Main Training Script
# ============================================================

def main():
    print("=" * 60)
    print("RegexGPT Fine-Tuning with QLoRA (HuggingFace)")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected! Training will be very slow.")
    
    # --------------------------------------------------------
    # Step 1: Load Model with 4-bit Quantization
    # --------------------------------------------------------
    print("\n[1/4] Loading model with 4-bit quantization...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Model loaded: {MODEL_NAME}")
    
    # --------------------------------------------------------
    # Step 2: Add LoRA Adapters
    # --------------------------------------------------------
    print("\n[2/4] Adding LoRA adapters...")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # --------------------------------------------------------
    # Step 3: Load Dataset
    # --------------------------------------------------------
    print("\n[3/4] Loading dataset...")
    
    train_dataset = load_dataset("json", data_files=f"{DATA_DIR}/train.jsonl", split="train")
    val_dataset = load_dataset("json", data_files=f"{DATA_DIR}/val.jsonl", split="train")
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # --------------------------------------------------------
    # Step 4: Training
    # --------------------------------------------------------
    print("\n[4/4] Starting training...")
    
    from trl import SFTConfig
    
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=True,  # RTX 3090 supports bfloat16
        logging_steps=10,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        optim="paged_adamw_8bit",
        seed=42,
        gradient_checkpointing=True,  # Save VRAM
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # New API in TRL 0.26+
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    
    # Train!
    trainer.train()
    
    # --------------------------------------------------------
    # Save Final Model
    # --------------------------------------------------------
    print("\nSaving model...")
    
    # Save LoRA adapters
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run inference.py to test the model")
    print("  2. Run app.py to launch the Gradio demo")
    
    # Print final training info
    print("\n📊 To view training metrics, run:")
    print(f"  tensorboard --logdir {OUTPUT_DIR}/logs")


if __name__ == "__main__":
    main()
