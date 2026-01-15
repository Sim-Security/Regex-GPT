"""
train_hf.py - QLoRA Fine-tuning script for RegexGPT (HuggingFace version)

Uses standard HuggingFace PEFT + TRL for training.
Alternative to train.py when Unsloth has compatibility issues.

MLflow Integration:
- Tracks hyperparameters, metrics, and model artifacts
- Set MLFLOW_TRACKING_URI environment variable for remote tracking server
- Default: logs to local ./mlruns directory
"""

import os
import json
from datetime import datetime
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import mlflow
import mlflow.pytorch

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

# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "RegexGPT-FineTuning"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")

# ============================================================
# Main Training Script
# ============================================================

def main():
    print("=" * 60)
    print("RegexGPT Fine-Tuning with QLoRA (HuggingFace)")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_memory:.1f} GB")
    else:
        gpu_name = "None"
        gpu_memory = 0
        print("WARNING: No GPU detected! Training will be very slow.")

    # --------------------------------------------------------
    # Initialize MLflow
    # --------------------------------------------------------
    print(f"\n[MLflow] Setting up experiment tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Generate run name with timestamp
    run_name = f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        print(f"[MLflow] Run ID: {run.info.run_id}")
        print(f"[MLflow] Experiment: {MLFLOW_EXPERIMENT_NAME}")

        # Log hyperparameters
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": ",".join(TARGET_MODULES),
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION,
            "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_seq_length": MAX_SEQ_LENGTH,
            "warmup_ratio": WARMUP_RATIO,
            "gpu_name": gpu_name,
            "gpu_memory_gb": gpu_memory,
        })

        # Log system info as tags
        mlflow.set_tags({
            "model_type": "causal_lm",
            "fine_tuning_method": "qlora",
            "framework": "huggingface",
            "task": "regex_generation",
        })

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
        trainable_pct = 100 * trainable_params / total_params
        print(f"Trainable parameters: {trainable_params:,} ({trainable_pct:.2f}%)")

        # Log model info to MLflow
        mlflow.log_params({
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": round(trainable_pct, 2),
        })

        # --------------------------------------------------------
        # Step 3: Load Dataset
        # --------------------------------------------------------
        print("\n[3/4] Loading dataset...")

        train_dataset = load_dataset("json", data_files=f"{DATA_DIR}/train.jsonl", split="train")
        val_dataset = load_dataset("json", data_files=f"{DATA_DIR}/val.jsonl", split="train")

        print(f"Training examples: {len(train_dataset)}")
        print(f"Validation examples: {len(val_dataset)}")

        # Log dataset info to MLflow
        mlflow.log_params({
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
        })

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
            report_to=["tensorboard", "mlflow"],  # Enable MLflow reporting
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
        train_result = trainer.train()

        # Log final training metrics to MLflow
        if train_result.metrics:
            mlflow.log_metrics({
                "final_train_loss": train_result.metrics.get("train_loss", 0),
                "train_runtime_seconds": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
            })

        # Run evaluation and log metrics
        eval_result = trainer.evaluate()
        if eval_result:
            mlflow.log_metrics({
                "final_eval_loss": eval_result.get("eval_loss", 0),
                "eval_runtime_seconds": eval_result.get("eval_runtime", 0),
            })

        # --------------------------------------------------------
        # Save Final Model
        # --------------------------------------------------------
        print("\nSaving model...")

        # Save LoRA adapters
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        # Log model artifacts to MLflow
        print("[MLflow] Logging model artifacts...")
        mlflow.log_artifacts(OUTPUT_DIR, artifact_path="model")

        # Log the LoRA config as artifact
        lora_config_dict = {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": TARGET_MODULES,
            "base_model": MODEL_NAME,
        }
        with open(f"{OUTPUT_DIR}/lora_config.json", "w") as f:
            json.dump(lora_config_dict, f, indent=2)
        mlflow.log_artifact(f"{OUTPUT_DIR}/lora_config.json")

        print(f"\n[MLflow] Run completed! View at: {MLFLOW_TRACKING_URI}")
        print(f"[MLflow] Run ID: {run.info.run_id}")

        print(f"\nTraining complete! Model saved to: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("  1. Run inference.py to test the model")
        print("  2. Run app.py to launch the Gradio demo")

        # Print final training info
        print("\nTo view training metrics:")
        print(f"  TensorBoard: tensorboard --logdir {OUTPUT_DIR}/logs")
        print(f"  MLflow UI:   mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
