# -*- coding: utf-8 -*-
"""
DPO Fine-tuning of Gemma-2B on FreightSense dispatcher preferences
Uses TRL (Transformer Reinforcement Learning) library
Requires: pip install trl transformers bitsandbytes peft accelerate datasets
GPU: Google Colab T4 (free) - Gemma-2B with 4-bit quantisation fits in 8GB

NOTE: Before running, you must:
1. Create a Hugging Face account (free) at huggingface.co
2. Request access to google/gemma-2b-it (approved instantly)
3. Create a free HF token at huggingface.co/settings/tokens
4. Run: huggingface-cli login
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType

def main():
    print("🚀 Starting DPO Fine-Tuning Pipeline for Gemma-2B...")
    
    # 1. Export Data
    from src.output.feedback_collector import export_for_dpo_training
    data_path = os.path.join("data", "dpo_training_data.json")
    export_for_dpo_training(data_path)
    
    # Load dataset
    try:
        dataset = load_dataset("json", data_files=data_path, split="train")
        print(f"Loaded {len(dataset)} preference pairs.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # 2. Model Configuration
    model_id = "google/gemma-2b-it"
    
    # 4-bit Quantisation (QLoRA) to fit on free T4 GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading tokenizer and model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Gemma requires pad token to be set if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # We need a reference model for DPO (usually the same base model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    # 3. LoRA Configuration (Low-Rank Adaptation)
    # We only train adapters, not the full 2B parameters
    peft_config = LoraConfig(
        r=16, # Rank of the update matrices (higher = more capacity, more memory)
        lora_alpha=32, # Scaling factor
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Attention layers
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. DPO Configuration
    # beta controls the KL penalty (how much we allow deviation from reference model)
    dpo_args = DPOConfig(
        beta=0.1, 
        per_device_train_batch_size=1, # 1 for 8GB VRAM
        gradient_accumulation_steps=4, # effectively batch size 4
        num_train_epochs=3,
        learning_rate=5e-5,
        remove_unused_columns=False,
        output_dir="models/gemma_dpo_adapter",
        logging_steps=10,
        max_prompt_length=256,
        max_length=512,
        optim="paged_adamw_8bit" # memory efficient optimizer
    )
    
    # 5. Training
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    # 6. Save Adapter
    print("Saving final adapter...")
    trainer.model.save_pretrained("models/gemma_dpo_adapter")
    tokenizer.save_pretrained("models/gemma_dpo_adapter")
    print("✅ DPO Fine-tuning complete!")

if __name__ == "__main__":
    # Ensure this is only run if intended (e.g. on Colab)
    print("Note: This script is intended to be run on Google Colab or a machine with a GPU.")
    # main()
