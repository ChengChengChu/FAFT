import torch
import argparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from datasets import Dataset
import csv
from trl import SFTTrainer
import os
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig, 
    Trainer,
    TrainingArguments,
    set_seed
)

def set_env():
    os.environ['HUGGINGFACE_HUB_CACHE'] = '/work/u5273929/huggingface_hub'
    os.environ['HF_HOME'] = '/work/u5273929/huggingface_hub'
    os.environ['WANDB_CACHE_DIR']='/work/u5273929/'
    os.environ['WANDB_PROJECT']='chatbot'
    os.environ['HF_DATASETS_CACHE']='/work/u5273929/huggingface_hub'

def set_arguments(parser):
    # parser.add_argument("--filename", type=str, default="/work/u5273929/tmp_RLCD/tmp/MedQA_1000.csv") 
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    # parser.add_argument("--save_path", type=str, default="MedQA_rewrite.csv") # save path

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--peft", action='store_true', default=False)
    parser.add_argument("--dataset", type=str, default='/work/u5273929/tmp_RLCD/tmp/MedQA_1000_reformat.csv')
    parser.add_argument("--type", type=str, default='llama2_rewrite')
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--output_path", type=str, default='example')
    # parser.add_argument('--flash_attn', type=str2bool, default=True if 'H100' in torch.cuda.get_device_name(0) else False)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1.41e-5)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--bz', type=int, default=1)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--report_to', type=str, default='wandb')
    parser.add_argument('--wandb_run_name', type=str, default='example')
    parser.add_argument('--gradient_checkpoint', action='store_true', default=False)

    args = parser.parse_args()

    return args

def create_model(model_id) :
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        config=model_config,
        torch_dtype= torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",  # Pass the device_map parameter here
    )
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)

    set_env()

    model, tokenizer = create_model(args.model_id)
    dest_dict = os.path.join('results', args.type)
    os.makedirs(dest_dict, exist_ok=True)
    output_dir = os.path.join(dest_dict, args.output_path)

    ### Prepare training dataset
    dataset = pd.read_csv(r"/work/u5273929/tmp_RLCD/tmp/MedQA_1000_reformat.csv")
    dataset['text'] = '<s>[INST] ' + dataset['Question'] + ' [/INST] ' + dataset['Answer'] + ' </s>'
    train_dataset = Dataset.from_pandas(dataset[['text']])

    eval_dataset = None

    training_args = TrainingArguments(
        per_device_train_batch_size=args.bz,
        # per_device_eval_batch_size=4 * args.bz,
        prediction_loss_only=True,
        # eval_steps=50,
        gradient_accumulation_steps= 32 // args.bz,
        gradient_checkpointing=args.gradient_checkpoint,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        num_train_epochs=args.epoch,
        do_train=True,
        do_eval=False,
        # evaluation_strategy='steps',
        weight_decay=0.1,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        report_to=args.report_to,
        run_name=args.wandb_run_name,
        save_strategy='epoch',
        output_dir=args.output_path,
    )

    if True:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","v_proj"]
        )
    else:
        peft_config = None

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        # formatting_func=format_alpaca_function if 'alpaca' in args.dataset else format_dolly_function,
        # data_collator=collator,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        dataset_num_proc=4,
        args=training_args,
        peft_config=peft_config
    )

    trainer.train()


if __name__ == "__main__":
    main()    

