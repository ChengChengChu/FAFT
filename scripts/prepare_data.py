import torch
import argparse
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import csv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)

B_INST, E_INST = "<s>[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
Please rewrite the answer of the following question in your own type."""
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS

def get_prompt(instruction):
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

def create_model(model_id) :
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model_config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        config=model_config,
        torch_dtype= torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto",  # Pass the device_map parameter here
    )

    return model, tokenizer

def create_model_input(question, answer) :
    return f"Here is the question answer pair:\nQuestion: {question}\nAnswer: {answer}."


def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)

    qa_pairs = []
    with open(args.filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qa_pairs.append([row['question'].strip(), row['answer'].strip()])
    
    # import pdb
    # pdb.set_trace()
    model, tokenizer = create_model(args.model_id)
    model.eval()
    written_data = []
    col = ['sentence']

    with torch.no_grad():
        for i in tqdm(range(len(qa_pairs))) :
            
            input_text = create_model_input(qa_pairs[i][0], qa_pairs[i][1])
            input_text = get_prompt(input_text)
            input_ids = tokenizer.encode(input_text, return_tensors='pt').to('cuda')
            output = model.generate(input_ids, max_new_tokens=512)
            tmp_reply = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
            written_data.append(tmp_reply.split(':')[-1].strip())
    
    df = pd.DataFrame(columns=col, data=written_data)
    df.to_csv(args.save_path)
    print(f"[INFO]: Generated data save in {args.save_path}.")
    

def set_arguments(parser):
    parser.add_argument("--filename", type=str, default="/work/u5273929/tmp_RLCD/tmp/MedQA_1000.csv") 
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--save_path", type=str, default="MedQA_rewrite.csv") # save path
    

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()    