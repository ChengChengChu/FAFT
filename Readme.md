## Prepare training data
Randomly extract 1000 (Q, A) pairs from MedQA dataset, save in data/.
To run data generation: 
```
python3 script/prepare_data.py \
  --filename ***.csv \
  --model_id meta-llama/Llama-2-7b-chat-hf \
  --save_path ***.csv 
```
Parameters: 
* --filename: Question, answer pairs to re-generate.
* --model_id: model path in huggingface, current support LLaMA-2, and LLaMA2-chat.
* --save_path: Path to save generated result.