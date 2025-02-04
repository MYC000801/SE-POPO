
'''
python generation/generate_eval_test.py --model_name_or_path MODEL_NAME --output_name_or_path FILE_NAME


'''



#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json
import os
from tqdm import tqdm
import random


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    output_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



model_path = script_args.model_name_or_path



print("model_path", model_path)
seed = 42
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=8192,
    load_format="auto",
    seed=seed,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.9,
    max_tokens=4096,
    n=1,
    stop_token_ids=[tokenizer.eos_token_id],
    #stop=["<|user|>"],
)


ds_test = load_dataset("RLHFlow/test_generation_2k", split="train")

ds_1 = load_dataset("RLHFlow/ultrafeedback_iter1", split="train")
ds_2 = load_dataset("RLHFlow/ultrafeedback_iter2", split="train")
ds_3 = load_dataset("RLHFlow/ultrafeedback_iter3", split="train")

#remove the duplication data used in the training process

test_set = ds_test['context_messages']


train_set = ds_1['context_messages']
test_set = [d for d in test_set if d not in train_set]

train_set = ds_2['context_messages']
test_set = [d for d in test_set if d not in train_set]

train_set = ds_3['context_messages']
test_set = [d for d in test_set if d not in train_set]


random.seed(42)

random.shuffle(test_set)
print(len(test_set))




batch_size = 100
messages_list = test_set
prompt_token = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
print(prompt_token[0])


outputs = llm.generate(prompt_token,
                sampling_params=sampling_params,
                use_tqdm=True)

generated_text_1 = [output.outputs[0].text for output in outputs]



prompts = [row[0]['content'] for row in test_set]




data = [{"instruction": p, "output": o1, "generator": script_args.output_name_or_path} 
        for p, o1 in zip(prompts, generated_text_1)]


print(len(data))





with open(f"test_{script_args.output_name_or_path}.json", "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


