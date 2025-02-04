
'''
python generation/generate_eval_alphca.py --model_name_or_path MODEL_NAME --output_name_or_path FILE_NAME


'''
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
    output_length: Optional[int] = field(
        default=4096,
        metadata={"help": "the location of the SFT model name or path"},
    )
    temperature: Optional[float] = field(
        default=0.6,
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
    temperature=script_args.temperature,
    top_p=0.9,
    max_tokens=script_args.output_length,
    n=1,
    stop_token_ids=[tokenizer.eos_token_id],
    #stop=["<|user|>"],
)


ds = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
batch_size = 100
messages_list = [[{"role": "user", "content": a}] for a in ds['instruction']]
#messages_short = [a for a in ds['instruction']]
prompt_token = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]
print(prompt_token[0])


outputs = llm.generate(prompt_token,
                sampling_params=sampling_params,
                use_tqdm=True)
generated_text_1 = [output.outputs[0].text for output in outputs]


prompts = [row for row in ds['instruction']]




data = [{"instruction": p, "output": o1, "generator": script_args.output_name_or_path} 
        for p, o1 in zip(prompts, generated_text_1)]


print(len(data))



with open(f"alphca_eval_{script_args.output_name_or_path}.json", "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)


