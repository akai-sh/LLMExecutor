import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch
import re
import ast
from joblib import load

def read_file(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def extract_result(result):
    out_lins = re.findall(r"Line:\s*(\d+)", result)
    if len(out_lins) > 1:
        return None
    pattern = r"Line:\s*(\d+)\s*Analysis:.*?Check:\s*({.*?})\s*Next statement:\s*(\d+|\w+)"
    matches = re.findall(pattern, result, re.DOTALL)
    if len(matches) != 1:
        return None
    current_line = int(matches[0][0])
    try:
        program_state = ast.literal_eval(matches[0][1])
    except:
        return None
    if matches[0][2] == 'completion' or matches[0][2] == 'Completion':
        next_line = 'completion'
    else:
        try:
            next_line = int(matches[0][2])
        except:
            return None
    return (current_line, program_state, next_line)

def check(answer, next_line_gt, subsequent_values_gt):
    if answer == None:
        return False
    current_line, program_state, next_line = answer

    if [current_line, program_state] == subsequent_values_gt[0] and next_line == next_line_gt:
        return True
    else:
        return False

def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def compute_relative_features(df):
    grouped = df.groupby('sample_id')
    global_stats = grouped[['max', 'min', 'mean', 'size']].agg({
        'max': ['max', 'min'],
        'min': ['max', 'min'],
        'mean': ['max', 'min'],
        'size': 'sum'
    })
    global_stats.columns = ['global_max_max', 'global_max_min',
                            'global_min_max', 'global_min_min',
                            'global_mean_max', 'global_mean_min',
                            'global_size_sum']
    df = df.merge(global_stats, on='sample_id', how='left')

    df['rel_max'] = (df['max'] - df['global_max_min']) / (df['global_max_max'] - df['global_max_min'] + 1e-6)
    df['rel_min'] = (df['min'] - df['global_min_min']) / (df['global_min_max'] - df['global_min_min'] + 1e-6)
    df['rel_mean'] = (df['mean'] - df['global_mean_min']) / (df['global_mean_max'] - df['global_mean_min'] + 1e-6)
    df['rel_size'] = df['size'] / df['global_size_sum']

    df['inv_std'] = 1 / (df['std'] + 1e-6)

    df['rel_min_sq'] = df['rel_min'] ** 2

    return df

def predict_best_group(new_sample_df, model):
    feature_cols = ['rel_max', 'rel_mean', 'inv_std', 'rel_size', 'rel_min_sq']
    new_sample_df = compute_relative_features(new_sample_df)
    features = new_sample_df[feature_cols]

    probas = model.predict_proba(features)
    scores = probas[:, 1]

    new_sample_df['score'] = scores
    best_group = new_sample_df.loc[new_sample_df['score'].idxmax()]
    return best_group['group_id'], new_sample_df

def get_most_common(extracted):
    freq = {}
    for i,e in enumerate(extracted):
        flag=False
        if i==0:
            freq[i]=[]
        else:
            for k in freq.keys():
                if extracted[k]==extracted[i]:
                    freq[k].append(i)
                    flag=True
                    break
            if not flag:
                freq[i] = []
    return freq

parser = argparse.ArgumentParser()
parser.add_argument("--executor_model_path", default='', type=str)
parser.add_argument("--tokenizer_path", default='Qwen/Qwen2.5-Coder-7B-Instruct', type=str)
parser.add_argument("--reward_model_path", default='', type=str)
parser.add_argument("--rf_model_path", default='', type=str)
parser.add_argument("--results_path", default='', type=str)
parser.add_argument("--data_path", default='', type=str)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num_sequences", default=32, type=int)
args = parser.parse_args()

print(args.executor_model_path)
print(args.reward_model_path)
print(args.rf_model_path)
print(args.data_path)
print(args.results_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# step 1: run executor
tokenizer = AutoTokenizer.from_pretrained(
    args.executor_model_path,
    padding_side="left",
    trust_remote_code=True
)

executor = AutoModelForCausalLM.from_pretrained(
    args.executor_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def process_batch(batch):
    if 'llama' in args.executor_model_path or 'Llama' in args.executor_model_path:
        inputs = [item["prompt"] for item in batch]
    else:
        inputs = [
            tokenizer.apply_chat_template([
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": item["prompt"]}
            ], add_generation_prompt=True, tokenize=False)
            for item in batch
        ]

    inputs = tokenizer(
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(executor.device)

    outputs = executor.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=args.num_sequences,
    )

    batch_responses = []
    for i in range(len(batch)):
        start_idx = i * args.num_sequences
        end_idx = start_idx + args.num_sequences
        sample_outputs = outputs[start_idx:end_idx]

        decoded_responses = [
            tokenizer.decode(seq[len(inputs.input_ids[i]):], skip_special_tokens=True)
            for seq in sample_outputs
        ]
        batch_responses.append(decoded_responses)

    return batch_responses


test_data = read_file(args.data_path)

progress_bar = tqdm(total=len(test_data), desc="Processing items")

items_with_outputs=[]
with open(args.results_path, "a+") as result_file:

    for batch in batch_generator(test_data, args.batch_size):
        responses = process_batch(batch)

        for item, item_responses in zip(batch, responses):

            item["model_outputs"] = item_responses
            items_with_outputs.append(item)
            # result_file.write(json.dumps(item) + "\n")

        progress_bar.update(len(batch))

progress_bar.close()

# step 2: run reward model
device='cuda'
tokenizer_qwen_7b = AutoTokenizer.from_pretrained(args.tokenizer_path)
tokenizer_qwen_7b.padding_side = "left"
tokenizer_qwen_7b.model_max_length = 2048

rewaed_model = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_path, num_labels=1, torch_dtype=torch.bfloat16
)
rm_pipe = pipeline(
    "sentiment-analysis",
    model=rewaed_model,
    # device="auto",
    device=device,
    tokenizer=tokenizer_qwen_7b,
    model_kwargs={"torch_dtype": torch.bfloat16}
)
rewaed_model.to(device)
pipe_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": args.batch_size
}

items_with_rewards=[]
for data_item in tqdm(items_with_outputs):
    answers = data_item['model_outputs']
    prompt = data_item['prompt']
    texts = []
    for answer in answers:
        text = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]
        texts.append(text)
    test_texts = [tokenizer_qwen_7b.apply_chat_template(text, tokenize=False, add_generation_prompt=False) for text in
                  texts]
    # outputs=
    pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
    rewards = [output[0]["score"] for output in pipe_outputs]
    data_item['rewards'] = rewards
    items_with_rewards.append(data_item)

# step 3: run random forest model
rf_model = load(args.model_path)
feature_cols = ['rel_max', 'rel_mean', 'inv_std', 'rel_size', 'rel_min_sq']

with open(args.results_path,'w')as w:
    num_true = 0
    sum = len(items_with_rewards)
    for i, item in enumerate(items_with_rewards):
        data = []
        answers = item['model_outputs']
        scores = item['rewards']
        extracted = [extract_result(answer) for answer in answers]
        freq = get_most_common(extracted)
        for k in freq.keys():
            group_scores = []
            group_scores.append(scores[k])
            for va in freq[k]:
                group_scores.append(scores[va])
            max_val = np.max(group_scores)
            min_val = np.min(group_scores)
            mean_val = np.mean(group_scores)
            std_val = np.std(group_scores)
            size_val = np.size(group_scores)
            result = extract_result(answers[k])

            data.append([i, k, max_val, min_val, mean_val, std_val, size_val])
        sample = pd.DataFrame(data, columns=['sample_id', 'group_id', 'max', 'min', 'mean', 'std', 'size'])
        best_group_id, sample_new = predict_best_group(sample, rf_model)
        final_answer = answers[int(best_group_id)]
        item['model_output']=final_answer
        result = extract_result(final_answer)
        if check(result, item['next_line'], item['subsequent_values']):
            num_true += 1
            item['flag']=1
        else:
            item['flag'] = 0
        w.write(json.dumps(item)+'\n')
print(num_true)
print(sum)
print(num_true / sum)