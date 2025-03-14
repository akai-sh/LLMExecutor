import argparse
import ast
import json
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--data_path",
                    default='./../../data/train/RVBS/rf_codellama.jsonl',
                    type=str,
                    help="path of input data")
parser.add_argument("--model_path",
                    default='./../../fine_tuned_models/random_forest_codellama.joblib',
                    type=str,
                    help="path to save processed data")
args = parser.parse_args()

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

def generate_sample_data():
    num=0
    data = []
    with open(args.data_path,'r')as r:
        lines = r.readlines()
        for i,l in enumerate(lines):
            item=json.loads(l)
            answers = item['model_output']
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
                if check(result, item['next_line'], item['subsequent_values']):
                    label = 1
                else:
                    label = 0
                data.append([i, k, max_val, min_val, mean_val, std_val, size_val, label])
    print(num)
    return pd.DataFrame(data, columns=['sample_id', 'group_id', 'max', 'min', 'mean','std', 'size', 'label'])



df = generate_sample_data()


# calculate features
def compute_relative_features(df):
    # calculate gloable features for every sample
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

    # calculate relative feature
    df['rel_max'] = (df['max'] - df['global_max_min']) / (df['global_max_max'] - df['global_max_min'] + 1e-6)
    df['rel_min'] = (df['min'] - df['global_min_min']) / (df['global_min_max'] - df['global_min_min'] + 1e-6)
    df['rel_mean'] = (df['mean'] - df['global_mean_min']) / (df['global_mean_max'] - df['global_mean_min'] + 1e-6)
    df['rel_size'] = df['size'] / df['global_size_sum']


    df['inv_std'] = 1 / (df['std'] + 1e-6)


    df['rel_min_sq'] = df['rel_min'] ** 2

    return df


df = compute_relative_features(df)
# 生成样本数据（保持相同的数据处理流程）
df_ = generate_sample_data()
df = compute_relative_features(df_)


feature_cols = ['rel_max', 'rel_mean', 'inv_std', 'rel_size', 'rel_min_sq']
X = df[feature_cols]
y = df['label']

# train random forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    class_weight='balanced'
)
model.fit(X, y)

# evaluate
y_pred = model.predict(X)
print("\nClassification Report:")
print(classification_report(y, y_pred))

from joblib import dump
dump(model, args.model_path)
