# LLMExeutor
This repo provides the code for reproducing the experiments in LLMExeutor, which fine-tuning smaller LLMs with explanations as program executors.
## prepare envirentments
```
conda create -n llmexecutor python=3.9
conda activate llmexecutor
cd LLMExecutor
pip install -r requirements.txt
```
## prepare data
Please download the dataset from [here](https://drive.google.com/drive/folders/1mdfxpxl_PNjpo_cbHBQlQ7tQLmfJW55W?usp=drive_link) and place the `data` folder inside the `LLMExecutor` folder 
## training Executor
We fine-tuned three variants (SFT, SWA, RVBS) of LLMs, including: `meta-llama/CodeLlama-7b-Instruct-hf`, `meta-llama/CodeLlama-13b-Instruct-hf`, `Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-Coder-14B-Instruct`
### SFT
#### CodeLlama:

```
cd train/Executor
# process data
python process_data_codellama.py --data_path ./../../data/train/Executor/sft.jsonl --save_path ./../../data/train/Executor/dataset_codellama
# CodeLlama-7b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_7b_sft --config_name meta-llama/CodeLlama-7b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-7b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-7b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/Executor/dataset_codellama
# CodeLlama-13b
python train_codellama.py --output_dir ./../../fine_tuned_models/codellama_13b_sft --config_name meta-llama/CodeLlama-13b-Instruct-hf --tokenizer_name meta-llama/CodeLlama-13b-Instruct-hf --model_name_or_path meta-llama/CodeLlama-13b-Instruct-hf --max_target_length 1024 --max_source_length 1024 --pad_to_max_length true --do_train true --learning_rate 1e-5 --lr_scheduler_type cosine --logging_steps 2 --num_train_epochs 3 --save_steps 1000 --per_device_train_batch_size 3 --overwrite_output_dir false --train_data ./../../data/train/Executor/dataset_codellama
```

#### Qwen2.5-Coder:


```
# process data
python process_data_qwen.py --data_path ./../../data/train/Executor/sft.jsonl --save_path ./../../data/train/Executor/sft_formated_qwen.jsonl
python binarize_data.py --input_path ./../../data/train/Executor/sft_formated_qwen.jsonl --output_path ./../../data/train/Executor/sft_processed_qwen.jsonl
# Qwen2.5-Coder-7b
python train.py     --model_name_or_path  Qwen/Qwen2.5-Coder-7B-Instruct    --data_path ./../../data/train/Executor/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_7b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "tensorboard"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True
# Qwen2.5-Coder-14b
python train.py     --model_name_or_path  Qwen/Qwen2.5-Coder-14B-Instruct    --data_path ./../../data/train/Executor/sft_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_14b_sft     --num_train_epochs 5     --per_device_train_batch_size 1    --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000    --learning_rate 1e-5    --weight_decay 0.0    --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "tensorboard"     --bf16 False    --tf32 False     --fp16 True     --truncate_source True 
```

### SWA
for SWA, we only including `Qwen/Qwen2.5-Coder-14B-Instruct`

```
python process_data_qwen.py --data_path ./../../data/train/Executor/swa.jsonl --save_path ./../../data/train/Executor/swa_formated_qwen.jsonl
python binarize_data.py --input_path ./../../data/train/Executor/sft_formated_qwen.jsonl --output_path ./../../data/train/Executor/swa_processed_qwen.jsonl
python train.py     --model_name_or_path  Qwen/Qwen2.5-Coder-14B-Instruct     --data_path ./../../data/train/Executor/swa_processed_qwen.jsonl.npy     --model_max_length 1280     --output_dir ./../../fine_tuned_models/qwen_14b_swa     --num_train_epochs 5     --per_device_train_batch_size 1     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 50     --save_total_limit 1000     --learning_rate 1e-5     --weight_decay 0.0     --warmup_steps 100    --lr_scheduler_type "cosine"     --logging_strategy "steps"    --logging_steps 1     --report_to "tensorboard"     --bf16 False    --tf32 False    --fp16 True     --truncate_source True 
```
### RVBS
CodeLlama
```
## train reward model codellama
python rm.py --model_name Qwen/Qwen2.5-Coder-7B-Instruct --output_path ./../../fine_tuned_models/rm_codellama --save_every_steps 100 --eval_every_steps 100 --train_set_path ./../../data/train/RVBS/rm_codellama_train.jsonl --valid_set_path ./../../data/train/RVBS/rm_codellama_valid.jsonl

##train random forest codellama
python train_random_forest.py --data_path ./../../data/train/RVBS/rf_codellama.jsonl --model_path ./../../fine_tuned_models/random_forest_codellama.joblib
```
Qwen2.5-Coder
```
## train reward model qwen
python rm.py --model_name Qwen/Qwen2.5-Coder-7B-Instruct --output_path ./../../fine_tuned_models/rm_qwen --save_every_steps 100 --eval_every_steps 100 --train_set_path ./../../data/train/RVBS/rm_qwen_train.jsonl --valid_set_path ./../../data/train/RVBS/rm_qwen_valid.jsonl

##train random forest qwen
python train_random_forest.py --data_path ./../../data/train/RVBS/rf_qwen.jsonl --model_path ./../../fine_tuned_models/random_forest_qwen.joblib
```
## ASE testing
### SFT
**Models:** `codellama_7b`, `codellama_13b`, `qwen_7b`, `qwen_14b`  
**Datasets:** `codenetmut`, `humaneval`

```bash
for model in "codellama_7b" "codellama_13b" "qwen_7b" "qwen_14b"; do
  for dataset in "codenetmut" "humaneval"; do
    python run_executor.py   --executor_model_path "./../../fine_tuned_models/${model}_sft/checkpoint_xx"   --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"   --data_path "./../../data/test/ASE/${dataset}.jsonl"
    python calculate_ASE.py --results_path "./../../results/ASE/${model}_sft_${dataset}.jsonl"
  done
done
```
### SWA
**Datasets:** `codenetmut`, `humaneval`

```bash
for dataset in "codenetmut" "humaneval"; do
  python run_executor.py \
    --executor_model_path "./../../fine_tuned_models/qwen_14b_swa/checkpoint_xx" \
    --results_path "./../../results/ASE/qwen_14b_swa_${dataset}.jsonl" \
    --data_path "./../../data/test/ASE/${dataset}.jsonl"

  python calculate_ASE_SWA.py --results_path "./../../results/ASE/qwen_14b_swa_${dataset}.jsonl"
done
```
### RVBS
**Model Mapping:**
- codellama_* → codellama
- qwen_* → qwen

```bash
declare -A model_map=([codellama_7b]="codellama" ...)

for model in "codellama_7b" "codellama_13b" "qwen_7b" "qwen_14b"; do
  for dataset in "codenetmut" "humaneval"; do
    python run_executor_RVBS.py \
      --executor_model_path "./../../fine_tuned_models/${model}_sft/checkpoint_xx" \
      --reward_model_path "./../../fine_tuned_models/rm_${model_map[$model]}_checkpoint_xx" \
      --rf_model_path "./../../fine_tuned_models/random_forest_${model_map[$model]}.joblib" \
      --results_path "./../../results/ASE/${model}_RVBS_${dataset}.jsonl" \
      --data_path "./../../data/test/ASE/${dataset}.jsonl"
      
    python calculate_ASE.py --results_path "./../../results/ASE/${model}_RVBS_${dataset}.jsonl"
  done
done
```
### Base Models
| Model          | Pretrained Path                          |
|----------------|------------------------------------------|
| codellama_7b   | meta-llama/CodeLlama-7b-Instruct-hf      |
| codellama_13b  | meta-llama/CodeLlama-13b-Instruct-hf    |
| qwen_7b        | Qwen/Qwen2.5-Coder-7B-Instruct          |
| qwen_14b       | Qwen/Qwen2.5-Coder-14B-Instruct         |

```bash
declare -A model_paths=([codellama_7b]="..." [codellama_13b]="..." ...)

for model in "${!model_paths[@]}"; do
  for dataset in "codenetmut" "humaneval"; do
    python run_executor.py \
      --executor_model_path "${model_paths[$model]}" \
      --results_path "./../../results/ASE/${model}_base_${dataset}.jsonl" \
      --data_path "./../../data/test/ASE/${dataset}.jsonl"
      
    python calculate_ASE.py --results_path "./../../results/ASE/${model}_base_${dataset}.jsonl"
  done
done
```


