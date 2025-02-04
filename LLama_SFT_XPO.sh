. "/project/anaconda/etc/profile.d/conda.sh"


# Base paths and settings
initial_model="RLHFlow/LLaMA3-SFT"
base_path="./iter_dpo"
mkdir $base_path
iteration_prefix="Test"




# Function to run a set of operations for a model iteration
run_iteration() {
    local iteration=$1
    local model_path=$2
    local jsonl_input=$3
    local json_output=$4
    local model_output=$5
    local alpha=$6
    local lr=$7
    local sam_model_path=$8

    conda activate vllm
    my_world_size=4
    infer_model=$2
    prompt_dir=$3
    output_dir=$4
    sam_model=$8
    
        CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 2 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 2 --temperature 1.0 --local_index 1 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 2 --temperature 1.0 --local_index 2 --my_world_size ${my_world_size} --eos_ids 128009 &
        CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model}  --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 2 --temperature 1.0 --local_index 3 --my_world_size ${my_world_size} --eos_ids 128009 &    
        wait
            python ./generation/merge_data.py --base_path ${output_dir} --output_dir "${output_dir}_data.json" --num_datasets 4
            accelerate launch annotate_data/get_rewards.py --dataset_name_or_path "${output_dir}_data.json" --output_dir $model_output --K 2



    conda activate rlhflow
    cat <<EOT > dpo_config.yaml
run_name: $iteration
output_dir: $iteration
model_name_or_path: $model_path
ref_model: $model_path
learning_rate: $7
num_train_epochs: 2
logging_steps: 2
gradient_checkpointing: true
do_train: true
do_eval: true
eval_steps: 10000
choose_type: max_random
train_dir: $model_output
eval_dir: $model_output
loss_type: xpo
lr_scheduler_type: cosine
max_length: 2048
max_prompt_length: 1000
eval_strategy: steps
bf16: true
per_device_train_batch_size: 2
per_device_eval_batch_size: 1
gradient_accumulation_steps: 16
report_to: wandb
label_smoothing: 0.0
save_steps: 9999
save_only_model: true 
warmup_ratio: 0.03
save_strategy: 'no'
alpha: $6
EOT

    accelerate launch --config_file ./configs/zero3.yaml dpo_iteration/run_wpo.py dpo_config.yaml
}


# Main loop for iterations
for i in {1..3}
do
    iteration_name="LLaMA3_iter${i}_XPO"
    jsonl_input="RLHFlow/ultrafeedback_iter${i}"
    json_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_XPO"
    model_output="${base_path}/${iteration_prefix}${i}_${iteration_name}_XPO_reward.json"

    # Determine the model path: first iteration uses the initial model, subsequent iterations use the previous iteration's model
    if [ $i -eq 1 ]; then
        model_path=$initial_model
        sam_model_path=$initial_model
    elif [ $i -eq 2 ]; then
        previous_iteration=$((i-1))
        model_path="LLaMA3_iter${previous_iteration}_XPO"
        sam_model_path=$initial_model
    else
        previous_iteration=$((i-1))
        sam_previous_iteration=$((i-2))
        model_path="LLaMA3_iter${previous_iteration}_XPO"
        sam_model_path="LLaMA3_iter${sam_previous_iteration}_XPO"
    fi



    if [ $i -eq 1 ]; then
        lr=5.0e-7
        alpha=1.0e-3
    elif [ $i -eq 2 ]; then
        lr=5.0e-7
        alpha=5.0e-4
    else
        lr=5.0e-7
        alpha=0.0e-6
    fi

    run_iteration $iteration_name $model_path $jsonl_input $json_output $model_output $alpha $lr $sam_model_path
done

