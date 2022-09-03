for mode in train val test; do
    if [ ! -d "data/retacred/$mode" ]; then
        mkdir -p results/retacred/$mode
    fi
done

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 src/run_prompt.py \
--data_dir data/retacred \
--output_dir results/retacred \
--model_type roberta \
--model_name_or_path roberta-large \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--max_seq_length 512 \
--warmup_steps 500 \
--learning_rate 3e-5 \
--learning_rate_for_new_token 1e-5 \
--num_train_epochs 5 \
--weight_decay 1e-2 \
--adam_epsilon 1e-6 \
--temps temp.txt
