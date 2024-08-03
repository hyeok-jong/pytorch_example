
start_time=$SECONDS
model_name_s=('dinov2')
learning_rate_s=(0.001)
learning_method_s=('classifier')


for model_name in "${model_name_s[@]}"; do
    for learning_rate in "${learning_rate_s[@]}"; do
        for learning_method in "${learning_method_s[@]}"; do

            elapsed_time=$((SECONDS - start_time))
            hours=$((elapsed_time / 3600))
            minutes=$(( (elapsed_time % 3600) / 60))
            seconds=$((elapsed_time % 60))

            echo "Current Time: $current_time"
            echo "Elapsed Time: $hours hour(s) $minutes minute(s) $seconds second(s)"

            CUDA_VISIBLE_DEVICES=0,1,2,3 \
            python main.py \
            --model_name "${model_name}" \
            --learning_method "${learning_method}" \
            --lr "${learning_rate}" \
            --epochs 20 \
            --optimizer sgd \
            --weight_decay 0.01 \
            --scheduler CAWR \
            --batch_size 64 \
            --image_size 224 \
            --mixup yes \
            --master_port 12357
        done
    done
done



