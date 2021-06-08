
dataset=tweet
sup_source=keywords
model=cnn

export CUDA_VISIBLE_DEVICES=0

python main.py --dataset ${dataset} --sup_source ${sup_source} --model ${model} --with_evaluation False
