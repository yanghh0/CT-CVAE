python ../parlai/scripts/train_model.py \
    --task reddit \
    --model transformer/generator \
    --model-file ../../checkpoint/transformer/pretrain/model \
    --batchsize 128 \
    --validation-metric loss \
    --skip-generation True \
    --validation-every-n-epochs 1 \
    --optimizer adamax \
    --learningrate 1e-4 \
    --dict-tokenizer bpe \
    --dict-lower True \
    --embedding-type glove \
    --embedding-size 300 \
    --d-model 512 \
    --ffn-size 2048 \
    --n-heads 8 \
    --n-encoder-layers 6 \
    --n-decoder-layers 8 \
    --dropout 0.1 \
    --text-truncate 360 \
    --label-truncate 72 \
    --person-tokens True \
    --num-epochs 10 \
    --warmup_updates 600 \
    --validation-patience 1
