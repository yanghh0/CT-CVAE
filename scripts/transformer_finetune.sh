python ../parlai/scripts/train_model.py \
    --task dailydialog,wizard_of_wikipedia,empathetic_dialogues,convai2 \
    --model transformer/generator \
    --init-model ../../checkpoint/transformer/pretrain/model \
    --model-file ../../checkpoint/transformer/joint-fine-tuning/model \
    --batchsize 16 \
    --validation-metric loss \
    --skip-generation True \
    --validation-every-n-epochs 0.25 \
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
    --relu-dropout 0.1 \
    --attention-dropout 0.1 \
    --text-truncate 360 \
    --label-truncate 72 \
    --person-tokens True \
    --validation-patience 10
