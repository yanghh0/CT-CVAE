python ../parlai/scripts/train_model.py \
    --task dailydialog,wizard_of_wikipedia,empathetic_dialogues,convai2 \
    --model transformer/generator \
    --model-file ../../checkpoint/transformer/model \
    --batchsize 16 \
    --validation-metric loss \
    --skip-generation True \
    --validation-every-n-epochs 0.25 \
    --optimizer adamax \
    --variant xlm \
    --learn-positional-embeddings True \
    --learningrate 1e-4 \
    --dict-tokenizer bpe \
    --dict-lower True \
    --embedding-size 512 \
    --activation gelu \
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
    --fp16 true \
    --validation-patience 10