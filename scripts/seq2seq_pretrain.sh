python ../parlai/scripts/train_model.py \
    --task reddit \
    --model seq2seq \
    --model-file ../../checkpoint/seq2seq/pretrain/model \
    --batchsize 192 \
    --validation-metric loss \
    --skip-generation True \
    --validation-every-n-epochs 1 \
    --optimizer adamax \
    --learningrate 1e-4 \
    --dict-tokenizer bpe \
    --dict-lower True \
    --rnn-class lstm \
    --embedding-type glove \
    --embeddingsize 300 \
    --hiddensize 512 \
    --numlayers 4 \
    --bidirectional True \
    --attention dot \
    --attention-time post \
    --lookuptable all \
    --text-truncate 360 \
    --label-truncate 72 \
    --person-tokens True \
    --num-epochs 10 \
    --validation-patience 1