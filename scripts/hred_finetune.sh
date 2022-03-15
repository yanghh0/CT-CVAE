python ../parlai/scripts/train_model.py \
    --task dailydialog,wizard_of_wikipedia,empathetic_dialogues,personachat,cornell_movie \
    --model hred \
    --init-model ../../checkpoint/hred/pretrain/model \
    --model-file ../../checkpoint/hred/joint-fine-tuning/model \
    --batchsize 32 \
    --validation-metric loss \
    --skip-generation True \
    --validation-every-n-epochs 0.2 \
    --optimizer adamax \
    --learningrate 1e-4 \
    --dict-tokenizer bpe \
    --dict-lower True \
    --embedding-type glove \
    --embeddingsize 300 \
    --hiddensize 512 \
    --numlayers 2 \
    --lookuptable all \
    --text-truncate 360 \
    --label-truncate 72 \
    --person-tokens True \
    --validation-patience 10