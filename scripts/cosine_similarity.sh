python parlai/scripts/eval_model.py \
    --datatype valid \
    --task twitter  \
    --model bert_ranker/bi_encoder_ranker \
    --bert-aggregation first \
    --batchsize 1
