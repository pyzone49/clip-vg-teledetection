python eval.py  \
--batch_size 128 --dataset dior_rs --imsize 224 \
--max_query_len 77 --data_root /Users/yacineflici/Documents/master-vmi/s2/TER  \
--split_root /Users/yacineflici/Documents/master-vmi/s2/TER/unsup_multi_source_msa \
--eval_model /Users/yacineflici/Documents/master-vmi/s2/TER/unsup_multi_source/dior_rs/best_checkpoint_08_01.pth \
--eval_set test --output_dir ./test \
--device cpu  ;