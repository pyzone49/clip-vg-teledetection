python  train_clip_vg.py \
--epochs 10 --batch_size 64 --lr 0.00025  --lr_scheduler cosine \
--aug_crop --aug_scale --aug_translate      \
--imsize 224 --max_query_len 77 \
--dataset dior_rs  --data_root /Users/yacineflici/Documents/master-vmi/s2/TER \
--split_root /Users/yacineflici/Documents/master-vmi/s2/TER/unsup_multi_source_msa \
--device cpu \
--output_dir /Users/yacineflici/Documents/master-vmi/s2/TER/CLIP-VG/test/dior_rs ;