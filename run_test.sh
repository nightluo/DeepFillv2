python3 test.py \
--baseroot './test_data/' \
--baseroot_mask './test_data_mask/' \
--results_path './results' \
--gan_type 'WGAN' \
--gpu_ids '4,5' \
--multi_gpu True \
--epoch 20 \
--batch_size 1 \
--num_workers 8 \
--pad_type 'zero' \
--activation 'elu' \
--norm 'none' \
