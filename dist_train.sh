CUDA_VISIBLE_DEVICES=0,1,6,7 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=4444 main.py  \
--config configs/17-back-translation-dtw-16.yaml  \
--device 0,1,2,3 \
--batch-size 4 \
--work-dir work-dir/train/17-back-translation-dtw-16/
