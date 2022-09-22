CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=2341 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0 \
--batch-size 2 \
--work-dir work-dir/train/21-dist-bs=2-gpu=1-02/
