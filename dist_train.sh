CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=4253 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/train/20-dist-bs=2-02/
