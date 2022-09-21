CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=5362 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0,1 \
--batch-size 4 \
--work-dir work-dir/testcodes/16-dist-02-baseline+vac+smkd/
