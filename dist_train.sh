CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=3243 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0 \
--batch-size 2 \
--work-dir work-dir/dist-gpu=1-bs=1-train/06-bs=2-lr=1e-4-baseline+vac+smkd/ \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
