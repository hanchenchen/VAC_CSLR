CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=3242 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0 \
--batch-size 1 \
--work-dir work-dir/dist-gpu=1-bs=1-train/05-bs=1-lr=1e-4-baseline+vac+smkd/ \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
