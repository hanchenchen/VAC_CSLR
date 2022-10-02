CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3456 main.py  \
--config configs/1002-00-baseline-tf.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/dist-gpu=2xbs=1-train/1002-01-2-tf-00/ \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
