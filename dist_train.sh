CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3561 main.py  \
--config configs/1002-02-conv-ctc-00.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/dist-gpu=2xbs=1-train/1005-16-wo-detach-15/ &
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/

