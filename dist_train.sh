CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3244 main.py  \
--config configs/11-tconv=C5-P2-C5-08.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/dist-gpu=2xbs=1-train/26-tf-dropout=0.3-11/ \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
