CUDA_VISIBLE_DEVICES=0,1  python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3562 main.py  \
--config configs/1002-02-conv-ctc-00.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/test-only/1011-29-LenPred-28/  \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
