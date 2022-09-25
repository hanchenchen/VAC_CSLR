CUDA_VISIBLE_DEVICES=0,1,4,5 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=3241 main.py  \
--config configs/baseline+vac+smkd.yaml  \
--device 0,1,2,3 \
--batch-size 4 \
--work-dir work-dir/dist-gpu=4-bs=4-train/03-lr=2e-4-02-baseline+vac+smkd/ \
# --load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
