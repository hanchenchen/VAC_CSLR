CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3444 main.py  \
--config configs/27-ctc-reg-decoder-26.yaml  \
--device 0,1 \
--batch-size 1 \
--work-dir work-dir/dist-gpu=2xbs=1-train/31-finetune-29/ \
--load-weights work-dir/dist-gpu=2xbs=1-train/29-SeqCTC-28/dev_0.30400_epoch35_model.pt \
# --phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
