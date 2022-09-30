CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 --master_port=3246 main.py  \
--config configs/13-bidire-CTC-11-08.yaml  \
--device 0 \
--batch-size 1 \
--work-dir work-dir/test/24-0929-debug-flip-13/ \
--load-weights work-dir/dist-gpu=2xbs=1-train/24.162-0929-debug-flip-13/dev_0.19900_epoch28_model.pt \
--phase test \
# --work-dir work-dir/test/22-backward_hook-21-dist-bs=2-gpu=1-02/
