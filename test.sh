python main.py  \
--config configs/08-add-clip-weight-00.yaml  \
--load-weights work-dir/train/09-12-layer-tf-08/dev_100.00_epoch17_model.pt \
--device 6,7 \
--phase test \
--work-dir work-dir/test/09-12-layer-tf-08_dev_100.00_epoch17_model/