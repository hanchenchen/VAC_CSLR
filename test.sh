python main.py  \
--config configs/baseline.yaml  \
--load-weights checkpoints/resnet18_baseline_dev_23.80_epoch25_model.pt \
--phase test \
--work-dir work-dir/only-test/resnet18_vac_smkd_dev_19.80_epoch35_model/  \
--device 3  \
--batch-size 2  \
--test-batch-size 2