python main.py  \
--config configs/baseline+vac+stmc.yaml  \
--load-weights checkpoints/resnet18_vac_smkd_dev_19.80_epoch35_model.pt \
--phase test \
--work-dir work-dir/test-codes/resnet18_vac_smkd_dev_19.80_epoch35_model/  \
--device 3  \
--batch-size 2  \
--test-batch-size 2