# Define
#export PYTHONPATH=/home/yingyan.li/repo/WoTE/
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset/maps"
export NAVSIM_EXP_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/exp/"
export NAVSIM_DEVKIT_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/"
export OPENSCENE_DATA_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset"
CONFIG_NAME=default 
python ./navsim/planning/script/run_training.py \
agent=WoTE_agent \
agent.config._target_=navsim.agents.WoTE.configs.${CONFIG_NAME}.WoTEConfig \
experiment_name=WoTE/${CONFIG_NAME} \
scene_filter=navtrain \
dataloader.params.batch_size=24 \
trainer.params.max_epochs=30 \
split=trainval