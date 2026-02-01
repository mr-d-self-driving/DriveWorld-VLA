# Define
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset/maps"
export NAVSIM_EXP_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/exp/"
export NAVSIM_DEVKIT_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/"
export OPENSCENE_DATA_ROOT="/e2e-data/evad-tech-vla/liulin/WoTE/dataset"
export HYDRA_FULL_ERROR=1
export HF_HOME="/e2e-data/evad-tech-vla/liulin/WoTE/ckpt"
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
#rm -rf rm /root/.cache/huggingface/modules/transformers_modules/ReCogDrive-VLM-2B && \
#ln -s /e2e-data/evad-tech-vla/liulin/WoTE/ckpt/ReCogDrive-VLM-2B /root/.cache/huggingface/modules/transformers_modules/ReCogDrive-VLM-2B/ && \
CONFIG_NAME=default_test

# evaluation, change the checkpoint_path
python ./navsim/planning/script/run_pdm_score_gpu.py \
agent=ImagineWorld_agent \
agent.config._target_=navsim.agents.ImagineWorld.configs.${CONFIG_NAME}.ImagineWorldConfigTest \
experiment_name=eval/WoTE/${CONFIG_NAME}/ \
split=test \
scene_filter=navtest \
