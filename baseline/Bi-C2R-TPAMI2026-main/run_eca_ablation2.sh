CONFIG_NAME=${1:-eca_noop}
CUDA_ID=${2:-1}
CUDA_VISIBLE_DEVICES=${CUDA_ID} python continual_train.py --config_file config/${CONFIG_NAME}.yml --logs-dir logs-${CONFIG_NAME}-setting2/ --setting 2
