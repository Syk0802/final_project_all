# Experiment all tricks without center loss : 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on
# Dataset 1: msmt17
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss

gpu=$1
delta=$2
vc_type=${3:-l2}
python3 tools/train_global.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('${gpu}')" \
SOLVER.DELTA "(${delta})" \
SOLVER.VC_TYPE "('${vc_type}')" \
SOLVER.VC_IN_DIM "(2048)" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('msmt17')" \
DATASETS.ROOT_DIR "('/xxx/xxx/data')" \
OUTPUT_DIR "('./results/global_${vc_type}_${delta}/msmt17')"
