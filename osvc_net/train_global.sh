# Experiment: 3-branch training (original + occluded + scaled) Baseline + VCLoss
# Dataset: msmt17
# imagesize: 384x128
# warmup_step 10
# random erase prob 0.5
# erase shape: oval | rect | blob | random   (default: oval)
# erase fill : zero | patch | mean | noise | random  (default: zero)
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
SOLVER.VC_NUM "(3)" \
SOLVER.VC_IN_DIM "(2048)" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('msmt17')" \
DATASETS.ROOT_DIR "('/xxx/xxx/data')" \
INPUT.RE_TYPE "('oval')" \
INPUT.RE_PROB "(1.1)" \
INPUT.RE_SHAPE_MODE "('oval')" \
INPUT.RE_FILL_MODE "('zero')" \
INPUT.RE_MIN_COUNT "(1)" \
INPUT.RE_MAX_COUNT "(1)" \
OUTPUT_DIR "('./results/global_${vc_type}_${delta}/msmt17')"
