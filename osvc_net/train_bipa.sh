# Experiment: 3-branch training (original + occluded + scaled) BiPA model + VCLoss
# Dataset: market1501
# imagesize: 384x128
# warmup_step 10
# random erase prob 1.1 (always on)
# erase shape: oval | rect | blob | random   (default: oval)
# erase fill : zero | patch | mean | noise | random  (default: zero)
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss

gpu=$1
delta=$2
vc_type=${3:-l2}
python3 tools/train_bipa.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('${gpu}')" \
SOLVER.DELTA "(${delta})" \
SOLVER.VC_TYPE "('${vc_type}')" \
SOLVER.VC_NUM "(3)" \
SOLVER.VC_IN_DIM "(4096)" \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/xxx/data')" \
INPUT.RE_TYPE "('oval')" \
INPUT.RE_PROB "(1.1)" \
INPUT.RE_SHAPE_MODE "('oval')" \
INPUT.RE_FILL_MODE "('zero')" \
INPUT.RE_MIN_COUNT "(1)" \
INPUT.RE_MAX_COUNT "(1)" \
OUTPUT_DIR "('./results/bipa_${vc_type}_${delta}/market')"
