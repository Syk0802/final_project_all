# Experiment all tricks without center loss without re-ranking: 256x128-bs16x4-warmup10-erase0_5-labelsmooth_on-laststride1-bnneck_on (=raw all trick, softmax_triplet.yml)
# Dataset 1: market1501
# imagesize: 256x128
# batchsize: 16x4
# warmup_step 10
# random erase prob 0.5
# labelsmooth: on
# last stride 1
# bnneck on
# without center loss
# without re-ranking

gpu=$1
expname=$2
dataset=$3

python3 tools/test.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('${gpu}')" \
DATASETS.NAMES "('${dataset}')"  \
MODEL.NAME "('resnet50_ibn_a')" \
DATASETS.ROOT_DIR "('/xxx/xxx/data')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('./results/$expname/')" \
OUTPUT_DIR "('./results/$expname/test/$dataset')"
