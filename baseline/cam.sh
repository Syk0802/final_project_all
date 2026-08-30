python3 cam.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/home/xxx/data/')" \
OUTPUT_DIR "('./results/cam_vis/market')" \
TEST.WEIGHT "('/perception-hl/xxx/extrawork/reid/reid_base/results/baseline/market/resnet50_model_78.pth')"
# OUTPUT_DIR "('./results/all_tricks/market')"
