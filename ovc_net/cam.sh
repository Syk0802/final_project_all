python3 cam.py \
--config_file='configs/softmax_triplet.yml' \
MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" \
DATASETS.ROOT_DIR "('/xxx/xxx/data/')" \
OUTPUT_DIR "('./vis/cam_vis/bipa')" \
MODEL.NAME "('resnet50_ibn_a')" \
TEST.WEIGHT "('./bipa_0.2_split2/market/resnet50_ibn_a_model_78.pth')"
# TEST.WEIGHT "('./bipa_0.0_split2/market/resnet50_ibn_a_model_78.pth')"
# TEST.WEIGHT "('./global_0.2/market/resnet50_ibn_a_model_78.pth')"
# TEST.WEIGHT "('./baseline/market/resnet50_ibn_a_model_78.pth')"
