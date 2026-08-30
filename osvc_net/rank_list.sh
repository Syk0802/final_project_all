MAT_FILE="./ranklist/market/market_01_oval_single/pytorch_result.mat"
OUT_DIR=$(dirname "${MAT_FILE}")

# python3 rank_list.py \
# --config_file='configs/softmax_triplet.yml' \
# --mat_file="${MAT_FILE}" \
# MODEL.DEVICE_ID "('1')" DATASETS.NAMES "('market1501')" \
# DATASETS.ROOT_DIR "('/xxx/extrawork/data/')" \
# OUTPUT_DIR "('${OUT_DIR}')" \
# TEST.WEIGHT "('/xxx/extrawork/code/svc_net/results/global/market_01_oval_single/resnet50_model_78.pth')"

python3 demo.py \
--mat_file="${MAT_FILE}" \
--output="${OUT_DIR}" \
--num_query 100
