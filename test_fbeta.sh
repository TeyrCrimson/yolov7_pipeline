NAME='yolov7_test'
DATA_CONFIG='data/yolov7.yaml'
WEIGHTS='weights/yolov7.pt'
GROUND_TRUTH_JSON='gt.json'
OUTPUT_CSV='./results.csv'

python test_fbeta.py \
  --data $DATA_CONFIG \
  --img-size 1280 \
  --batch-size 32 \
  --conf-thres 0.001 \
  --iou-thres 0.65 \
  --task 'test' \
  --save-json \
  --gt-file $GROUND_TRUTH_JSON \
  --results-csv $OUTPUT_CSV \
  --device 0 \
  --weights $WEIGHTS \
  --name $NAME