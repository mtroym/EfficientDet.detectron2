# EfficientDet.detectron2


# Train COCO.
```shell script
DETECTRON2_DATASETS=../../data/ python3 train.py --config-file configs/Base-EfficientDet-COCO.yaml --num-gpus 8```
```

# Install Detectron2.
```shell script
python3 -m pip install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html -i https://pypi.douban.com/simple
```
