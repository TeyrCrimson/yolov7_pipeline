# Helper Functions

## Dataset Functions
### Combining Datasets
Combines datasets of the same annotation format together

Relevant scripts:
- `combine_yolo_datasets.py`
- For combining COCO datasets, you may use [pyodi](https://gradiant.github.io/pyodi/reference/apps/coco-merge/). For example:
```bash
pyodi coco merge coco_1.json coco_2.json output.json
```

### Converting Datasets with Different Annotation Formats
Converts datasets between COCO and YOLO format

*Note*: YOLO annotation format consists of a `.txt` file for each image in the same directory. Each `.txt` file contains the annotations for the corresponding image file, with each line indicating 1 bounding box:
```<object-class> <xc> <yc> <width> <height>```
* `xc` and `yc`: The X and Y coordinates of the object's center point within the image, normalized to be between 0 and 1.

Each image must be of the same size, `640x640` or `1280x1280` are preferred, although rectangle images can be used too. In the case where:
- images are bigger than the specified size, `tiling.py` can be used.
- images are smaller than the specified size, `padding.py` can be used.


Relevant scripts (COCO to YOLO):
- `coco_to_yolo.py` / `run_coco_to_yolo.sh`
- `tiling.py`
- `split_train_val_test.py`

Relevant scripts(YOLO to COCO)
- `yolo_to_coco.py`

### Visualising YOLO Datasets
Visualise a YOLO dataset image and its annotations

Relevant script:
- `yolo_annotator.py`