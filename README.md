## COCO-2017

### YOLO Bounded Boxes Marking

<img src=https://github.com/lexra/COCO-2017/assets/33512027/92ae9f85-cd36-4c56-a167-adf0c2426b85 width=800 />


#### Marking TXT Files

```bash
ls -l train2017
...
-rw-rw-r-- 1 jasonc jasonc  315416 Aug 16  2014 000000581929.jpg
-rw-rw-r-- 1 jasonc jasonc      78 Jul 14 18:34 000000581929.txt
...
```

```bash
cat train2017/000000581929.txt
```

```bash
17 0.539766 0.492976 0.317406 0.345796
17 0.557742 0.265619 0.083922 0.087191
```

#### Annotations translation from a given JSON file

```python
python3 COCO2YOLO/COCO2YOLO.py -j annotations/instances_train2017.json -o train2017
```

### Train / Validation File

#### Train.txt

Generating the train.txt with both XXX.txt / XXX.jpg co-existing / matching. 

```bash
for J in $(ls train2017 | grep txt | awk -F '.' '{print $1}'); do \
    echo "$(pwd)/train2017/${J}.jpg" ; \
done | tee train.txt
```

#### Valid.txt

Generating the valid.txt with both XXX.txt / XXX.jpg co-existing / matching. 

```bash
for J in $(ls val2017 | grep txt | awk -F '.' '{print $1}'); \
    do echo $(pwd)/val2017/${J}.jpg ; \
done | tee valid.txt
```

### Dark Detector Test

#### three-companies.jpg

```
../darknet detector test \
    cfg/yolo-person.data \
    cfg/yolo-person.cfg \
    backup/yolo-person_final.weights \
    pixmaps/three-companies.jpg \
    -dont_show -thresh 0.60
```

<img src=https://github.com/lexra/COCO-2017/assets/33512027/a5de4938-adf2-405c-8e30-0a5c65952e83 width=800 />

#### people.jpg

```bash
../darknet detector test \
	   cfg/yolo-person.data \
    cfg/yolo-person.cfg backup/yolo-person_final.weights \
    pixmaps/people.jpg \
    -thresh 0.40 \
    -dont_show
```

<img src=https://github.com/lexra/COCO-2014/assets/33512027/ef1e20ee-4a6f-496c-9100-8785c1d6258e width=800 />


```bash
 CUDA-version: 11070 (12010), cuDNN: 8.9.2, GPU count: 1
 OpenCV version: 4.2.0
 0 : compute_capability = 610, cudnn_half = 0, GPU: NVIDIA GeForce MX250
net.optimized_memory = 0
mini_batch = 1, batch = 1, time_steps = 1, train = 0
   layer   filters  size/strd(dil)      input                output
   0 Create CUDA-stream - 0
 Create cudnn-handle 0
conv      8       3 x 3/ 2    160 x 160 x   1 ->   80 x  80 x   8 0.001 BF
   1 conv      8       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   2 conv      8/   8  3 x 3/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   3 conv      4       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   4 0.000 BF
   4 conv      8       1 x 1/ 1     80 x  80 x   4 ->   80 x  80 x   8 0.000 BF
   5 conv      8/   8  3 x 3/ 1     80 x  80 x   8 ->   80 x  80 x   8 0.001 BF
   6 conv      4       1 x 1/ 1     80 x  80 x   8 ->   80 x  80 x   4 0.000 BF
   7 dropout    p = 0.150        25600  ->   25600
   8 Shortcut Layer: 3,  wt = 0, wn = 0, outputs:  80 x  80 x   4 0.000 BF
   9 conv     24       1 x 1/ 1     80 x  80 x   4 ->   80 x  80 x  24 0.001 BF
  10 conv     24/  24  3 x 3/ 2     80 x  80 x  24 ->   40 x  40 x  24 0.001 BF
  11 conv      8       1 x 1/ 1     40 x  40 x  24 ->   40 x  40 x   8 0.001 BF
  12 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  13 conv     32/  32  3 x 3/ 1     40 x  40 x  32 ->   40 x  40 x  32 0.001 BF
  14 conv      8       1 x 1/ 1     40 x  40 x  32 ->   40 x  40 x   8 0.001 BF
  15 dropout    p = 0.150        12800  ->   12800
  16 Shortcut Layer: 11,  wt = 0, wn = 0, outputs:  40 x  40 x   8 0.000 BF
  17 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  18 conv     32/  32  3 x 3/ 1     40 x  40 x  32 ->   40 x  40 x  32 0.001 BF
  19 conv      8       1 x 1/ 1     40 x  40 x  32 ->   40 x  40 x   8 0.001 BF
  20 dropout    p = 0.150        12800  ->   12800
  21 Shortcut Layer: 16,  wt = 0, wn = 0, outputs:  40 x  40 x   8 0.000 BF
  22 conv     32       1 x 1/ 1     40 x  40 x   8 ->   40 x  40 x  32 0.001 BF
  23 conv     32/  32  3 x 3/ 2     40 x  40 x  32 ->   20 x  20 x  32 0.000 BF
  24 conv      8       1 x 1/ 1     20 x  20 x  32 ->   20 x  20 x   8 0.000 BF
  25 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  26 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  27 conv      8       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x   8 0.000 BF
  28 dropout    p = 0.150        3200  ->   3200
  29 Shortcut Layer: 24,  wt = 0, wn = 0, outputs:  20 x  20 x   8 0.000 BF
  30 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  31 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  32 conv      8       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x   8 0.000 BF
  33 dropout    p = 0.150        3200  ->   3200
  34 Shortcut Layer: 29,  wt = 0, wn = 0, outputs:  20 x  20 x   8 0.000 BF
  35 conv     48       1 x 1/ 1     20 x  20 x   8 ->   20 x  20 x  48 0.000 BF
  36 conv     48/  48  3 x 3/ 1     20 x  20 x  48 ->   20 x  20 x  48 0.000 BF
  37 conv     16       1 x 1/ 1     20 x  20 x  48 ->   20 x  20 x  16 0.001 BF
  38 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  39 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  40 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  41 dropout    p = 0.150        6400  ->   6400
  42 Shortcut Layer: 37,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  43 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  44 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  45 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  46 dropout    p = 0.150        6400  ->   6400
  47 Shortcut Layer: 42,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  48 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  49 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  50 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  51 dropout    p = 0.150        6400  ->   6400
  52 Shortcut Layer: 47,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  53 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  54 conv     96/  96  3 x 3/ 1     20 x  20 x  96 ->   20 x  20 x  96 0.001 BF
  55 conv     16       1 x 1/ 1     20 x  20 x  96 ->   20 x  20 x  16 0.001 BF
  56 dropout    p = 0.150        6400  ->   6400
  57 Shortcut Layer: 52,  wt = 0, wn = 0, outputs:  20 x  20 x  16 0.000 BF
  58 conv     96       1 x 1/ 1     20 x  20 x  16 ->   20 x  20 x  96 0.001 BF
  59 conv     96/  96  3 x 3/ 2     20 x  20 x  96 ->   10 x  10 x  96 0.000 BF
  60 conv     24       1 x 1/ 1     10 x  10 x  96 ->   10 x  10 x  24 0.000 BF
  61 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  62 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  63 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  64 dropout    p = 0.150        2400  ->   2400
  65 Shortcut Layer: 60,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  66 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  67 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  68 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  69 dropout    p = 0.150        2400  ->   2400
  70 Shortcut Layer: 65,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  71 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  72 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  73 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  74 dropout    p = 0.150        2400  ->   2400
  75 Shortcut Layer: 70,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  76 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  77 conv    136/ 136  3 x 3/ 1     10 x  10 x 136 ->   10 x  10 x 136 0.000 BF
  78 conv     24       1 x 1/ 1     10 x  10 x 136 ->   10 x  10 x  24 0.001 BF
  79 dropout    p = 0.150        2400  ->   2400
  80 Shortcut Layer: 75,  wt = 0, wn = 0, outputs:  10 x  10 x  24 0.000 BF
  81 conv    136       1 x 1/ 1     10 x  10 x  24 ->   10 x  10 x 136 0.001 BF
  82 conv    136/ 136  3 x 3/ 2     10 x  10 x 136 ->    5 x   5 x 136 0.000 BF
  83 conv     48       1 x 1/ 1      5 x   5 x 136 ->    5 x   5 x  48 0.000 BF
  84 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  85 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  86 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  87 dropout    p = 0.150        1200  ->   1200
  88 Shortcut Layer: 83,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  89 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  90 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  91 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  92 dropout    p = 0.150        1200  ->   1200
  93 Shortcut Layer: 88,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  94 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
  95 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
  96 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
  97 dropout    p = 0.150        1200  ->   1200
  98 Shortcut Layer: 93,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
  99 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
 100 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
 101 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
 102 dropout    p = 0.150        1200  ->   1200
 103 Shortcut Layer: 98,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
 104 conv    224       1 x 1/ 1      5 x   5 x  48 ->    5 x   5 x 224 0.001 BF
 105 conv    224/ 224  3 x 3/ 1      5 x   5 x 224 ->    5 x   5 x 224 0.000 BF
 106 conv     48       1 x 1/ 1      5 x   5 x 224 ->    5 x   5 x  48 0.001 BF
 107 dropout    p = 0.150        1200  ->   1200
 108 Shortcut Layer: 103,  wt = 0, wn = 0, outputs:   5 x   5 x  48 0.000 BF
 109 max                3x 3/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 110 route  108                                            ->    5 x   5 x  48
 111 max                5x 5/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 112 route  108                                            ->    5 x   5 x  48
 113 max                9x 9/ 1      5 x   5 x  48 ->    5 x   5 x  48 0.000 BF
 114 route  113 111 109 108                        ->    5 x   5 x 192
 115 conv     96       1 x 1/ 1      5 x   5 x 192 ->    5 x   5 x  96 0.001 BF
 116 conv     96/  96  5 x 5/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 117 conv     96       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 118 conv     96/  96  5 x 5/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 119 conv     96       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  96 0.000 BF
 120 conv     18       1 x 1/ 1      5 x   5 x  96 ->    5 x   5 x  18 0.000 BF
 121 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
nms_kind: greedynms (1), beta = 0.600000
 122 route  115                                            ->    5 x   5 x  96
 123 upsample                 2x     5 x   5 x  96 ->   10 x  10 x  96
 124 route  123 80                                 ->   10 x  10 x 120
 125 conv    120/ 120  5 x 5/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.001 BF
 126 conv    120       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.003 BF
 127 conv    120/ 120  5 x 5/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.001 BF
 128 conv    120       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x 120 0.003 BF
 129 conv     18       1 x 1/ 1     10 x  10 x 120 ->   10 x  10 x  18 0.000 BF
 130 yolo
[yolo] params: iou loss: ciou (4), iou_norm: 0.07, obj_norm: 1.00, cls_norm: 1.00, delta_norm: 1.00, scale_x_y: 1.00
nms_kind: greedynms (1), beta = 0.600000
Total BFLOPS 0.054
avg_outputs = 15199
 Allocate additional workspace_size = 0.04 MB
Loading weights from backup/yolo-person_last.weights...
 seen 64, trained: 3568 K-images (55 Kilo-batches_64)
Done! Loaded 131 layers from weights-file
 Detection layer: 121 - type = 28
 Detection layer: 130 - type = 28
pixmaps/people.jpg: Predicted in 439.069000 milli-seconds.
person: 56%
person: 99%
person: 77%
person: 28%
person: 65%
person: 98%
person: 98%
person: 57%
person: 43%
person: 100%
person: 89%
person: 40%
person: 81%
person: 80%
person: 40%
```

### Generate the .CC file

#### Python Packages Requirement

```bash
typing-extensions==3.7.4.3
python-dateutil==2.8.2
packaging==21.2
flatbuffers==1.12
requests==2.31.0
chardet==4.0.0
elastic-transport==8.0.0
google-auth==2.15.0
protobuf==3.19.6
urllib3==1.26.2
grpcio==1.32.0
testresources
numpy==1.19.5
setuptools
scipy
scikit-learn==0.20.3
opencv-python==4.2.0.32
opencv-contrib-python==4.2.0.32
tensorflow==2.4.4
keras_applications
tensorflow-model-optimization==0.5.0
tensorflow-addons
matplotlib
tqdm
pillow
mnn
Cython
pycocotools
keras2onnx
tf2onnx==0.4.2
onnx
onnxruntime
tfcoreml==1.1
sympy
imgaug
imagecorruptions
bokeh
tidecv
pyproject-toml
enum34
h5py
```

```bash
pip3 install -r requirements.txt
```

#### Convert to Keras

```bash
python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
    --config_path cfg/yolo-person.cfg \
    --weights_path backup/yolo-person.weights \
    --output_path backup/yolo-person.h5
```

#### Convert to Tensorflow Lite

```bash
python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py \
    --keras_model_file backup/yolo-person.h5 \
    --annotation_file train.txt \
    --output_file backup/yolo-person.tflite
```

Note please, we use the `trainvalno5k.txt` annotation file for `--annotation_file` input parameter. 

#### XXD

```
xxd -i backup/yolo-person.tflite > backup/yolo-person.cc
```

