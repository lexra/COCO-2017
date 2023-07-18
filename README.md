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


```
../darknet detector test \
    cfg/yolo-person.data \
    cfg/yolo-person.cfg \
    backup/yolo-person_final.weights \
    pixmaps/three-companies.jpg \
    -dont_show -thresh 0.60
```

<img src=https://github.com/lexra/COCO-2017/assets/33512027/a5de4938-adf2-405c-8e30-0a5c65952e83 width=800 />


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

