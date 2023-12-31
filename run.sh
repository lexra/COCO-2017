#!/bin/bash -e

##############################
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

CONFIG_LIST=("yolo-grayscale" "yolo-person" "yolo-default" "yolo-fatest" "yolov3-tiny")
TARGET_CONFIG=$1

##########################################################
function Usage () {
        echo "Usage: $0 \${TARGET_CONFIG}"
        echo "CONFIG list: "
        for i in ${CONFIG_LIST[@]}; do echo "  - $i"; done
        exit 0
}
if ! `IFS=$'\n'; echo "${CONFIG_LIST[*]}" | grep -qx "${TARGET_CONFIG}"`; then
        Usage
fi
NAME=${TARGET_CONFIG}
CFG="cfg/${NAME}.cfg"
GPUS="-gpus 0"
WEIGHTS=""


##############################
if [ ! -e train2017.zip ]; then
	wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations_trainval2017.zip
	wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
	wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip
fi
if [ ! -e train2017/000000581929.jpg ]; then
	rm -rf val2017 train2017 annotations
	unzip -o train2017.zip
	unzip -o val2017.zip
	rm -rf _train2017 ; mkdir -p _train2017; for J in `ls train2017`; do python3 change24.py $(pwd)/train2017/$J $(pwd)/_train2017/$J ; mv -fv $(pwd)/_train2017/$J $(pwd)/train2017/$J ; done
	rm -rf _val2017 ; mkdir -p _val2017; for J in `ls val2017`; do python3 change24.py $(pwd)/val2017/$J $(pwd)/_val2017/$J ; mv -fv $(pwd)/_val2017/$J $(pwd)/val2017/$J ; done
fi
rm -rfv annotations
unzip -o annotations_trainval2017.zip

##############################
W=$(cat ${CFG} | grep width | awk -F '=' '{print $2}')
H=$(cat ${CFG} | grep height | awk -F '=' '{print $2}')

##############################
git clone https://github.com/tw-yshuang/coco2yolo.git || true
git clone https://github.com/immersive-limit/coco-manager.git || true
git clone https://github.com/alexmihalyk23/COCO2YOLO.git || true
export PYTHONPATH=`pwd`/coco2yolo:${PYTHONPATH}

echo ""
echo -e "${YELLOW} => Generate annotation txt files for train2017 ${NC}"
find train2017 -name "*.txt" | xargs rm -rf
if [ "${NAME}" == "yolo-person" ]; then
	python3 coco-manager/filter.py --input_json annotations/instances_train2017.json --output_json annotations/filter_train2017.json --categories 'person'
	mv -fv annotations/filter_train2017.json annotations/instances_train2017.json
	python3 coco-manager/filter.py --input_json annotations/instances_val2017.json --output_json annotations/filter_val2017.json --categories 'person'
	mv -fv annotations/filter_val2017.json annotations/instances_val2017.json
fi
python3 COCO2YOLO/COCO2YOLO.py -j annotations/instances_train2017.json -o train2017

echo -e "${YELLOW} => Generate annotation txt files for val2017 ${NC}"
find val2017 -name "*.txt" | xargs rm -rf
python3 COCO2YOLO/COCO2YOLO.py -j annotations/instances_val2017.json -o val2017

##############################
echo -e "${YELLOW} => Generate train.txt for all train2017 annotations ${NC}"
for J in $(ls train2017 | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/train2017/${J}.jpg" ; done | tee train.txt

##############################
echo -e "${YELLOW} => Generate valid.txt for all val2017 annotations ${NC}"
for J in $(ls val2017 | grep txt | awk -F '.txt' '{print $1}'); do echo "$(pwd)/val2017/${J}.jpg" ; done | tee valid.txt

##############################
sed "s|/work/himax/Yolo-Fastest/COCO-2017|`pwd`|" -i cfg/${NAME}.data

##############################
echo -e "${YELLOW} => Detector train ${NC}"
[ "$TERM" == "xterm" ] && GPUS="${GPUS} -dont_show"
[ -e ../data/labels/100_0.png ] && ln -sf ../data .
mkdir -p backup
[ -e backup/${NAME}_last.weights ] && WEIGHTS=backup/${NAME}_last.weights
../darknet detector train cfg/${NAME}.data ${CFG} ${WEIGHTS} ${GPUS} -mjpeg_port 8090 -map

##############################
if [ -e ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py ]; then
	echo -e "${YELLOW} => Convert to Keras ${NC}"
	git -C ../keras-YOLOv3-model-set checkout tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py
	sed "s|model_input_shape = \"160x160\"|model_input_shape = \"${W}x${H}\"|" -i ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py

	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/convert.py \
		--config_path cfg/${NAME}.cfg \
		--weights_path backup/${NAME}_final.weights \
		--output_path backup/${NAME}.h5

	echo -e "${YELLOW} => Convert to Tensorflow Lite ${NC}"
	#python3 ../keras-YOLOv3-model-set/tools/model_converter/keras_to_tensorflow.py --input_model backup/${NAME}.h5 --output_model backup/${NAME}.tflite
	python3 ../keras-YOLOv3-model-set/tools/model_converter/fastest_1.1_160/post_train_quant_convert_demo.py --keras_model_file backup/${NAME}.h5 --annotation_file train.txt --output_file backup/${NAME}.tflite || true

	echo -e "${YELLOW} => Generate backup/${NAME}-$(date +'%Y%m%d').cc file ${NC}"
	xxd -i backup/${NAME}.tflite > backup/${NAME}-$(date +'%Y%m%d').cc || true
	#ls -l backup/${NAME}-$(date +'%Y%m%d').cc
	#echo ""
	#cat backup/${NAME}-$(date +'%Y%m%d').cc | grep tflite_len

	#python3 ../keras-YOLOv3-model-set/eval_yolo_fastest_160_1ch_tflite.py \
	#       --model_path backup/${NAME}.tflite --anchors_path cfg/${NAME}.anchors --classes_path cfg/${NAME}.names --annotation_file train.txt --json_name ${NAME}.json || true
	#python3 ../pycooc_person.py \
	#       --res_path ../keras-YOLOv3-model-set/coco_results/${NAME}.json --instances_json_file annotations/instances_train2017.json || true
fi

##############################
echo ""
echo -e "${YELLOW} Detector Test: ${NC}"
echo -e "${YELLOW} ../darknet detector test cfg/${NAME}.data cfg/${NAME}.cfg backup/${NAME}_final.weights pixmaps/people.jpg ${NC}"
echo ""
exit 0
