## COCO-2017

### YOLO Bounded Boxes Marking

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

Generating the train.txt with both XXX.txt / XXX.jpg co-existing. 

```bash
for J in $(ls train2017 | grep txt | awk -F '.' '{print $1}'); do \
    echo "$(pwd)/train2017/${J}.jpg" ; \
done | tee train.txt
```

#### Valid.txt

Generating the valid.txt with both XXX.txt / XXX.jpg co-existing. 

```bash
for J in $(ls val2017 | grep txt | awk -F '.' '{print $1}'); \
    do echo "$(pwd)/val2017/${J}.jpg" ; \
done | tee valid.txt
```
