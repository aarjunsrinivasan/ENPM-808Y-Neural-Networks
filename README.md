# ENPM-808Y-Neural-Networks
# AFP

#### Advancing the Automated Fiber Placement

This document details steps to extract/transform/load labeled data, convert to protocol buffers, train the model and run detection procedures to classify defects in images. 

##### Defect Detection

Install requirements file.

`
pip install -r requirements.txt
`

######For installing and using VOTT refer the VOTT_Instructions.pdf

Steps to train/test model for defect prediction in layup process.

###### 1. Preprocess PASCAL VOC output for train/val split
 

_Preprocess VOTT output for train/val file structure and convert .bmp to .jpg_

`
python object-detection/algorithm/tools/vott_voc.py --data_dir data/vott_data/VOTT-Tool-PascalVOC-export/
`


###### 2. Convert datasets to TFRecords

-- launch tensorboard 

`
tensorboard --logdir logs
`
-- training dataset

`
python object-detection/algorithm/tools/voc2012.py --data_dir data/vott_data/VOTT-Tool-PascalVOC-export/ --split train --output_file data/vott_data/\
data_tfrecord/train.tfrecord --classes data/vott_data/class.names`

-- validation dataset
`
python object-detection/algorithm/tools/voc2012.py --data_dir data/vott_data/VOTT-Tool-PascalVOC-export/ --split val --output_file data/vott_data/\
/data_tfrecord/val.tfrecord --classes data/vott_data/class.names`

###### 3. Training the object detection model 

`
python object-detection/algorithm/train.py --dataset data/vott_data/data_tfrecord/train.tfrecord --val_dataset data/vott_data/data_tfrecord/val.tfrecord --classes data/vott_data/class.names --num_classes 4 --mode fit --transfer none --batch_size 8  --epochs 30 --size 416
`

###### 4. Testing the object detection model 

`
python object-detection/algorithm/detect.py --classes data/vott_data/class.names --num_classes 4 --weights checkpoints/yolov3_train_200.tf --image data/vott_data/vott-integration-PascalVOC-export/JPEGImages/Scaled_001A0.jpg --output output.jpg
`

-- detect from validation set

`
python object-detection/algorithm/detect.py --classes data/vott_data/class.names --num_classes 4 --weights checkpoints/yolov3_train_200.tf --tfrecord data/vott_data/data_tfrecord/val.tfrecord --size 416
`
	


