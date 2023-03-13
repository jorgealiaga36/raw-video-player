# Real Time YOLOv5 Object Detection 

This repository contains an implementation of a YOLOv5 model to make inferences on a Youtube video.

## 1. Initial Configuration

1. Create (and activate) a new environment, named `yolo-dect` with Python 3.9.

	- __Linux__ or __Mac__: 
	```
	conda create -n yolo-dect python=3.9
	source activate yolo-dect
	```
	- __Windows__: 
	```
	conda create --name yolo-dect python=3.9
	activate yolo-dect
	```

2. Clone current proyect repository and navigate to the downloaded folder.
```
git clone https://github.com/jorgealiaga36/yolov5-detection.git
cd yolov5-detection
```

3. Clone official repository of the YOLOv5 model.
```
git clone https://github.com/ultralytics/yolov5.git
```

4. Install required pip packages.
```
pip install -r requirements.txt
```

## 2. Usage

For running the code:

1. Make sure you are within the conda enviroment and the proyect directory previously cloned.
2. Run the following command:
```
~$ python yolov5-detection.py --url [link-to-video] --out-source [output-root] --name [video-name]
```

Where:
* `--url` or `-u`: Link to video we want to detect.
* `--out-source` or `-outs`: Directory to store video downloaded.
* `--name` or `-n`: Downloaded video name.




