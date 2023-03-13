# Raw Video Player/Converter

This proyect is an implementation of a .RAW video player or converter (into a .mkv file).

## 1. Initial Configuration

1. Create (and activate) a new environment, named `raw-tool` with Python 3.9.

	- __Linux__ or __Mac__: 
	```
	conda create -n raw-tool python=3.9
	source activate raw-tool
	```
	- __Windows__: 
	```
	conda create --name raw-tool python=3.9
	activate raw-tool
	```

2. Clone current proyect repository and navigate to the downloaded folder.
```
git clone https://github.com/jorgealiaga36/raw-video-player.git
cd raw-video-player
```

3. Install required pip packages.
```
pip install -r requirements.txt
```

## 2. Usage

For running the code:

1. Make sure you are within the conda enviroment and the proyect directory previously cloned.
2. Run the following command:
```
~$ python raw_video_player.py --mode [mode-selected] --input-source [input-video-root] --cfg-source [conf-file-root] --output-source [-output-root]
```

Where:
* `--mode` or `-m`: Mode selected (play or convert input video).
* `--input-source` or `-ins`: Input video root.
* `--cfg-source` or `-cfgs`: Configuration file root.
* `--output-source` or `-outs`: Output video root.

### 2.1. Play instructions

Keys configurated for performing the following actions:

* Press `space-bar`: Pause video.
* Press `a`: Go a frame backwards.
* Press `d`: Go a frame upwards.
* Press `s`: Save frame.
* Press `esc`: End playing video.





