# Gaussian_YOLOv3_build

## 0. Gaussian_YOLOv3 서론
Gaussian YOLOv3의 가장 큰 특징은 YOLOv3 버전에서 bounding box의 loss를 계산하고 최적화 해준다는 점에서 실시간 자율주행에 더욱 경쟁력을 가진다.
정확도가 YOLOv3에 비해 훨씬 향상되었으며 학습 시간 또한 YOLOv3과 비슷하다.   
[motokimura/Pytorch_Gaussian_YOLOv3](https://github.com/motokimura/PyTorch_Gaussian_YOLOv3)의 코드를 참조해서 빌드했으며 pytorch를 활용해서 빌드하였다.
위 github은 linux 기준으로 빌드하지만 역자는 window 기준으로 빌드한다.

## 1. setting
- anaconda3 가상환경 활용
- visual studio 2019
- cuda 10.1 & cudnn 7.6.5
- requirements
```
pip install opencv-python # openCV install
pip install matplotlib
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html) # torch install
pip install Cython==0.29.1
pip3 install “git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI” # pycocotool == 2.0 install
pip install tensorboardX
pip install PyYAML
```
```
git clone https://github.com/motokimura/PyTorch_Gaussian_YOLOv3.git
```

## 2. Training COCO data
기본적인 학습 데이터는 coco 2017 dataset을 활용한다.   
```
# 먼저 'COCO' 폴더를 만들어준다.
cd COCO

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
```
wget, unzip과 같은 명령어를 윈도우에서 사용하려면 반드시 구글링을 통해서 다운받아 줘야한다.   
다음으로 weights 파일을 받아준다.
```
# 'weights' 폴더를 만든다.
cd weights

wget https://pjreddie.com/media/files/yolov3.weights
wget https://pjreddie.com/media/files/darknet53.conv.74
```
위의 yolov3.weights 파일은 학습을 통해서 만들어진 weights고 darknet53.conv.74는 학습을 해줄 weights 파일이다.   
   
이제 학습을 진행한다.
```
python train.py --help

usage: train.py [-h] [--cfg CFG] [--weights_path WEIGHTS_PATH] [--n_cpu N_CPU]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--eval_interval EVAL_INTERVAL] [--checkpoint CHECKPOINT]
                [--checkpoint_dir CHECKPOINT_DIR] [--use_cuda USE_CUDA]
                [--debug] [--tfboard_dir TFBOARD_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --cfg CFG             config file. see readme
  --weights_path WEIGHTS_PATH
                        darknet weights file
  --n_cpu N_CPU         number of workers
  --checkpoint_interval CHECKPOINT_INTERVAL
                        interval between saving checkpoints
  --eval_interval EVAL_INTERVAL
                        interval between evaluations
  --checkpoint CHECKPOINT
                        pytorch checkpoint file path
  --checkpoint_dir CHECKPOINT_DIR
                        directory where checkpoint files are saved
  --use_cuda USE_CUDA
  --debug               debug mode where only one image is trained
  --tfboard_dir TFBOARD_DIR     tensorboard path for logging
  ```
  위는 train.py를 어떻게 사용할지 간단히 확인하고 본인의 조건에 맞게 학습을 진행하면 된다.   
     
  **그리고 학습 전 주의사항이 있는데 오류가 나는 코드를 발견해서 고치고 진행하기를 권장하는 부분이 있다.
  models에 yolo_layer.py 187 line을 보면 obj_mask[b] = 1- pred_best_iou 이 있는데 이를**
  ```
  obj_mask[b] = ~(pred_best_iou)
  ```
  **처럼 바꿔야 한다. 여기서 pred_best_iou는 boolean type인데 '-'를 사용할 수 없다고 오류가 발생했기 때문에 not 명령어로 고쳐주었다.**
  ```
  python train.py --cfg config/gaussian_yolov3_default.cfg --weights_path weights/darknet53.conv.74 --tfboard_dir ./log
  ```
  학습은 iter 1000번 당 checkpoints 폴더에 값들을 저장한다. 그리고 gaussian_yolov3_default.cfg는 아래와 같다.
  ```
  MODEL:
  TYPE: YOLOv3
  BACKBONE: darknet53
  ANCHORS: [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]]
  ANCH_MASK: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  N_CLASSES: 80
  GAUSSIAN: True
TRAIN:
  LR: 0.001
  MOMENTUM: 0.9
  DECAY: 0.0005
  BURN_IN: 1000
  MAXITER: 500000
  STEPS: (400000, 450000)
  BATCHSIZE: 4
  SUBDIVISION: 16
  IMGSIZE: 608
  LOSSTYPE: l2
  IGNORETHRE: 0.7
  GRADIENT_CLIP: 2000.0
AUGMENTATION:
  RANDRESIZE: True
  JITTER: 0.3
  RANDOM_PLACING: True
  HUE: 0.1
  SATURATION: 1.5
  EXPOSURE: 1.5
  LRFLIP: True
  RANDOM_DISTORT: True
TEST:
  CONFTHRE: 0.8
  NMSTHRE: 0.45
  IMGSIZE: 416
NUM_GPUS: 1
```

## 3. Inference
이제 학습 결과를 확인을 위해서 500,000의 학습을 진행한 값을 motokimura drive에서 제공한다. [pretrained weights](https://drive.google.com/file/d/1zAFDSga9XLrsUBNHV3S2SvL1YWEsDB_p/view)   
그리고 demo.ipynb를 통해서 값을 확인해보자.

## 4. Custom Training
