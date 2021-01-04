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
pip install numpy==1.17.5
pip install opencv-python # openCV install
pip install matplotlib
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html) # torch install
pip install Cython==0.29.1
pip3 install “git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI” # pycocotool == 2.0 install
pip install tensorboardX
pip install PyYAML
```
```
git clone https://github.com/Normal-case/Gaussian_YoloV3.git
```

## 2. Training COCO data
기본적인 학습 데이터는 coco 2017 dataset을 활용한다.   
```
mkdir COCO
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
mkdir weights
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
  IMGSIZE: 416
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
이제 coco 데이터가 아닌 다른 데이터로 학습을 진행하는 방법에 대해서 설명하고자 한다. 그러기 위해서는 먼저 학습을 위한 image 데이터와 그 이미지에서 detection 된 결과 값이 있어야 한다.   
데이터 준비
```
mkdir COCO
cd COCO

mkdir train2017 # train image 저장 공간
mkdir val2017 # valid image 저장 공간
mkdir label_t # train image bounding box text 저장 공간
mkdir label_v # valid image bounding box text 저장 공간
mkdir annotations # annotation 저장 공간
cd ..
```
train2017에는 train image를 val2017에는 valid image를 넣어주고 label_t, label_v에는 각각 train bounding box text, valid bounding box text파일을 준비한다.
그리고 annotations에는 json 파일을 준비해주는 데 json 파일은 image와 label을 준비한 뒤 아래 명령을 실행하면 만들어진다.
```
python create_json.py
```

bounding box text 형식은 처음부터 class, x_center, y_center, width, height 정보를 나타낸다.   

이 두가지가 마련되면 json 파일을 만들어 annotations에 저장한다. json의 형식은 coco 데이터 형식으로 아래와 같다.
```
dataset = {
			'categories' : { 'id' : class_id, 'name' : class_name, 'supercategory' : category_name},
			'images' : { 'filename' : image_name, 'id' : image_id, 'width':image_size, 'height':image_size},
			'annotations':{'area':width_b*height_b, 'bbox':[x1, y1, width_b, height_b], \
										 'category_id':class_id, 'id':annotation_id, 'image_id':image_id, \
										 'iscrowd':0, 'segmentation':[[x1, y1, x2, y1, x1, y2, x2, y2]]}
}
```
1. 먼저 categories의 경우 분류하고 싶은 class 정보를 저장한다. 고양이와 개를 구분하고 싶다면 고양이(class_name)는 0(class_id), 개(class_name)는 1(class_id)로 설정하는 것이다. supercategory에는 적당히 animal 정도로 채워넣으면 된다. 주의할 점은 따로 고양이와 개가 없는 사진을 학습하고 싶다면 background class를 0으로 남겨놓고 고양이는 1, 개는 2로 설정하는 것이 좋다.
2. images는 사진의 정보를 저장하는 것인데 filename은 이미지의 이름 id는 이미지 고유 번호를 하는 것이 좋다. coco 데이터의 경우 이미지 이름이 '000000000012.jpg' 형식으로 되어있고 굳이 id와 맞출 필요는 없다. width와 height의 경우 사진 해상도가 1920 x 1200 등의 값을 넣어주면 된다.
3. annotation은 bounding box의 정보를 저장하는 것인데 area는 사진 내에서 그 물체가 찾아지는 박스 크기를 나타낸다. 그리고 bbox는 박스의 x, y의 최소값과 박스의 너비, 높이를 넣어준다. category_id는 그 박스에서 검출된 class가 무엇인지 class_id 값을 넣어주는 것이고 id는 annotation의 고유한 id를 생성해주면 된다. 주의할 점은 annotation_id는 모든 annotation 마다 고유한 id값으로 중복되는 값이 있으면 안된다. 그리고 image_id는 annotation 박스가 어떤 이미지에서 검출된 것인지를 알기 위해서 저장되는 값이다. iscrowd는 한 박스 내에서 여러가지 물체가 검출되는지 여부이다(0은 single object, 1은 group object). 마지막으로 segmentation은 박스의 꼭지점을 저장하는 부분이다.   
json 만드는 파일은 [json](https://github.com/Normal-case/Gaussian_YoloV3/blob/master/create_json.py)를 참조   
그리고 학습을 위한 darknet53.conv.74를 다운받아 준다.
```
mkdir weights
cd weights

wget https://pjreddie.com/media/files/darknet53.conv.74
```

이렇게 custom data 준비가 끝났다면 그에 맞춰 코드를 변경해줘야 할 부분이 있다.   
**1. config 조정**
```
MODEL:
  TYPE: YOLOv3
  BACKBONE: darknet53
  ANCHORS: [[10, 13], [16, 30], [33, 23],
            [30, 61], [62, 45], [59, 119],
            [116, 90], [156, 198], [373, 326]]
  ANCH_MASK: [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
  N_CLASSES: 65 # custom class num
  GAUSSIAN: True
TRAIN:
  LR: 0.001
  MOMENTUM: 0.9
  DECAY: 0.0005
  BURN_IN: 1000
  MAXITER: 50000
  STEPS: (40000, 45000) # (maxiter * 0.8, maxiter * 0.9)
  BATCHSIZE: 2 
  SUBDIVISION: 2
  IMGSIZE: 416
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
  LRFLIP: False
  RANDOM_DISTORT: True
TEST:
  CONFTHRE: 0.8
  NMSTHRE: 0.45
  IMGSIZE: 416
NUM_GPUS: 1
```
가장 중요한 부분이 N_CLASSES 부분인데 이 부분을 custom class 수의 맞게 변경해준다. 
그리고 사실상 maxiter를 500,000번 돌린다는 건 엄청난 시간이 들기 때문에 시간을 잘 고려해서 조정해준다. 
그리고 batchsize의 경우 2로 했는데 GTX 1660 super 기준에서는 2를 넘어가면 out of memory 에러가 발생한다.   

**2. utils/utils.py 수정**   
utils.py에 get_coco_label_names() 함수를 보면 coco_label_names와 coco_class_ids 부분을 custom data에 맞게 수정해줌   

**3. cocodataset.py 수정**   
dataset/cocodataset.py를 보시면 이미지와 json의 경로 등을 지정해주기 때문에 이를 custom에 맞게 변경   
또한 line 85와 91의 '{:06f}'.format(id_) 부분은 custom 이미지 이름에 맞게 변경해주어야 한다.   

**4. cocoapi_evaluator.py 수정**   
utils/cocoapi_evaluator.py의 line 75에 보면 아래와 같다.
```
outputs = postprocess(outputs, 65, self.confthre, self.nmsthre)
```
여기서 65는 class 수를 의미하는데 이를 custom에 맞게 변경해준다.

training build
```
python train.py --cfg config/gaussian_yolov3_default.cfg --weights_path weights/darknet53.conv.74 --tfboard_dir ./log
```
마지막으로 custom train inference
```
mkdir demo
cd demo
mkdir image # image 저장
mkdir result # detection된 image 저장
cd ..
python inference.py
```
inference.py에 image 경로 및 checkpoint 경로 변경 후 inference 실행 결과는 ./demo/result에 저장된다.
