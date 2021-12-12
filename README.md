# Image-Enhancement-4th-place-solution (DACON, LG AI Research)
This repository is the 4th place solution for [Camera Image Quality Improvement AI Contest](https://dacon.io/competitions/official/235746/overview/description) (카메라 이미지 품질 향상 AI 경진대회).

## Overview
- 전반적인 학습 및 추론 프로세스는 다음 그림과 같습니다.
<img src="overview.png" width="400px" height="400px" title="Process overview"/><br>

## Requirements

- Ubuntu 18.04
- Python 3.8.11
- CUDA 11.0
- pip install git
- pip install -r requirements.txt

## Augmetation Sample

- augmentation_sample 폴더에 학습에 사용한 Augmentation에 대한 샘플 이미지가 있습니다.

## Preprocessing

- 10folds로 나누어 90%를 train 10%를 test로 사용하였습니다.
`--img_size` : patch crop 이미지의 크기
`--stride` : patch추출에 사용할 stride
`--resize` : 이미지 resize 크기
`--v_threshold` : HSV중 V(명도)에 해당하는 threshold
`--folds` : train, valid를 나눌 fold의 수
`--valid_fold` : valid set에 해당하는 fold
`--patch_growth_count` : 더 큰 패치를 자르기위해 img_size에 곱해지는 횟수
`--process_num` : multiprocess에 할당하는 process의 수
`--input_path` : 학습에 사용할 input image의 경로
`--target_path` : 학습에 사용할 target image의 경로
`--csv_path` : 학습에 사용할 이미지의 해당하는 csv파일의 경로
`--save_path` : 자른 이미지 저장경로

#### 0. data 저장

1) train : 대회에서 배포한 train 데이터와 csv파일을 data/train_original 폴더에 저장
2) test : 대회에서 배포한 test 데이터를 data 폴더에 저장

#### 1. 256size : 67065장

`python make_patch.py --img_size 256 --stride 128 --resize 256 --folds 10 --valid_fold 0`

#### 2. 352size : 57497장

`python make_patch.py --img_size 352 --stride 176 --resize 352 --folds 10 --valid_fold 0`


## Training

#### 3. 256size

`python main.py --img_size 256 --exp_num 1`

#### 4. 352size

`python main.py --img_size 352 --exp_num 2`


## testing

#### 5. Inference

- 위에서 학습한 결과는 256size : results/001/, 352size : results/002/ 에 각각 "{size}_best.pth" 모델로 저장됩니다. (처음부터 학습하여 inference를 진행하려면 각 "{size}_best.pth"를 weights 폴더로 이동해 주셔야합니다.)

`python test.py`