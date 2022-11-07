# CV03 @AI Tech 4기 Image Classification Competition

## **Competition Overview**

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다

이를 해결하고 추가적인 분류를 적용하여서 마스크 착용여부(o, x), 성별(male, female), 나이(30이하, 30~60, 60이상)등

18개의 class를 분류하는 모델을 설계하고자 한다
![image](https://user-images.githubusercontent.com/68888169/200252310-6085da88-0a73-4b53-97d1-9a8e27a4eb70.png)



## Getting Started    
### Dependencies
- torch==1.12.1
- torchvision==0.13.1
- pandas~=1.2.0
- scikit-learn~=0.24.1
- matplotlib==3.5.1
- numpy~=1.21.5
- python-dotenv~=0.16.0
- Pillow~=7.2.0
- sklearn~=0.0
- timm==0.6.11
- wandb==0.13.4

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`
