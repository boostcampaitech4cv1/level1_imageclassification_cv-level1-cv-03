# CV03 @AI Tech 4기 Image Classification Competition

## **Competition Overview**

**COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다**

이를 해결하고 추가적인 분류를 적용하여서 마스크 착용여부(o, x), 성별(male, female), 나이(30이하, 30~60, 60이상)등

18개의 class를 분류하는 모델을 설계하고자 한다
<br><img src="https://user-images.githubusercontent.com/68888169/200252310-6085da88-0a73-4b53-97d1-9a8e27a4eb70.png" width="640"/><br>

○ 개발환경
- 통합개발환경(IDE): VS Code
- 사용 서버: AI Stages 내 원격 V100 서버 (Linux OS / Python 3.8.13) SSH
- 사용 라이브러리: PyTorch 1.12.1 / TorchVision 0.13.1 / Jupyter Notebook 6.4.3
- 협업도구: Github, Slack, Zoom
- 실험관리도구: WanDB (Weight and Bias)

<br><img src="https://user-images.githubusercontent.com/97649344/200766469-01456d68-e1a0-434c-9afa-27e78d5e7dde.png" width="640"/><br>

<br><img src="https://user-images.githubusercontent.com/97649344/200766525-e016aead-501b-4de5-a2a3-04923442755d.png" width="640"/><br>

① 하나의 이미지를 마스크착용/연령/성별 모델에 학습 및 추론하여 각각의 라벨로 분리
- 각각의 모델을 적용하기에 앞서 마스크/성별/연령을 확인하는데 적합한 Data Augmentation
기법을 적용한 뒤 각 클래스별 모델에 데이터 적재

② 이후 각 결과 Label에 클래스별 가중치를 부여하여 멀티클래스 라벨(0~17)을 추출
- 결과에서 요구하는 멀티클래스 라벨은 마스크라벨 * 6 + 성별라벨 * 3 + 연령라벨 * 1
- 클래스별 라벨링은 각각의 클래스 구분 태스크를 분리하여 학습/추론하여 확인

③ 원본 이미지와 추출된 멀티클래스 라벨을 맵핑하여 CSV 파일로 출력

<br>
□ 프로젝트 팀 구성 및 역할
- 김동인_T4029(★팀장) : 마스크 착용 상태 판별 모델링, 데이터 시각화 <br>
- 박제원_T4092(팀원) : 마스크 착용 상태 판별 모델링, mask branch 관리 <br>
- 신현수_T4115(팀원) : 연령대 판별 모델링, age branch 관리 <br>
- 유승종_T4134(팀원) : 연령대 판별 모델링, Baseline Code 분석, 실험결과 관리 <br>
- 정승윤_T4198(팀원) : 성별 판별 모델링, gender branch 관리 <br>


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
