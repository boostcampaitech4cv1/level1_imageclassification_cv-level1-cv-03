# 🏆CV03 AI Tech 4기 Image Classification Competition🏆
![image](https://user-images.githubusercontent.com/108051004/201252519-073a692f-d14d-461b-a31e-d7236f4f0dd2.png)

## **Competition Overview**

**COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다**

이를 해결하고 추가적인 분류를 적용하여서 마스크 착용여부(o, x), 성별(male, female), 나이(30이하, 30~60, 60이상)등

18개의 class를 분류하는 모델을 설계하고자 한다
<br><img src="https://user-images.githubusercontent.com/68888169/200252310-6085da88-0a73-4b53-97d1-9a8e27a4eb70.png" width="640"/><br>


## 📒 프로젝트 개요
### ⚙️ 개발환경

- 통합개발환경(IDE): VS Code
- 사용 서버: AI Stages 내 원격 V100 서버 (Linux OS / Python 3.8.13) SSH
- 사용 라이브러리: PyTorch 1.12.1 / TorchVision 0.13.1 / Jupyter Notebook 6.4.3
- 협업도구: Github, Slack, Zoom
- 실험관리도구: WanDB (Weight and Bias)

### 📋 프로젝트 구조 및 사용 데이터셋의 구조도
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

## 🧑‍💻 프로젝트 팀 구성 및 역할

- 김동인_T4029(★팀장) : 마스크 착용 상태 판별 모델링, 데이터 시각화 
- 박제원_T4092(팀원) : 마스크 착용 상태 판별 모델링, mask branch 관리 
- 신현수_T4115(팀원) : 연령대 판별 모델링, age branch 관리 
- 유승종_T4134(팀원) : 연령대 판별 모델링, Baseline Code 분석, 실험결과 관리 
- 정승윤_T4198(팀원) : 성별 판별 모델링, gender branch 관리 



## 🗃️ 프로젝트 수행 절차 및 방법

### 1. **기획 단계**
   - 프로젝트를 수행함에 있어 분류해야하는 클래스는 크게 세 분류로 나누어지며 세부적으로 18개로 구분되므로, 한 개의 모델로 동시에 3개의 클래스를 분류해 내는 것에 어려움이 존재할 것으로 예상됨.
   - 이에 따라, 큰 분류인 마스크 착용 상태, 연령대, 성별로 모델을 분리하여 개발을 진행하기로 결정함.
   - 각 모델을 분리하기에 앞서 개발할 모델의 성능과 비교하기 위한 기준점이 되는 모델을 선정함으로써, 실험 결과와의 비교가 용이하도록 설정.
   - 기준점이 되는 Baseline 모델을 테스트하며 Data의 문제점을 분석하여, 적용하고자 하는 Augmentation 기법을 가늠
   - 실험 관리를 위한 도구로 TensorBoard와 WandB 중 직관적인 UI와 간편한 Metric 관리를 지원하는 WandB를 선택함
   - Git으로 버전관리를 진행하며 mask, age, gender로 Branch를 구분하여 각 태스크 간의 충돌이 최소화되도록 설정



### 2. **수행 단계**
   - 주어진 학습데이터를 바탕으로 탐색적 분석(EDA)를 진행하고, 데이터 레벨에서의 문제점을 식별
   - 클래스 별로 Baseline Model 기준 최적의 Data Augmentation 기법을 실험하여 도출
   - 1차적으로 수행한 Data Augmentation 기법을 바탕으로 조금 더 나은 성능의 Pretrained Model 탐색
   - 위 과정으로 확인한 최적 Pretrained Model에 더해 Data Augmentation 방법 및 Fine Tuning 수행
   - WandB 플랫폼과 연동하여 학습을 진행한 데이터를 저장하며, 학습 시 사용한 파라미터들도 함께 기록



### 3. **종합 및 검증 단계**
   - 클래스 별로 완성한 모델들을 사용하여 추론을 진행, Validation 데이터로 성능지표 확인
   - Baseline으로 생성한 라벨들과 각 클래스 별 최적 모델로부터 추론한 결과를 조합하며 AI Stages 결과 제출을 실행, 가장 높은 성능을 기록한 조합으로 최종 모델 결정



## 📊 **프로젝트 수행 결과**

### 1. 탐색적 분석(EDA) 및 데이터 전처리

   * EDA 수행 결과 아래와 같은 문제점을 식별할 수 있었음.

      *	잘못된 라벨링(성별 오기재, 18세 미만 또는 60세 이상은 모두 18세 또는 60세로 라벨링 됨)

      *	클래스 불균형(남여 비율이 약 3:7, 마스크 정상착용/미착용/오착용이 각각 5:1:1, 60대 데이터수가 타 클래스에 비해 절대적으로 	부족)

      | ![image](https://user-images.githubusercontent.com/108051004/201254732-94430f11-c82d-4ba5-8600-87ea242018e4.png) | ![image](https://user-images.githubusercontent.com/108051004/201254781-176eedf6-4984-4eaa-8128-a8982c2a8a96.png) |
      | ------------------------------------------------------------ | ------------------------------------------------------------ |
      | 나이 분포                                                    | 연령대 클래스 분포                                           |

      * 라벨링이 잘못된 데이터는 데이터셋에서 제외하여 학습을 진행함

      * 위 문제를 해결하기 위하여 데이터 수가 적은 클래스들에 대하여 Augmentation을 통한 Over-sampling으로 각 클래스들의 데이터 수를 맞춰주는 것에 초점을 맞춤



### 2. 모델 개요

   * Baseline으로 선택한 모델은 EfficientNet B0을 활용하였는데, 학습에 사용할 수 있는 데이터셋이 약 18,900개의 이미지로 제한적이기 때문에 Overfitting을 방지하기 위하여 지나치게 깊은 모델을 사용하는 것을 지양하였음

   * Optimizer는 초기에 SGD에 Learning Rate 0.01을 사용하였으나, 적절한 Step Size를 찾는데 지나치게 많은 시간이 소요된다는 단점으로 인하여 Adam Optimizer에 Learning Rate 0.0001을 사용하였음

   * 모델 구조는 크게 Mask, Age, Gender로 구분하여 각 모델에서 생성된 라벨을 종합하여 최종 Submission을 결과물로 취하는 구조를 사용하였음



### 3. 모델 선정 및 분석

   * **EfficientNet_B0 (Mask)** : 
   ResNet50, ResNet152의 경우 Overfitting이 발생하였으며, GoogLeNet과 VGG19는 무겁고 깊다보니 학습에 효과적이지 않다는 부분이 단점으로 지적되었음. 따라서 적당한 수렴속도와 빠른 학습속도를 통해 시간을 효율적으로 활용하기 위하여 EfficientNetB0를 활용하게 되었음

   

   * **ResNet152 (Age)** : 
   ResNet152, ResNet50, EfficientNetB0, VGG19 등으로 테스트를 해 본 결과 ResNet152가 가장 좋은 성능을 보였으며, 깊이가 깊은 모델인 만큼 이미지의 세세한 부분의 feature들까지 잡아내어 인물의 주름과 같은 얼굴의 특성들을 잘 학습한 것으로 사료됨

   

   * **EfficientNet_B3 (Gender)** : 
   ResNet이나 DenseNet을 활용하려 했으나, 성능적으로 EfficientNet이 더 우수하여 결정하게 되었으며, B0보다 좀 더 깊은 모델을 활용해 본 결과 B3에서 가장 좋은 결과를 얻을 수 있었음



### 4. 모델 평가 및 개선

   * 주어진 데이터셋 자체의 볼륨이 작고 클래스 간 불균형이 심했으며, 특히 연령대의 데이터 간 경계가 모호한 부분 때문에 Data Augmentation 기법에 집중하여 모델 학습을 진행함으로써 좋은 결과를 얻을 수 있었음

   

   * 또한 손수건 형태의 마스크(Bandana)가 포함된 학습데이터는 200건이 채 되지 않고 학습데이터와 차이가 큰 형상의 테스트데이터가 확인됨에 따라 별도 Augmentation을 적용하여 성능을 개선함



### 5. 시연 결과

   * 최종 모델 성능 

      * F1-Score 0.7693

      * Accuracy 81.3175%

   * 각 클래스 별 Valid F1 Score

      | ![Mask f1 score](https://user-images.githubusercontent.com/108051004/201269542-706a3505-ab04-4b5d-b4f5-dfae761902a1.png) | ![Age f1 score](https://user-images.githubusercontent.com/108051004/201269555-ed153447-cf63-4764-b34c-da11ef67b83f.png) | ![Gender f1 score](https://user-images.githubusercontent.com/108051004/201269558-0fb68bd7-d726-47fb-9e23-70b9c82a969d.png) |
      | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
      | Mask                                                         | Age                                                          | Gender                                                       |



## 🗨️ 자체 평가 의견

### 🙂 잘한 점들

* 작업 분할을 통한 업무 효율화를 실현하였음

* 학습결과 확인을 위한 시각화 도구를 만듦으로써, 모델을 테스트하는 중간 단계에서 문제를 빠르게 식별하고 대응할 수 있었음

* 익숙하진 않았지만 Git을 활용하여 버전을 관리하고 개별 Branch를 사용하여 충돌을 예방할 수 있었음

    

### 🤔 시도 했으나 잘 되지 않았던 것들

* 부족한 학습데이터로 인해 모델이 쉽게 Overfitting되는 현상을 예방하기 위한 Cross Validation 기법을 활용하고 싶었으나, 구현능력과 시간적 압박으로 해내지 못한 점이 아쉬움.

* RetinaFace 모델을 사용하여 마스크 착용 여부를 판단해보려 했으나 구현하지 못했음



### 😥 아쉬웠던 점들

* 프로세스를 큰 클래스별로 구분한 점은 성능 점수를 올리는 데에 적합하지만 서비스를 위한 경량화와는 반대의 경향성을 가지게 되어 실무에는 적합하지 않다는 점이 아쉬움

* TorchVision에서 지원하는 Pretrained 모델만 사용하여 학습을 진행함에 따라, 논문을 통해 다른 모델을 구현해보는 연습을 하지 못한 부분이 아쉬움

* cGAN 모델을 활용한 Face Aging 모델을 도입해서 마스크 미착용 이미지로 마스크 착용한 모습을 생성해보는 시도를 해보지 못한 점이 아쉬움



### 🤓 프로젝트를 통해 배운 점 또는 시사점

* 모델링의 중요성 외에도 정확한 데이터셋 분석과 이에 적합한 Data Augmentation 기법을 적용하는 것으로도 좋은 성능을 얻을 수 있다는 점을 알 수 있었음

* Overfitting이라는 문제가 모델을 학습시키는 과정에서 생각보다 많이 발생하며, 모델 성능에 치명적이라는 점을 알게 되었고, 이를 개선하기 위한 많은 노력과 통찰력이 필요하다는 것을 깨달았음

