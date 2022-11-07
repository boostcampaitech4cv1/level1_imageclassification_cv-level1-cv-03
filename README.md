# CV03 @AI Tech 4기 Image Classification Competition

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
