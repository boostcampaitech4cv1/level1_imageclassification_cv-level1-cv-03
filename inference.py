import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_mask import TestDataset


def load_model(saved_model, num_classes, device, task_num):
    tasks = ['model_mask', 'model_gender', 'model_age']
    model_name = None
    if task_num == 0:
        model_name = args.model_mask
    elif task_num == 1:
        model_name = args.model_gender
    else:
        model_name = args.model_age

    model_cls = getattr(import_module(tasks[task_num]), model_name)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


# @torch.no_grad()
# def inference(data_dir, model_dir, output_dir, args):
#     """
#     """
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")

#     num_classes = MaskBaseDataset.num_classes  # 18
#     model = load_model(model_dir, num_classes, device).to(device)
#     model.eval()

#     img_root = os.path.join(data_dir, 'images')
#     info_path = os.path.join(data_dir, 'info.csv')
#     info = pd.read_csv(info_path)

#     img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
#     dataset = TestDataset(img_paths, args.resize)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=args.batch_size,
#         num_workers=multiprocessing.cpu_count() // 2,
#         shuffle=False,
#         pin_memory=use_cuda,
#         drop_last=False,
#     )

#     print("Calculating inference results..")
#     preds = []
#     with torch.no_grad():
#         for idx, images in enumerate(loader):
#             images = images.to(device)
#             pred = model(images)
#             pred = pred.argmax(dim=-1)
#             preds.extend(pred.cpu().numpy())

#     info['ans'] = preds
#     save_path = os.path.join(output_dir, f'output.csv')
#     info.to_csv(save_path, index=False)
#     print(f"Inference Done! Inference result saved at {save_path}")

@torch.no_grad()
def inference_mask(data_dir, model_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 3
    model = load_model(model_dir, num_classes, device, 0).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    
    return info

@torch.no_grad()
def inference_gender(data_dir, model_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 2
    model = load_model(model_dir, num_classes, device, 1).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    
    return info

@torch.no_grad()
def inference_age(data_dir, model_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = 3
    model = load_model(model_dir, num_classes, device, 2).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    
    return info

def merge_result(info_mask, info_gender, info_age, output_dir):
    tmp = []
    for i,j,k in zip(info_mask['ans'], info_gender['ans'], info_age['ans']):
        tmp.append(i*6 + j*3 + k)
    info_mask['ans'] = tmp

    save_path = os.path.join(output_dir, args.output_name+'.csv')
    info_mask.to_csv(save_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model_mask', type=str, default='ModelMask', help='model type (default: BaseModel)')
    parser.add_argument('--model_gender', type=str, default='ModelGender', help='model type (default: BaseModel)')
    parser.add_argument('--model_age', type=str, default='ModelAge', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_mask_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_mask'))
    parser.add_argument('--model_gender_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_gender'))
    parser.add_argument('--model_age_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp_age'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--output_name', type=str, default='output')

    args = parser.parse_args()

    data_dir = args.data_dir
    mask_model_dir = args.model_mask_dir
    gender_model_dir = args.model_gender_dir
    age_model_dir = args.model_age_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    info_mask = inference_mask(data_dir, mask_model_dir, args)
    info_gender = inference_gender(data_dir, gender_model_dir, args)
    info_age = inference_age(data_dir, age_model_dir, args)
    merge_result(info_mask, info_gender, info_age, output_dir)

