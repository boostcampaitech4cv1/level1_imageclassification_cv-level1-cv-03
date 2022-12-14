import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset_age import * #CustomDataset, train_transform_Over60_1
from loss import create_criterion

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import wandb
import pandas as pd


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size
    choices = random.sample(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    task = "age"
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]

        title = f"{task} - gt: {gt}, pred: {pred}"
        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset_age"), 'MaskBaseDataset')  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes

    df = pd.DataFrame({'img_path' : dataset.image_paths, 'label' :dataset.age_labels})
    train_df, val_df, _, _ = train_test_split(df, df['label'].values, test_size=args.val_ratio, random_state=args.seed, stratify=df['label'].values)
    train_df_label_0 = train_df[train_df['label']==0]   # ~30   : 7174
    train_df_label_1 = train_df[train_df['label']==1]   # 30~60 : 3646
    train_df_label_2 = train_df[train_df['label']==2]   # 60~   : 1075

    # -- augmentation
    transform_module = getattr(import_module("dataset_age"), 'train_transform_1')
    train_transform_1 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    transform_module = getattr(import_module("dataset_age"), 'val_transform')
    val_transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    ###################################### 내가 추가한 Augmentation들 ######################################
    transform_module = getattr(import_module("dataset_age"), 'train_transform_Over60_1')
    train_transform_over60_1 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    transform_module = getattr(import_module("dataset_age"), 'train_transform_Over60_2')
    train_transform_over60_2 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    transform_module = getattr(import_module("dataset_age"), 'train_transform_Over60_3')
    train_transform_over60_3 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    transform_module = getattr(import_module("dataset_age"), 'train_transform_Over60_4')
    train_transform_over60_4 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    transform_module = getattr(import_module("dataset_age"), 'train_transform_Over60_5')
    train_transform_over60_5 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    transform_module = getattr(import_module("dataset_age"), 'train_transform_30to60')
    train_transform_30to60 = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    ########################################################################################################


    train_img_paths_0, train_labels_0 = train_df_label_0['img_path'].values, train_df_label_0['label'].values
    train_img_paths_1, train_labels_1 = train_df_label_1['img_path'].values, train_df_label_1['label'].values
    train_img_paths_2, train_labels_2 = train_df_label_2['img_path'].values, train_df_label_2['label'].values
    train_dataset = []
    # 기본 이미지들을 데이터셋에다가 추가
    train_dataset.append(CustomDataset(train_img_paths_0, train_labels_0, train_transform_1))
    train_dataset.append(CustomDataset(train_img_paths_1, train_labels_1, train_transform_1))
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_1))

    ###################################### 내가 추가한 Augmentation들 ######################################
    # 30 이상 60 미만 데이터 2배로 증강 -> 7292장
    train_dataset.append(CustomDataset(train_img_paths_1, train_labels_1, train_transform_30to60))

    # 60 이상 데이터 6배로 증강 -> 6450장
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_over60_1))
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_over60_2))
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_over60_3))
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_over60_4))
    train_dataset.append(CustomDataset(train_img_paths_2, train_labels_2, train_transform_over60_5))
    ########################################################################################################
    train_set = ConcatDataset(train_dataset)

    val_img_paths, val_labels = val_df['img_path'].values, val_df['label'].values
    val_set = CustomDataset(val_img_paths, val_labels, val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model_age"), args.model)
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf

    for epoch in range(1, args.epochs+1):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
                wandb.log({
                    "Train loss":train_loss,
                    "Train acc":train_acc
                })

        scheduler.step()

        # val loop
        epoch_preds = []
        epoch_labels = []
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                epoch_preds += preds.detach().cpu().numpy().tolist()
                epoch_labels += labels.detach().cpu().numpy().tolist()  

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=True
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            val_f1 = f1_score(epoch_preds, epoch_labels, average="macro")

            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {val_f1:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            wandb.log({
                "Valid loss":val_loss,
                "Valid acc":val_acc,
                "Valid f1-score":val_f1,
                "results":figure
            })

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 1)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[256, 192], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='ResNet152', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate (default:1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='f1', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp_age', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    wandb.init(project  ="age_classification", entity="cv_3")
    wandb.config.update(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
