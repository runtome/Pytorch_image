import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets
import timm
import timm.optim
import timm.scheduler
import evaluate
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from lightning.fabric import Fabric
from utility import model_information
import os
import argparse




def main():
    parser = argparse.ArgumentParser(description="Your program description here")
    parser.add_argument("--model_name", default= "hf_hub:timm/tf_efficientnet_b7.ns_jft_in1k", type=str, help="Model name")
    parser.add_argument("--num_classes", default=8, type=int, help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--base_path", type=str, help="Base path for data")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for StratifiedKFold (default: 5)")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size (default: 8)")
    parser.add_argument("--image_path", type=str, default='train\\train', help="input image path example train/train")
    parser.add_argument("--img_size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--random", type=int, default=42, help="Random stage (default: 42)")
    parser.add_argument("--precision", type=bool, default=False, help="precision= 16-mixed")
    parser.add_argument("--num_accumulate", type=int, default=4, help="Num of small batch size")

    #Config paramitor 
    args = parser.parse_args()

    #Print model information
    model_information(args.model_name , args.num_classes, args.img_size,args.train_batch_size)

    criterion = nn.CrossEntropyLoss()
    # Cross Validation Configuration
    metric = evaluate.load("accuracy")
    torch.set_float32_matmul_precision('high')
    ##Savename 
    save_name = args.model_name.split('/')[-1]

    ##Pricision 
    if args.precision:
        precision = "16-mixed"
        fabric = Fabric(accelerator="cuda",precision=precision) # , precision="16-mixed"
    else:
        fabric = Fabric(accelerator="cuda")
    fabric.launch()



    #Transforms
    transforms = {
    "train": T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ]),
    "test": T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        T.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    ])
    }

    #Image path 
    base = os.getcwd()
    image_path = os.path.join(base,args.image_path)

    dataset = datasets.ImageFolder(root=image_path, # target folder of images
                                    transform=transforms["train"], # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    # Get the data and labels
    data = [item[0] for item in dataset.samples]
    labels = [item[1] for item in dataset.samples]

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.random)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    ## Train loop 

    all_eval_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels)):

        # Split data into train and validation sets
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        # Create data loaders for training and validation
        train_dataloader = torch.utils.data.DataLoader(train_set, 
                                                        batch_size=args.train_batch_size, 
                                                        shuffle=True,)
        val_dataloader = torch.utils.data.DataLoader(val_set, 
                                                        batch_size=args.eval_batch_size, 
                                                        shuffle=False,)

        print(f"Fold {fold+1} of {args.num_folds}")

        # Load Model
        model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes)


        # Load Optimizer and Scheduler
        optimizer = timm.optim.create_optimizer_v2(model, opt="AdamW", lr=5e-4)
        # optimizer = timm.optim.Lookahead(optimizer, alpha=0.5, k=6)    # update the slow weight every k steps
                                                                    # update the optimizer by combine slow weight and fast weight * alpha

        model, optimizer = fabric.setup(model, optimizer)

        scheduler = timm.scheduler.create_scheduler_v2(optimizer, num_epochs=args.num_epochs)[0]

        # Load Data: split train and valition set based on kfold
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

        # Reset Model Info
        info = {
            "metric_train": [],
            "metric_val": [],
            "train_loss": [],
            "val_loss": [],
            "best_metric_val": -999,
            "best_val_loss": 0,
        }

        for epoch in range(args.num_epochs):
            train_loss_epoch = []
            val_loss_epoch = []

            train_preds = []
            train_targets = []

            val_preds = []
            val_targets = []

            num_updates = epoch * len(train_dataloader)

            ### === Train Loop === ###
            model.train()
            for idx, batch in enumerate(tqdm(train_dataloader)):
                inputs, targets = batch
                # inputs = {k: v.to(device) for k,v in inputs.items()}
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                fabric.backward(loss)

                # === Gradient Accumulation === #
                if ((idx + 1) % args.num_accumulate == 0) or (idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    scheduler.step_update(num_updates=num_updates)
                    optimizer.zero_grad()
                # ============================= #

                train_loss_epoch.append(loss.item())
                train_preds += outputs.argmax(-1).detach().cpu().tolist()
                train_targets += targets.tolist()
            ### ==================== ###

            # optimizer.sync_lookahead()              # Sync slow weight and fast weight
            scheduler.step(epoch + 1)

            ### === Evaluation Loop === ###
            model.eval()
            with torch.no_grad():
                for batch in tqdm(val_dataloader):
                    inputs, targets = batch
                    # inputs = {k: v.to(device) for k,v in inputs.items()}
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Log Values
                    val_loss_epoch.append(loss.item())
                    val_preds += outputs.argmax(-1).detach().cpu().tolist()
                    val_targets += targets.tolist()
            ### ======================= ###

            # Log Data
            # metric_train = metric.compute(predictions=train_preds, references=train_targets, average='macro')['f1']
            # metric_val = metric.compute(predictions=val_preds, references=val_targets, average='macro')['f1']
            metric_train = metric.compute(predictions=train_preds, references=train_targets)['accuracy']
            metric_val = metric.compute(predictions=val_preds, references=val_targets)['accuracy']

            info["metric_train"].append(metric_train)
            info["metric_val"].append(metric_val)

            info["train_loss"].append(np.average(train_loss_epoch))
            info["val_loss"].append(np.average(val_loss_epoch))

            if metric_val > info["best_metric_val"]:
                print("New Best Score!")
                info["best_metric_val"] = metric_val
                info['best_val_loss'] = np.average(val_loss_epoch)
                torch.save(model, f"{save_name}_fold_no{fold}.pt")

            print(info)
            print(f"Fold: {fold} | Epoch: {epoch} | Metric: {metric_val} | Training Loss: {np.average(train_loss_epoch)} | Validation Loss: {np.average(val_loss_epoch)}")

        # save all best metric val
        all_eval_scores.append(info["best_metric_val"])

if __name__ == '__main__':
    main()


