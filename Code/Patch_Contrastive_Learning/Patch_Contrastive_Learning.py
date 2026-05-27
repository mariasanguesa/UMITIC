import torch
from torchvision import models
from Code.Patch_Contrastive_Learning.data_aug.contrastive_learning_dataset import ContrastiveLearningDataset_PCL
from Code.Patch_Contrastive_Learning.models.resnet_simclr import ResNetSimCLR_pretrained
from Code.Patch_Contrastive_Learning.simclr import SimCLR
import os

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def patch_contrastive_learning(args):
                
    # Create infer_loader
    dataset_PCL = ContrastiveLearningDataset_PCL(args, training=False)
    infer_loader = torch.utils.data.DataLoader(
        dataset_PCL, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False, drop_last=True)
    
    # Create train_loader
    dataset_PCL = ContrastiveLearningDataset_PCL(args,training=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_PCL, batch_size=args['CNN_Batch_Size'], shuffle=True,
        num_workers=args['N_Workers'], pin_memory=False, drop_last=True)

    # Create ResNet model
    model = ResNetSimCLR_pretrained(in_channels=dataset_PCL.in_channels, base_model=args['CNN_Architecture'], out_dim=args['CNN_Out_Dimensions'])

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), args['CNN_Learning_Rate'], weight_decay=args['CNN_Weight_Decay'])

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,last_epoch=-1)

    # Train patch contrastive learning 
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    if len([f for f in os.listdir(simclr.model_dir) if f.endswith('.tar')]) < 10:
        simclr.train(train_loader)

    # Infer model to get embeddings
    infer_dir = args['path']+'Patch_Contrastive_Learning/Image_Patch_Representation/'
    if not os.path.exists(infer_dir):
        os.mkdir(infer_dir)
    if len([f for f in os.listdir(args['path']+'Patch_Contrastive_Learning/Image_Patch_Representation/') if f.endswith('.npy')]) == 0:
        simclr.infer(infer_loader, infer_dir)

    # Clustering patches into phenotypes
    patch_dir = args['path']+'Patch_Contrastive_Learning/Clustering_Patches/'
    if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
    if len([f for f in os.listdir(patch_dir) if f.endswith('.png')]) == 0:        
        simclr.cluster_patches(infer_dir, patch_dir)
        