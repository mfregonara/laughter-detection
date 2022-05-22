import torch

checkpoint = torch.load('./checkpoints/in_use/resnet_with_augmentation/best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
