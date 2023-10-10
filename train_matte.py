import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from Dataset.dataset_hyebin import CelebAHQ_COM, CelebAHQ_UNCOM, Prefetcher
from loss import IoU, Precision, RegressionLoss, LapLoss, CompositionLoss
from utils.tb_util import TensorBoardLogger
from utils.train_utils import TrainEpoch, ValidEpoch
from utils.cosin_schedular import CosineAnnealingWarmUpRestarts

import os
import shutil
import datetime

import toml
import argparse 
import utils
from   utils import CONFIG
from   pprint import pprint

import networks
from   torch.nn import SyncBatchNorm # for mgmatting

import numpy as np
import random

torch.manual_seed(8282)
torch.cuda.manual_seed(8282)
torch.cuda.manual_seed_all(8282) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(8282)
random.seed(8282)

print('Torch Version: ', torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--config', type=str, default='/home/jhb/base/FaceExtraction_new/config/train.toml')
parser.add_argument('--local_rank', type=int, default=0)

# Parse configuration
args = parser.parse_args()
with open(args.config) as f:
    utils.load_config(toml.load(f))

# Check if toml config file is loaded
if CONFIG.is_default:
    raise ValueError("No .toml config loaded.")

CONFIG.phase = args.phase
if args.local_rank == 0:
    print('CONFIG: ')
    pprint(CONFIG)
CONFIG.local_rank = args.local_rank

##################################################################
###                    DataLoader Setting                      ###
##################################################################

# train_dataset = CelebAHQ_COM(CONFIG.data.train_root,
#                              CONFIG.data.folder_list,
#                              CONFIG.data.img_ext,
#                              CONFIG.data.mask_ext,
#                              'train')

valid_dataset = CelebAHQ_COM(CONFIG.data.test_root,
                             CONFIG.data.folder_list,
                             CONFIG.data.img_ext,
                             CONFIG.data.mask_ext,
                             'test')

occ_ratio = CONFIG.data.occlusion
train_dataset = CelebAHQ_UNCOM(CONFIG.data.fg_dir,
                             CONFIG.data.bg_txt,
                             CONFIG.data.folder_list,
                             CONFIG.data.img_ext,
                             CONFIG.data.mask_ext,
                             CONFIG.data.rand_dir,
                             occ_ratio,
                             'train')


print("DATASET SIZE")
print('train:', len(train_dataset), '=', CONFIG.data.num_sample, '*', len(train_dataset)/CONFIG.data.num_sample , 'valid: ', len(valid_dataset))
train_loader = DataLoader(train_dataset, batch_size=CONFIG.model.batch_size, shuffle=True, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)
train_loader = Prefetcher(train_loader)

epochs = CONFIG.train.total_epoch
model_root = './experiments/' 
if not os.path.exists(model_root):
    os.mkdir(model_root)

print(CONFIG.model.model)
DEVICE = 'cuda' # :0,1'
tb_log_dir = './tensorboard/'
if CONFIG.model.model == 'resnet18':
    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = 1
    ATTENTION = None
    ACTIVATION = None

    model = smp.Unet(encoder_name=ENCODER,
                    encoder_weights=ENCODER_WEIGHTS,
                    decoder_attention_type=ATTENTION,
                    classes=CLASSES,
                    activation=ACTIVATION)
else:
    model = networks.get_generator(name=CONFIG.model.model)
    if CONFIG.dist:
        model = SyncBatchNorm.convert_sync_batchnorm(model)


model = nn.DataParallel(model.to(DEVICE)) # , device_ids=[0,1])
# state_dict = torch.load('pretrained/epoch_26_best.ckpt')
# model.load_state_dict(state_dict)

# optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.train.G_lr, betas=(CONFIG.train.beta1, CONFIG.train.beta2))
optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(CONFIG.train.beta1, CONFIG.train.beta2))
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=20, T_mult=2, eta_max=CONFIG.train.G_lr, T_up=2, gamma=0.3)
check_lr = 0 # for constance lr

# To Use CompositionLoss, We have to set CelebA_UNCOM()
criterions = [RegressionLoss('l1').to(DEVICE), LapLoss('l1').to(DEVICE), CompositionLoss('l1').to(DEVICE)]
metrics = [RegressionLoss('l1'), IoU(threshold=0.0), Precision(threshold=0.0)]


min_score = 100
start_epoch = 1
exp_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
tb_logger = TensorBoardLogger(tb_log_dir=tb_log_dir, exp_string=exp_string, local_rank=CONFIG.local_rank )

save = os.path.join(model_root, exp_string)
os.makedirs(save, exist_ok=True)
shutil.copy(args.config, os.path.join(save, args.config.split('/')[0]))
ck_save = os.path.join(save, 'ckpts')
save = os.path.join(save, 'res')
os.makedirs(save, exist_ok=True)
os.makedirs(ck_save, exist_ok=True)


train_epoch = TrainEpoch(
    model=model,
    loss=criterions, # criterion,
    metrics=metrics,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    verbose=True,
    sv_pth=save
)

valid_epoch = ValidEpoch(
    model=model,
    loss=criterions, # criterion,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
    sv_pth=save
)

if not os.path.exists(save):
    os.mkdir(save)

print("Tensorboard Record : ", os.path.abspath(save))
print("Checkpoint Record : ", os.path.abspath(ck_save))


if CONFIG.train.resume_checkpoint:
    print('Resume checkpoint: {}'.format(CONFIG.train.resume_checkpoint))
    pth_path = CONFIG.train.resume_checkpoint
    checkpoint = torch.load(pth_path, map_location = lambda storage, loc: storage.cuda(DEVICE))

    if 'epoch' in checkpoint:
        #####
        ## If I save the model with epoch, lr, optimizer... etc
        #####
        start_epoch = checkpoint['iter']
        print('Loading the trained models from epoch {}...'.format(start_epoch))
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        if not CONFIG.train.reset_lr:
            if 'opt_state_dict' in checkpoint.keys():
                try:
                    optimizer.load_state_dict(checkpoint['opt_state_dict'])
                except ValueError as ve:
                    print("{}".format(ve))
            else:
                print('No Optimizer State Loaded!!')

            # if 'lr_state_dict' in checkpoint.keys():
            #     try:
            #         self.G_scheduler.load_state_dict(checkpoint['lr_state_dict'])
            #     except ValueError as ve:
            #         self.logger.error("{}".format(ve))
        # else:
        #     self.G_scheduler = lr_scheduler.CosineAnnealingLR(self.G_optimizer,
        #                                                         T_max=self.train_config.total_step - self.resume_step - 1)

        if 'loss' in checkpoint.keys():
            min_score = checkpoint['loss']

    else:
        #####
        ## When only the model is saved
        #####
        model.load_state_dict(checkpoint, strict=True)
        if not CONFIG.train.reset_lr: # if False
            start_epoch = 11 # Default setting
            


    # if self.model_config.arch.encoder == "vgg_encoder":
    #     utils.load_VGG_pretrain(self.G, self.model_config.imagenet_pretrain_path)
    #     utils.load_imagenet_pretrain(self.G, self.model_config.imagenet_pretrain_path)


loss_meter = [loss.__name__ for loss in criterions]
metric_meter = [loss.__name__ for loss in metrics]
for epoch in range(start_epoch, epochs+1):
    

    # if epoch % 20 == 0 and epoch != 0:
    
    #     valid_dataset = CelebAHQ_COM(CONFIG.data.test_root,
    #                             CONFIG.data.folder_list,
    #                             CONFIG.data.img_ext,
    #                             CONFIG.data.mask_ext,
    #                             'test')

    #     occ_ratio += 0.2
    #     if occ_ratio >= 1.0:
    #         occ_ratio = 1.0
    #     train_dataset = CelebAHQ_UNCOM(CONFIG.data.fg_dir,
    #                             CONFIG.data.bg_txt,
    #                             CONFIG.data.folder_list,
    #                             CONFIG.data.img_ext,
    #                             CONFIG.data.mask_ext,
    #                             CONFIG.data.rand_dir,
    #                             occ_ratio,
    #                             # CONFIG.data.occlusion,
    #                             'train')

    #     train_loader = DataLoader(train_dataset, batch_size=CONFIG.model.batch_size, shuffle=True, num_workers=4, drop_last=False)
    #     valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=4, drop_last=False)
    #     train_loader = Prefetcher(train_loader)


    print(f'\n Epoch: {epoch}/{epochs}')
    
    """====== Update Learning Rate ====="""
    # print(scheduler.get_lr()[0])
    scheduler.step()
    cur_lr = scheduler.get_lr()[0]

    if cur_lr <= 1e-5:
        check_lr = 1

    if check_lr == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5


    train_logs = train_epoch.run(train_loader, epoch)
    valid_logs = valid_epoch.run(valid_loader, epoch)

    # for loss in loss_meter:
    #     tb_logger.scalar_summary('Loss_'+loss, train_logs[loss], epoch)
    for loss in train_logs.keys():
        tb_logger.scalar_summary('Loss_'+loss, train_logs[loss], epoch)
    for loss in metric_meter:
        tb_logger.scalar_summary('Metric_'+loss, train_logs[loss], epoch)
    for loss in metric_meter:
        tb_logger.scalar_summary('Metric_'+loss, valid_logs[loss], epoch, phase='test')
    tb_logger.scalar_summary('LR', optimizer.param_groups[0]['lr'], epoch)

    if min_score > valid_logs['Regression']:
        print('best model')
        min_score = valid_logs['Regression']
        model_name = f'epoch_{epoch}_best.ckpt'
        # for name in os.listdir(model_root):
        #     if 'best' in name:
        #         model_name = f'epoch_{epoch}_best.ckpt'
        #         to_rename = os.path.join(ck_save, name)
        #         new_name = '_'.join(name.split('_')[:2])+'.ckpt'
        #         new_name = os.path.join(ck_save, new_name)
        #         os.rename(to_rename, new_name)

    else:
        model_name = f'epoch_{epoch}.ckpt'

    state_dict = model.state_dict()
    if 'best' in model_name or epoch % 5 == 0:
        model_pth = os.path.join(ck_save, model_name)
        torch.save(state_dict, model_pth)

    torch.save({
        'epoch': epoch,
        'loss': loss,
        'state_dict': state_dict,
        'opt_state_dict': optimizer.state_dict(),
        'lr_state_dict': scheduler.state_dict() 
        }, model_pth)
    
    print(f'Epoch: {epoch}, model saved')
    # for name in os.listdir(ck_save):
    #     if ('best' not in name) and (name != model_name):
    #         to_remove = os.path.join(ck_save, name)
    #         os.remove(to_remove)

    # if epoch == 20: # decrease the learning rate to 1e-5 from epoch 20
    #     optimizer.param_groups[0]['lr'] = 1e-5
    #     print('Decrease decoder learning rate to 1e-5')
