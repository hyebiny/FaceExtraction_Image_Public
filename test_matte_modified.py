import segmentation_models_pytorch as smp
import os
import cv2
import argparse
import numpy as np

import torch
from torch import nn

import utils
import networks
import toml
from utils import CONFIG
from   torch.nn import SyncBatchNorm # for mgmatting

# from loss import OhemBCELoss, DiceLoss, IoU, Precision

# loss_bce = OhemBCELoss(thresh=0.7, n_min=256 ** 2-1).to(DEVICE)
# loss_dice = DiceLoss().to(DEVICE)

# def criterion(pred, gt):
#     bce = loss_bce(pred, gt)
#     # dice = loss_dice(pred, gt)
#     return bce
#     # return dice


# metrics = [IoU(threshold=0.0), Precision(threshold=0.0)]

from torchvision.utils import make_grid 

def single_inference(model, image_dict, post_process=False):

    with torch.no_grad():
        image = image_dict['image']
        alpha_shape = image_dict['alpha_shape']
        image = image.cuda()
        pred = model(image)
        h, w = alpha_shape

        # alpha_pred = make_grid(pred, nrow=image.shape[0], padding=0)
        # alpha_pred = alpha_pred.detach().permute(1, 2, 0).cpu().numpy() * 255
        if CONFIG.model.model == 'modnet':
            pred_semantic, pred_detail, pred = pred

        if CONFIG.model.model == 'aematter2':
            pred_semantic, pred = pred

        if CONFIG.model.model == 'mgmatting':
            pred = pred['alpha_os8']
        
        alpha_pred = pred[0, 0, ...].data.cpu().numpy() * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h+32, 32:w+32]
        # print(alpha_pred.shape)

        # alpha_pred[alpha_pred>1]=1
        # alpha_pred[alpha_pred<0]=0

        return alpha_pred


def generator_tensor_dict(image_path, args):
    # read images
    # image = cv2.resize(cv2.imread(image_path), (256, 256))
    image = cv2.imread(image_path)
    # mask = cv2.imread(mask_path, 0)
    # mask = (mask >= args.guidance_thres).astype(np.float32) ### only keep FG part of trimap
    # mask = mask.astype(np.float32) / 255.0 ### soft trimap
    # sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}
    # sample = {'image': image, 'alpha_shape': image.shape[:2]}
    sample = {'image_np': image.copy(), 'image': image, 'alpha_shape': image.shape[:2]}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        # padded_mask = np.pad(sample['mask'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        # sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        # padded_mask = np.pad(sample['mask'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        # sample['mask'] = padded_mask

    # # ImageNet mean & std
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    # convert GBR images to RGB
    # Opencv converts images to GRB, but PIL uses RGB
    # image, mask = sample['image'][:,:,::-1], sample['mask']
    image = sample['image'][:,:,::-1] 

    # image[image > 255] = 255
    # image[image < 0]   = 0

    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)
    # mask = np.expand_dims(mask.astype(np.float32), axis=0)

    # normalize image
    # Opencv has the range [0,255], PIL uses [0,1]
    image /= 255.

    # to tensor
    # sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
    sample['image'] = torch.from_numpy(image)
    # sample['image'] = sample['image'].sub_(mean).div_(std)

    # add first channel
    # sample['image'], sample['mask'] = sample['image'][None, ...], sample['mask'][None, ...]
    sample['image'] = sample['image'][None, ...]

    return sample

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./pretrained/epoch_16_best.ckpt', help="path of checkpoint")
    parser.add_argument('--image-dir', type=str, default='/home/jhb/base/test_dataset/10_1/id1_hanbam2_000418_9_0', help="input image dir")
    parser.add_argument('--image-ext', type=str, default='.png', help="input image ext")
    parser.add_argument('--num', type=int, default=5, help="the number of test images")
    parser.add_argument('--output', type=str, default='./output_resized/', help="output dir")


    # Parse configuration
    args = parser.parse_args()
    roots = args.checkpoint.split('/')
    root = ''
    for r in roots[2:-2]:
        root += r+'/'
    print(root)
    config = os.path.join('./experiments/'+root, 'train.toml')
    with open(config) as f:
        utils.load_config(toml.load(f))
    # with open(args.config) as f:
    #     utils.load_config(toml.load(f))

    # # Check if toml config file is loaded
    # if CONFIG.is_default:
    #     raise ValueError("No .toml config loaded.")

    os.makedirs(args.output, exist_ok=True)
    args.output = os.path.join(args.output, args.image_dir.split('/')[-1])
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # if not os.path.exists(os.path.join(output_dir, ))


    print(CONFIG.model.model)
    DEVICE = 'cuda' # :0,1'
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

    # load checkpoint
    print(args.checkpoint)
    checkpoint = torch.load(args.checkpoint)
    print(checkpoint.keys())
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    # model.load_state_dict(checkpoint, strict=True)

    # If we do not use dataparallel, modify the layers' names : remove 'module.' in the front
    # ckpt = {}
    # for i, (k,v) in enumerate(checkpoint.items()):
    #     ckpt.update({k[7:]:v}) 
    # model.load_state_dict(ckpt, strict=True)

    # inference
    # model = model.train()
    model = model.eval()


    all_folders = [item for item in os.listdir(args.image_dir) if not os.path.isfile(os.path.join(args.image_dir, item))]

    for folder in all_folders:

        if folder == '11k-sot':
            continue

        folder_dir = os.path.join(args.image_dir, folder)
        img_dir = os.path.join(folder_dir, 'img')

        output_folder = os.path.join(args.output, folder)
        os.makedirs(output_folder, exist_ok=True)

        count = 0
        for image_name in sorted(os.listdir(img_dir)):
            image_path = os.path.join(img_dir, image_name)        
            print('Image: ', image_path)

            # assume image and mask have the same file name
            # mask_path = os.path.join(args.mask_dir, image_name.replace(args.image_ext, args.mask_ext))
            # print('Image: ', image_path, ' Mask: ', mask_path)

            image_dict = generator_tensor_dict(image_path, args)
            alpha_pred = single_inference(model, image_dict)

            
            # import pdb; pdb.set_trace()
            alpha_pred = alpha_pred[:, :, None].repeat(3, axis=2)
            count += 1
            if args.num != -1:
                alpha_pred_binary = np.where(alpha_pred>127.5, 255, 0)
                row1 = np.concatenate([image_dict['image_np'], alpha_pred, alpha_pred_binary], axis=1)
                row2 = np.concatenate([image_dict['image_np']/2+alpha_pred/2, image_dict['image_np']/2+alpha_pred_binary/2, image_dict['image_np']*(alpha_pred_binary/255)], axis=1)
                grid = np.concatenate([row1, row2], axis=0).clip(0, 255)

                cv2.imwrite(os.path.join(args.output, folder+'_'+image_name), grid) #@#
                # cv2.imwrite(os.path.join(args.output, folder+'_'+image_name), alpha_pred*255) 
                if count == args.num:
                    break
            else:
                cv2.imwrite(os.path.join(output_folder, image_name), alpha_pred) #@#
