import torch
import os
import cv2
import math
import random
import numbers
import numpy as np
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from   utils import CONFIG
import imutils

# for color transfer
from sklearn.cluster import KMeans


interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp
    


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, phase="test", real_world_aug = False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        self.phase = phase
        if real_world_aug:
            self.RWA = iaa.SomeOf((1, None), [
                iaa.LinearContrast((0.6, 1.4)),
                iaa.JpegCompression(compression=(0, 60)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
            ], random_order=True)
        else:
            self.RWA = None

    def __call__(self, sample):
        # convert GBR images to RGB
        image, alpha = sample['image'][:,:,::-1], sample['mask']
        
     
        if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
            image[image > 255] = 255
            image[image < 0]   = 0
            image = np.round(image).astype(np.uint8)
            image = np.expand_dims(image, axis=0)
            image = self.RWA(images=image)
            image = image[0, ...]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        
        # normalize image
        image /= 255.
        alpha /= 255.
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1

        sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(alpha)
        # ImageNet Normalization
        # sample['image'] = sample['image'].sub_(self.mean).div_(self.std)
        


        if 'occ' in sample:
            image, alpha = sample['occ'][:,:,::-1], sample['occ_mask']
            bg, skin = sample['bg'][:,:,::-1], sample['skin']
            if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
                image[image > 255] = 255
                image[image < 0]   = 0
                image = np.round(image).astype(np.uint8)
                image = np.expand_dims(image, axis=0)
                image = self.RWA(images=image)
                image = image[0, ...]
                bg[bg > 255] = 255
                bg[bg < 0]   = 0
                bg = np.round(bg).astype(np.uint8)
                bg = np.expand_dims(bg, axis=0)
                bg = self.RWA(images=bg)
                bg = bg[0, ...]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1)).astype(np.float32)
            alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
            bg = bg.transpose((2, 0, 1)).astype(np.float32)
            skin = np.expand_dims(skin.astype(np.float32), axis=0)
            
            
            # normalize image
            image /= 255.
            alpha /= 255.
            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1
            bg /= 255.
            skin /= 255.
            skin[skin < 0] = 0
            skin[skin > 1] = 1

            sample['occ'], sample['occ_mask'] = torch.from_numpy(image), torch.from_numpy(alpha)
            sample['bg'], sample['skin'] = torch.from_numpy(bg), torch.from_numpy(skin)

        if 'trimap' in sample:
            trimap = sample['trimap']
            trimap[trimap < 85] = 0     # black = 0
            trimap[trimap >= 170] = 2   # white = 255
            trimap[trimap >= 85] = 1    # unknown = gray
            sample['trimap'] = torch.from_numpy(trimap).to(torch.long)
            
            # make channel = 1
            sample['trimap'] = sample['trimap'][None,...].float()
            # make channel = 3
            # sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2,0,1).float()


        return sample
    


class RandomAffine(object):
    """
    Random affine translation
    """
    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['image'], sample['mask']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        sample['image'], sample['mask'] = fg, alpha


        # augment the occluders with affine
        if 'occ' in sample:
            # scale=[0.8, 1.2], shear=8, degree=15, flip=0.5, probability of transform: 0.7
            if np.random.rand() <= 1.0: # 0.7:
                occ, occ_mask = sample['occ'], sample['occ_mask']
                rows, cols, ch = occ.shape
                params = self.get_params((-15, 15), None, [0.8, 1.2], (-8, 8), 0.5, occ.size)
                
                center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
                M = self._get_inverse_affine_matrix(center, *params)
                M = np.array(M).reshape((2, 3))
                occ = cv2.warpAffine(occ, M, (cols, rows),
                                    flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
                occ_mask = cv2.warpAffine(occ_mask, M, (cols, rows),
                                    flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

                sample['occ'], sample['occ_mask'] = occ, occ_mask

        return sample


    @ staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix
    
class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['image'], sample['mask']
        # if alpha is all 0 skip
        if np.all(alpha==0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32)/255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand()*(1.1 - sat_bar)/5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat>1] = 2 - sat[sat>1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand()*(1.1 - val_bar)/5-(1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val>1] = 2 - val[val>1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['image'] = fg*255

        return sample

class Resize(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=( CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2

    def __call__(self, sample):
        fg, alpha = sample['image'], sample['mask']
        h, w = alpha.shape
        if h != self.output_size[0] or w != self.output_size[1]:
            fg, alpha = fg.astype(np.uint8), alpha.astype(np.uint8)
            fg = cv2.resize(fg, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha = cv2.resize(alpha, self.output_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            fg, alpha = fg.astype(np.float32), alpha.astype(np.float32)
            sample.update({'image': fg, 'mask': alpha})
        return sample
    
    
class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,100)]

    def __call__(self, sample):
        alpha = sample['mask']
        h, w = alpha.shape

        max_kernel_size = max(30, int((min(h,w) / 2048) * 30))

        ### generate trimap
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])


        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
        sample['trimap'] = trimap

        return sample


class Composite(object):
    """
    Resize the fg(occluders) & 
    Composite the bg(face) and fg(occluders)
    """ 
    def __init__(self, occlusion=0.5, ratio=[0.2, 0.6]):
        self.occlusion = occlusion
        self.ratio = ratio

    def __call__(self, sample):
        img, mask, occ, occ_mask = sample['image'], sample['mask'], sample['occ'], sample['occ_mask']
        sample['skin'] = mask
        sample['bg'] = img


        # the probability that occlusion occurs
        if np.random.random()<=self.occlusion:

            ##### random resize fg(occluder)

            # src_rect = cv2.boundingRect(img)
            # occ_rect = cv2.boundingRect(occ)
            src_rect = [0,0,img.shape[1], img.shape[0]] 
            occ_rect = [0,0,occ.shape[1], occ.shape[0]] 
            
            try:
                scale_factor = (((src_rect[2]*src_rect[3]))/(occ_rect[2]*occ_rect[3]) )*np.random.uniform(self.ratio[0], self.ratio[1])
                scale_factor=np.sqrt(scale_factor)
            except Exception as e:
                print(e)
                scale_factor=1
                
            occ, occ_mask = occ.astype(np.uint8), occ_mask.astype(np.uint8)
            h, w = occ_mask.shape
            new_size = tuple(np.round(np.array([w, h]) * scale_factor).astype(int))
            occ = cv2.resize(occ, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            occ_mask = cv2.resize(occ_mask, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            occ, occ_mask = occ.astype(np.float32), occ_mask.astype(np.float32)


            ##### Composit the image

            img[img>255] = 255
            img[img<0] = 0
            mask[mask>255] = 255
            mask[mask<0] = 0
            occ[occ>255] = 255
            occ[occ<0] = 0
            occ_mask[occ_mask>255] = 255
            occ_mask[occ_mask<0] = 0

            # Add rotate around center
            src_center=(src_rect[0]+(src_rect[2]/2),(src_rect[1]+src_rect[3]/2))
            occ_coord = np.random.uniform([src_rect[0],src_rect[1]], [src_rect[0]+src_rect[2],src_rect[1]+src_rect[3]])
            rotation= self.angle3pt((src_center[0],occ_coord[1]),src_center,occ_coord)
            if occ_coord[1]>src_center[1]:
                rotation=rotation+180
            occ  = imutils.rotate_bound(occ,rotation)
            occ_mask = imutils.rotate_bound(occ_mask,rotation)

            # make occluder mask's shape to image_mask's shape
            occlusion_mask=np.zeros(mask.shape, np.float32)
            occlusion_mask[(occlusion_mask>0) & (occlusion_mask<255)] = 255
            comp, final_mask, occ_mask, occ = self.paste_over(occ,occ_mask,img,mask,occ_coord,occlusion_mask, sample['randOcc'])
            final_mask[mask==0]=0


            sample.update({'image':comp, 'mask':final_mask, 'occ':occ, 'img_shape':final_mask.shape})
            occ_mask[mask > 1] = 0
            sample['occ_mask'] = occ_mask

            # ## imwrite --> to check
            # os.makedirs('/home/jhb/base/FaceExtraction/check', exist_ok = True)
            # cv2.imwrite('/home/jhb/base/FaceExtraction/check/comp.png', comp)
            # cv2.imwrite('/home/jhb/base/FaceExtraction/check/mask.png', final_mask)
            # cv2.imwrite('/home/jhb/base/FaceExtraction/check/occ.png', occ)
            # cv2.imwrite('/home/jhb/base/FaceExtraction/check/occ_mask.png', occ_mask)
            # cv2.imwrite('/home/jhb/base/FaceExtraction/check/skin_mask.png', sample['skin'])
        else:
            # no occlusion : occ_mask = black mask
            black = np.ones(mask.shape)*255
            occ = np.ones(img.shape)*255
            sample.update({'image':img, 'mask':mask, 'occ':occ, 'occ_mask':black})
            


        del sample['randOcc']

        return sample
    
    
    def angle3pt(self, a, b, c):
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
        
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang
    
    # https://github.com/isarandi/synthetic-occlusion
    def paste_over(self, im_src, occluder_mask, im_dst, dst_mask, center, occlusion_mask, randOcc):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visie).
        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src

        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        occluder_mask =occluder_mask[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        color_src = region_src[..., 0:3]


        # alpha = (region_src[..., 3:].astype(np.float32)/255)
        alpha = (occluder_mask.astype(np.float32)/255)
        if randOcc:
            if np.random.rand()<0.3:
                trans = np.random.uniform(0.4, 0.7)
                alpha *= trans
            else:
                trans = 1

        # kernel = np.ones((3,3),np.uint8)
        # alpha = cv2.erode(alpha,kernel,iterations = 1)
        # alpha = cv2.GaussianBlur(alpha,(5,5),0)
        alpha = np.expand_dims(alpha, axis=2)
        alpha = np.repeat(alpha, 3, axis=2)

        im_dst_cp = im_dst.copy()
        dst_mask_cp = dst_mask.copy()
        occ_mask_cp = occlusion_mask.copy()
        color = np.zeros(im_dst.shape, dtype=np.uint8)

        if randOcc:
            if np.random.rand()<0.3:
                occ_mask_cp = occ_mask_cp.astype(np.float32)
                occ_mask_cp *= trans
                occ_mask_cp = occ_mask_cp.astype(np.uint8)
                occluder_mask = occluder_mask.astype(np.float32)
                occluder_mask *= trans
                occluder_mask = occluder_mask.astype(np.float32)
        
        occ_mask_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.add(occlusion_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
        dst_mask_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.subtract(dst_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
        im_dst_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (alpha * color_src + (1 - alpha) * region_dst)
        color[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = color_src

        return im_dst_cp,dst_mask_cp,occ_mask_cp,color


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, sample):
        fg, alpha = sample['image'], sample['mask']
        if np.random.uniform(0, 1) < self.prob:
            fg = cv2.flip(fg, 1)
            alpha = cv2.flip(alpha, 1)
        sample['image'], sample['mask'] = fg, alpha

        return sample
    


class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["img_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample
        
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w

        padded_image = np.pad(sample['image'], ((0,pad_h), (0, pad_w), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((0,pad_h), (0, pad_w)), mode="reflect")

        sample['image'] = padded_image
        sample['mask'] = padded_mask

        return sample
    

class CelebAHQ_COM():
    ## 이미 occluder이 모두 적용되어있는 image, mask pair를 사용하는 경우
    def __init__(self, root_dir, folder_list, img_ext, mask_ext, phase='train'):
        self.root_dir = root_dir
        self.phase = phase
        
        if phase == 'train':
            self.transform = transforms.Compose([
                        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
                        RandomJitter(),
                        Resize(),
                        ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug) ,               
                    ])
        elif phase == 'test':
            self.transform = transforms.Compose([
                        OriginScale(),
                        ToTensor(),              
                    ])
        else:
            raise ValueError(" -- PHASE ERROR -- ")
        
        self.folder_list = folder_list
        self.img_list, self.mask_list = self.get_pair_list(self.root_dir, self.folder_list, img_ext, mask_ext, phase)
        self.img_num = len(self.img_list)


    def get_pair_list(self, root_dir, folder_list, img_ext, mask_ext, phase):

        img_list, mask_list = [], []
        for folder in folder_list:

            if folder == 'celeba':
                txt = '/home/jhb/dataset/source/face/CelebAMask-HQ-WO-'+phase+'.txt'
                img_path = '/home/jhb/dataset/source/face/CelebAMask-HQ/CelebA-HQ-img'
                mask_path = '/home/jhb/dataset/source/face/CelebAMask-HQ-masks_hair_corrected'

                with open(txt, 'r') as f:
                    for line in f:
                        file = line.strip()
                        name = file.split('.')[0]
                        img_list.append(os.path.join(img_path, name+'.jpg'))
                        mask_list.append(os.path.join(mask_path, name+'.png'))

            else:
                if folder != "sim2" and folder != 'am2k':
                    path = os.path.join(root_dir, folder)
                    img_path = os.path.join(path, 'img')
                    mask_path = os.path.join(path, 'mask')
                    for file in os.listdir(img_path):
                        name = file.split('.')[0]
                        img_list.append(os.path.join(img_path, name+img_ext))
                        mask_list.append(os.path.join(mask_path, name+mask_ext))

        return img_list, mask_list


    def __getitem__(self, idx):

        img = cv2.imread(self.img_list[idx % self.img_num])
        mask = cv2.imread(self.mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        image_name = os.path.split(self.img_list[idx % self.img_num])[-1]
        sample = {'image': img, 'mask': mask, 'image_name': image_name, 'img_shape':mask.shape}
        sample = self.transform(sample)

        return sample
    
    def __len__(self):
        return len(self.img_list)
    

class CelebAHQ_UNCOM():
    ## 이미 occluder이 모두 적용되어있는 image, mask pair를 사용하는 경우
    def __init__(self, fg_dir, bg_txt, folder_list, img_ext, mask_ext, rand_dir, occlusion=0.5, phase='train'):
        self.fg_dir = fg_dir
        self.bg_txt = bg_txt
        self.phase = phase
        
        if phase == 'train':
            self.transform = transforms.Compose([
                        # individual augmentation to fg(occluder), bg(face)
                        RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5), # for fg, bg
                        RandomJitter(), # for bg
                        Resize(), # for bg
                        GenMask(), # for loss, generate trimap
                        Composite(occlusion=occlusion, ratio=[0.2, 0.6]), # 0.2, 0.6   0.5, 1.0
                        ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug) ,               
                    ])
            
            self.transform_hiu = transforms.Compose([
                        # individual augmentation to fg(occluder), bg(face)
                        RandomAffine(degrees=30, scale=[0.5, 1.0], shear=10, flip=0.5), # for fg, bg
                        RandomJitter(), # for bg
                        Resize(), # for bg
                        GenMask(), # for loss, generate trimap
                        Composite(occlusion=occlusion, ratio=[0.2, 0.6]), # 0.2, 0.6   0.5, 1.0
                        ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug) ,               
                    ])
            
        elif phase == 'test': # 높은 확률로 train에서만 Uncom 사용. 
            self.transform = transforms.Compose([
                        OriginScale(),
                        ToTensor(),              
                    ])
        else:
            raise ValueError(" -- PHASE ERROR -- ")
        
        self.folder_list = folder_list
        init_img_list, init_mask_list = self.get_img_pair_list(self.bg_txt, img_ext, mask_ext, phase)
        init_occ_list, init_occ_mask_list, init_rand_list = self.get_occ_pair_list(self.fg_dir, self.folder_list, rand_dir, phase)
        
        # # sampling by occ
        # img_num = len(init_img_list)
        # self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list = [], [], [], [], []
        # for i in range(len(init_occ_list)):
        #     # sample the face imgs per occ
        #     img_index = np.random.choice(img_num, CONFIG.data.num_sample)
        #     for j in range( CONFIG.data.num_sample):
        #         self.img_list.append(init_img_list[img_index[j]])
        #         self.mask_list.append(init_mask_list[img_index[j]])
        #         self.occ_list.append(init_occ_list[i])
        #         self.occ_mask_list.append(init_occ_mask_list[i])
        #         self.rand_list.append(init_rand_list[i])

        
        # sampling by face
        occ_num = len(init_occ_list)
        self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list = [], [], [], [], []
        for i in range(len(init_img_list)):
            # sample the face imgs per occ
            img_index = np.random.choice(occ_num, CONFIG.data.num_sample)
            for j in range( CONFIG.data.num_sample):
                self.img_list.append(init_img_list[i])
                self.mask_list.append(init_mask_list[i])
                self.occ_list.append(init_occ_list[img_index[j]])
                self.occ_mask_list.append(init_occ_mask_list[img_index[j]])
                self.rand_list.append(init_rand_list[img_index[j]])
        
        self.img_num = len(self.img_list)


    def get_img_pair_list(self, bg_txt, img_ext, mask_ext, phase):

        img_list, mask_list = [], []
        img_path = '/home/jhb/dataset/source/face/CelebAMask-HQ/CelebA-HQ-img'
        # mask_path = '/home/jhb/dataset/source/face/CelebAMask-HQ-masks_corrected'  #  '.png'
        mask_path = '/home/jhb/dataset/source/face/CelebAMask-HQ-masks_hair_corrected1'  #  '.jpg'
        print(mask_path)

        with open(bg_txt, 'r') as f:
            for line in f:
                file = line.strip()
                name = file.split('.')[0]
                img_list.append(os.path.join(img_path, name+'.jpg'))
                mask_list.append(os.path.join(mask_path, name+'.jpg')) #  '.png'
                # hair_path.append(os.path.join(hair_path, name+'.jpg'))

        return img_list, mask_list
    

    def get_occ_pair_list(self, fg_dir, folder_list, rand_dir, phase):

        img_list, mask_list, rand_list = [], [], []
        for folder in folder_list:
            path = os.path.join(fg_dir, folder)
            if folder == 'rand':
                rdir = os.path.join(rand_dir + phase, 'rand')
                img_path = os.path.join(rdir, 'occlusion_img')
                mask_path = os.path.join(rdir, 'occlusion_mask')

                rand_sample = np.random.choice(os.listdir(img_path), 200, False)
                count = 0
                for file in rand_sample:
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.png'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(True)
                    count += 1
                print(folder, ':', count)
            elif folder == '11k':
                img_path = os.path.join(path, 'Hands')
                mask_path = os.path.join(path, '11k-hands_masks')
                for file in os.listdir(mask_path):
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.jpg'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(False)
                    count += 1
                print(folder, ':', count)
            elif "sim" in folder:
                path = os.path.join(path, phase)
                count = 0
                for fo in os.listdir(path):
                    path_folder = os.path.join(path, fo)
                    if not os.path.isdir(path_folder):
                        continue
                    if fo in CONFIG.data.sim_list1: 
                        print(fo, CONFIG.data.sim_list1)
                        continue
                    img_path = os.path.join(path_folder, 'fg')
                    mask_path = os.path.join(path_folder, 'alpha')
                    for file in os.listdir(img_path):
                        if file.endswith(('jpg', 'png')):
                            name = file.split('.')[0]
                            img_list.append(os.path.join(img_path, name+'.jpg'))
                            mask_list.append(os.path.join(mask_path, name+'.jpg'))
                            rand_list.append(True if fo in CONFIG.data.sim_list2 else False)
                            count += 1
                print(folder, ':', count)

            elif folder == 'hiu':
                txt_file = os.path.join(path, 'hiu_'+phase+'.txt')
                file_list = []
                with open(txt_file, 'r') as file:
                    for line in file.readlines():
                        file_list.append(line.strip())
                img_path = os.path.join(path, 'fg')
                mask_path = os.path.join(path, 'alpha')

                count = 0
                # randomly select 200 for train, 100 for test like 11k
                train_num = np.random.choice(len(file_list), 200, False)
                for i in train_num: # range(len(file_list)): # train_num:
                    img_list.append(os.path.join(img_path, file_list[i]+'.jpg'))
                    mask_list.append(os.path.join(mask_path, file_list[i]+'_mask.png'))
                    rand_list.append(False)
                    count += 1
                print(folder, ':', count)

            elif folder == 'am2k':
                img_path = os.path.join(path, 'train/fg')
                mask_path = os.path.join(path, 'train/mask')

                rand_sample = np.random.choice(os.listdir(img_path), 1000, False)
                count = 0
                for file in rand_sample:
                    if file.endswith('.png'):
                        img_list.append(os.path.join(img_path, file))
                        mask_list.append(os.path.join(mask_path, file))
                        rand_list.append(False)
                        count +=1 
                print("AM2K : ", count)

            else:
                raise ValueError(" -- Folder ERROR --  : ", folder)


        return img_list, mask_list, rand_list


    def __getitem__(self, idx):
        # self.img_list, self.mask_list self.occ_list, self.occ_mask_list 
        img = cv2.imread(self.img_list[idx % self.img_num])
        mask = cv2.imread(self.mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        occ = cv2.imread(self.occ_list[idx % self.img_num])
        occ_mask = cv2.imread(self.occ_mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        
        occluder_rect = cv2.boundingRect(occ_mask)
        crop_occ_mask = occ_mask[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
        crop_occ = occ[ occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 

        # color transfer
        if 'hiu' in self.occ_list[idx % self.img_num] :
            # crop_occ = self.color_transfer(img, mask, occ, occ_mask)
            crop_occ = self.color_transfer(img, mask, crop_occ, crop_occ_mask)
        
        # composite 2 occluders
        crop_occ, crop_occ_mask = self._composite_occ(crop_occ, crop_occ_mask, idx)

        image_name = os.path.split(self.img_list[idx % self.img_num])[-1]
        occ_name = os.path.split(self.occ_list[idx % self.img_num])[-1]
        sample = {'image': img, 'mask': mask, 'occ': crop_occ, 'occ_mask': crop_occ_mask, 'randOcc':self.rand_list[idx % self.img_num], 'image_name': image_name, 'occ_name':occ_name, 'img_shape':mask.shape}
        
        if crop_occ.shape[0] == 0 or crop_occ.shape[1] == 0: 
            print('error', self.occ_list[idx % self.img_num])
            import sys
            sys.exit()
        if 'hiu' in self.occ_list[idx % self.img_num] and self.phase == 'train':
            sample = self.transform_hiu(sample)
        else:
            sample = self.transform(sample)
        

        return sample

    def skin(self, color):
        temp = np.uint8([[color]])
        color = cv2.cvtColor(temp,cv2.COLOR_RGB2HSV)
        color=color[0][0]
        e8 = (color[0]<=25) and (color[0]>=0)
        e9 = (color[1]<174) and (color[1]>58)
        e10 = (color[2]<=255) and (color[2]>=50)
        return (e8 and e9 and e10)
        
    def doDiff(self, img,want_color1,skin_color,size):
        diff01=want_color1[0]/skin_color[0]
        diff02=(255-want_color1[0])/(255-skin_color[0])
        diff03=(255*(want_color1[0]-skin_color[0]))/(255-skin_color[0])
        diff11=want_color1[1]/skin_color[1]
        diff12=(255-want_color1[1])/(255-skin_color[1])
        diff13=(255*(want_color1[1]-skin_color[1]))/(255-skin_color[1])
        diff21=want_color1[2]/skin_color[2]
        diff22=(255-want_color1[2])/(255-skin_color[2])
        diff23=(255*(want_color1[2]-skin_color[2]))/(255-skin_color[2])
        diff1=[diff01,diff11,diff21]
        diff2=[diff02,diff12,diff22]
        diff3=[diff03,diff13,diff23]
        for  i in range(size[0]):
            for j in range(size[1]):
                self.doDiffHelp(img,i,j,skin_color,diff1,diff2,diff3)

    def doDiffHelp(self, img,i,j,skin_color,diff1,diff2,diff3):
        for k in range(3):
            if(img[i,j,k]<skin_color[k]):
                img[i,j,k]*=diff1[k]
            else:
                img[i,j,k]=(diff2[k]*img[i,j,k])+diff3[k]
            
    def color_transfer(self, img, img_mask, occ, occ_mask):

        # 1) get the skin color
        img_tmp = img.copy().astype(np.float32)
        img_tmp = cv2.bitwise_and(img_tmp, img_tmp, mask=img_mask)
        img_tmp = img_tmp.astype(np.uint8)
        img_tmp = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2RGB)
        img_tmp = img_tmp.reshape((img_tmp.shape[0] * img_tmp.shape[1], 3))
        clt = KMeans(n_clusters = 4)
        clt.fit(img_tmp)

        def centroid_histogram(clt):
            # Grab the number of different clusters and create a histogram
            # based on the number of pixels assigned to each cluster.
            numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
            (hist, _) = np.histogram(clt.labels_, bins = numLabels)

            # Normalize the histogram, such that it sums to one.
            hist = hist.astype("float")
            hist /= hist.sum()

            # Return the histogram.
            return hist

        def get_color(hist, centroids):
            # Obtain the color with maximum percentage of area covered.
            maxi=0
            COLOR=[0,0,0]
            # Loop over the percentage of each cluster and the color of
            # each cluster.
            for (percent, color) in zip(hist, centroids):
                if(percent>maxi):
                    if(self.skin(color)):
                        COLOR=color
            return COLOR

        # Obtain the color and convert it to HSV type
        hist = centroid_histogram(clt)
        skin_color = get_color(hist, clt.cluster_centers_)
        skin_color = np.uint8([[skin_color]])
        # skin_color = cv2.cvtColor(skin_color,cv2.COLOR_RGB2HSV)
        skin_color=skin_color[0][0]

        # 2) get the hand color
        occ_tmp = occ.copy()
        occ_tmp = occ_tmp.astype(np.float32)
        occ_tmp[occ_mask==0] = np.nan
        hand_color = [np.nanmean(occ_tmp[:,:,0]), np.nanmean(occ_tmp[:,:,1]), np.nanmean(occ_tmp[:,:,2])]

        # 3) color transfrom
        skin_color = np.uint8([[skin_color]])
        skin_color=skin_color[0][0]
        skin_color=np.int16(skin_color)

        # Change the color maintaining the texture.
        size = occ.shape
        occ1 = np.float32(cv2.cvtColor(occ,cv2.COLOR_BGR2RGB))
        self.doDiff(occ1,skin_color,hand_color,size) # (img1,want_color1,skin_color,size)
        img2 = np.uint8(occ1)
        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)

        # Get the two images ie. the skin and the background.
        occ_mask_inv = cv2.bitwise_not(occ_mask)
        imgLeft = cv2.bitwise_and(occ, occ, mask=occ_mask_inv)
        skinOver = cv2.bitwise_and(img2, img2, mask = occ_mask)
        new_occ = cv2.add(imgLeft,skinOver)

        return new_occ

    def _composite_occ(self, occ, mask, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.img_num) + idx
            fg2 = cv2.imread(self.occ_list[idx2 % self.img_num])
            alpha2 = cv2.imread(self.occ_mask_list[idx2 % self.img_num], 0) # .astype(np.float32)/255.
            # h, w = alpha.shape
            # fg2 = cv2.resize(fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            # alpha2 = cv2.resize(alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            occluder_rect = cv2.boundingRect(alpha2)
            crop_alpha2 = alpha2[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
            crop_fg2 = fg2[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 

        
            h, w = mask.shape
            crop_fg2 = cv2.resize(crop_fg2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            crop_alpha2 = cv2.resize(crop_alpha2, (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

            mask = mask.astype(np.float32)/255.
            crop_alpha2 = crop_alpha2.astype(np.float32)/255.

            # mgmatting
            # alpha_tmp = 1 - (1 - mask) * (1 - crop_alpha2)

            # wild matting
            alpha_tmp = crop_alpha2 * (1 - mask)
            if  np.any(alpha_tmp < 1):
                # mgmatting 
                # occ = occ.astype(np.float32) * mask[:,:,None] + crop_fg2.astype(np.float32) * (1 - mask[:,:,None])

                # wild matting
                # no modification in occ
                # occ = occ.astype(np.float32) * mask[:,:,None] + fg2.astype(np.float32) * (1 - mask[:,:,None])

                # The overlap of two 50% transparency should be 25%
                mask = alpha_tmp * 255
                mask = mask.astype(np.uint8)
                occ = occ.astype(np.uint8)

        # if np.random.rand() < 0.25:
        #     fg = cv2.resize(occ, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        #     alpha = cv2.resize(alpha, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

        return occ, mask
    
    def __len__(self):
        return len(self.img_list)



class Prefetcher():
    """
    Modified from the data_prefetcher in https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    """
    def __init__(self, loader):
        self.orig_loader = loader
        self.stream = torch.cuda.Stream()
        self.next_sample = None

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            for key, value in self.next_sample.items():
                if isinstance(value, torch.Tensor):
                    self.next_sample[key] = value.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        sample = self.next_sample
        if sample is not None:
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key].record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            # throw stop exception if there is no more data to perform as a default dataloader
            raise StopIteration("No samples in loader. example: `iterator = iter(Prefetcher(loader)); "
                                "data = next(iterator)`")
        return sample

    def __iter__(self):
        self.loader = iter(self.orig_loader)
        self.preload()
        return self


if __name__ == '__main__':
    import numpy as np
    from Dataset.utils import tensor2img


    def show_mask(I, mask):
        mask = tensor2img(mask) // 255
        I = I * (1 - mask) + (I * mask * 0.5) + (mask * np.array([0, 85, 255]) * 0.5)
        I = I.astype('uint8')
        return I


    fetcher = Prefetcher(batch_size=4, name='test')
    for idx in range(1000):
        I, mask = next(fetcher)
        I = torch.clamp(I, 0, 1)
        face = I * mask
        I = tensor2img(I)
        show_face = show_mask(I, mask)
        face = tensor2img(face)
        show = np.concatenate((I, show_face, face), axis=0)
        cv2.imshow('show', show[..., ::-1])
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
