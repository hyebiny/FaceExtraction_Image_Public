import os
import cv2
import numpy as np
from   tensorboardX import SummaryWriter


class TensorBoardLogger(object):
    def __init__(self, tb_log_dir, exp_string, local_rank):
        """ß
        Initialize summary writer
        """
        self.exp_string = exp_string
        self.tb_log_dir = tb_log_dir
        self.val_img_dir = os.path.join(self.tb_log_dir, 'val_image')
        self.local_rank = local_rank

        if local_rank == 0:
            os.makedirs(self.tb_log_dir, exist_ok=True)
            os.makedirs(self.val_img_dir, exist_ok=True)

            self.writer = SummaryWriter(self.tb_log_dir+'/' + self.exp_string)
        else:
            self.writer = None

    def scalar_summary(self, tag, value, step, phase='train'):
        if self.local_rank == 0:
            sum_name = '{}/{}'.format(phase.capitalize(), tag)
            self.writer.add_scalar(sum_name, value, step)

    def image_summary(self, image_set, step, phase='train', save_val=True):
        """
        Record image in tensorboard
        The input image should be a numpy array with shape (C, H, W) like a torch tensor
        :param image_set: dict of images
        :param step:
        :param phase:
        :param save_val: save images in folder in validation or testing
        :return:
        """
        if self.local_rank == 0:
            for tag, image_numpy in image_set.items():
                sum_name = '{}/{}'.format(phase.capitalize(), tag)
                image_numpy = image_numpy.transpose([1, 2, 0])

                image_numpy = cv2.resize(image_numpy, (360, 360), interpolation=cv2.INTER_NEAREST)

                if len(image_numpy.shape) == 2:
                    image_numpy = image_numpy[None, :,:]
                else:
                    image_numpy = image_numpy.transpose([2, 0, 1])
                self.writer.add_image(sum_name, image_numpy, step)

            if (phase=='test') and save_val:
                tags = list(image_set.keys())
                image_pack = self._reshape_rgb(image_set[tags[0]])
                image_pack = cv2.resize(image_pack, (512, 512), interpolation=cv2.INTER_NEAREST)

                for tag in tags[1:]:
                    image = self._reshape_rgb(image_set[tag])
                    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
                    image_pack = np.concatenate((image_pack, image), axis=1)

                cv2.imwrite(os.path.join(self.val_img_dir, 'val_{:d}'.format(step)+'.png'), image_pack)

    @staticmethod
    def _reshape_rgb(image):
        """
        Transform RGB/L -> BGR for OpenCV
        """
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = image.transpose([1, 2, 0])
            image = image[...,::-1]
        elif len(image.shape) == 3 and image.shape[0] == 1:
            image = image.transpose([1, 2, 0])
            image = np.repeat(image, 3, axis=2)
        elif len(image.shape) == 2:
            # image = image.transpose([1,0])
            image = np.stack((image, image, image), axis=2)
        else:
            raise ValueError('Image shape {} not supported to save'.format(image.shape))
        return image

    def __del__(self):
        if self.writer is not None:
            self.writer.close()


if __name__ == '__main__':
    epoch = TensorBoardLogger(None, None, None)
