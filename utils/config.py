from easydict import EasyDict

# Base default config
CONFIG = EasyDict({})
# to indicate this is a default setting, should not be changed by user
CONFIG.is_default = True
CONFIG.version = "baseline"
CONFIG.phase = "train"
# distributed training
CONFIG.dist = False
# global variables which will be assigned in the runtime
CONFIG.local_rank = 0
CONFIG.gpu = 0
CONFIG.world_size = 1

# Model config
CONFIG.model = EasyDict({})
# use pretrained checkpoint as encoder
CONFIG.model.imagenet_pretrain = True
CONFIG.model.imagenet_pretrain_path = "/home/liyaoyi/Source/python/attentionMatting/pretrain/model_best_resnet34_En_nomixup.pth"
CONFIG.model.batch_size = 16
CONFIG.model.mask_channel = 0
CONFIG.model.model = 'resnet18'

# Training config
CONFIG.train = EasyDict({})
CONFIG.train.total_epoch = 10
CONFIG.train.val_epoch = 1
# basic learning rate of optimizer
CONFIG.train.G_lr = 1e-3
# beta1 and beta2 for Adam
CONFIG.train.beta1 = 0.5
CONFIG.train.beta2 = 0.999
# weight of different losses
CONFIG.train.rec_weight = 1
CONFIG.train.lap_weight = 1
CONFIG.train.com_weight = 1
# clip large gradient
CONFIG.train.clip_grad = True
# resume the training (checkpoint file name)
CONFIG.train.resume_checkpoint = None
# reset the learning rate (this option will reset the optimizer and learning rate scheduler and ignore warmup)
CONFIG.train.reset_lr = False


# Dataloader config
CONFIG.data = EasyDict({})
CONFIG.data.cutmask_prob = 0
CONFIG.data.workers = 0
# data path for training and validation in training phase
CONFIG.data.train_root = None
CONFIG.data.test_root = None
CONFIG.data.fg_dir = None
CONFIG.data.bg_txt = None
CONFIG.data.rand_dir = None
CONFIG.data.folder_list = None
CONFIG.data.sim_list1 = None
CONFIG.data.sim_list2 = None
CONFIG.data.num_sample = 2
CONFIG.data.img_ext = '.jpg'
CONFIG.data.mask_ext = '.png'
# feed forward image size (untested)
CONFIG.data.crop_size = 512
CONFIG.data.occlusion = 1.0
CONFIG.data.real_world_aug = True
CONFIG.data.random_interp = True


# # Logging config
# CONFIG.log = EasyDict({})
# CONFIG.log.tensorboard_path = "./logs/tensorboard"
# CONFIG.log.tensorboard_step = 100
# # save less images to save disk space
# CONFIG.log.tensorboard_image_step = 500
# CONFIG.log.logging_path = "./logs/stdout"
# CONFIG.log.logging_step = 10
# CONFIG.log.logging_level = "DEBUG"
# CONFIG.log.checkpoint_path = "./checkpoints"
# CONFIG.log.checkpoint_step = 10000


def load_config(custom_config, default_config=CONFIG, prefix="CONFIG"):
    """
    This function will recursively overwrite the default config by a custom config
    :param default_config:
    :param custom_config: parsed from config/config.toml
    :param prefix: prefix for config key
    :return: None
    """
    if "is_default" in default_config:
        default_config.is_default = False

    for key in custom_config.keys():
        full_key = ".".join([prefix, key])
        if key not in default_config:
            raise NotImplementedError("Unknown config key: {}".format(full_key))
        elif isinstance(custom_config[key], dict):
            if isinstance(default_config[key], dict):
                load_config(default_config=default_config[key],
                            custom_config=custom_config[key],
                            prefix=full_key)
            else:
                raise ValueError("{}: Expected {}, got dict instead.".format(full_key, type(custom_config[key])))
        else:
            if isinstance(default_config[key], dict):
                raise ValueError("{}: Expected dict, got {} instead.".format(full_key, type(custom_config[key])))
            else:
                default_config[key] = custom_config[key]