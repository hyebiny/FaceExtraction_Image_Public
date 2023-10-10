####
# from https://github.com/ZHKKKe/MODNet/blob/master/src/models/backbones/__init__.py
###


from networks.backbones.wrapper import *




#------------------------------------------------------------------------------
#  Replaceable Backbones
#------------------------------------------------------------------------------

SUPPORTED_BACKBONES = {
    'mobilenetv2': MobileNetV2Backbone,
}