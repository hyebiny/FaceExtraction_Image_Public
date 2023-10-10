"""
Based on https://github.com/yucornetto/MGMatting/blob/main/code-base/networks/generators.py
"""

import torch
import torch.nn as nn

from   utils import CONFIG
from   networks import encoders, decoders, ops
from   networks.aematter_model import AEMatter
from   networks.modenet import MODNet


class Mgmatting_Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Mgmatting_Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image):
        # inp = torch.cat((image, guidance), dim=1)
        embedding, mid_fea = self.encoder(image)
        embedding = self.aspp(embedding)
        pred = self.decoder(embedding, mid_fea)

        return pred


def get_generator(name):
    if name == 'mgmatting':
        generator = Mgmatting_Generator(encoder='res_shortcut_encoder_29', decoder='res_shortcut_decoder_22')
    elif name == 'aematter':
        generator = AEMatter()
    elif name == 'modnet':
        generator = MODNet()
    else:
        raise NotImplementedError("Unknown model name {}".format(name))
    return generator