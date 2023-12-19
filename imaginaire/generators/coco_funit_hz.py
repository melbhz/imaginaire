# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md
import torch
from torch import nn

from imaginaire.generators.funit import (MLP, ContentEncoder, Decoder,
                                         StyleEncoder)


class Generator(nn.Module):
    r"""COCO-FUNIT Generator.
    """

    def __init__(self, gen_cfg, data_cfg):
        r"""COCO-FUNIT Generator constructor.

        Args:
            gen_cfg (obj): Generator definition part of the yaml config file.
            data_cfg (obj): Data definition part of the yaml config file.
        """
        super().__init__()
        self.generator = COCOFUNITTranslator(**vars(gen_cfg))

    def forward(self, data):
        r"""In the FUNIT's forward pass, it generates a content embedding and
        a style code from the content image, and a style code from the style
        image. By mixing the content code and the style code from the content
        image, we reconstruct the input image. By mixing the content code and
        the style code from the style image, we have a translation output.

        Args:
            data (dict): Training data at the current iteration.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_a = self.generator.style_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])
        images_trans = self.generator.decode(content_a, style_b)
        images_recon = self.generator.decode(content_a, style_a)

        net_G_output = dict(images_trans=images_trans,
                            images_recon=images_recon)
        return net_G_output

    def inference(self, data, keep_original_size=True):
        r"""COCO-FUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_content (tensor): Content images.
              - images_style (tensor): Style images.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            keep_original_size (bool): If ``True``, output image is resized
            to the input content image size.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])
        output_images = self.generator.decode(content_a, style_b)
        if keep_original_size:
            height = data['original_h_w'][0][0]
            width = data['original_h_w'][0][1]
            # print('( H, W) = ( %d, %d)' % (height, width))
            output_images = torch.nn.functional.interpolate(
                output_images, size=[height, width])
        file_names = data['key']['images_content'][0]
        return output_images, file_names

    def inference_tensor(self, content_a, style_b):
        output_images = self.generator.decode(content_a, style_b)
        return output_images

    def inference_test(self, data, keep_original_size=True):
        r"""COCO-FUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_content (tensor): Content images.
              - images_style (tensor): Style images.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            keep_original_size (bool): If ``True``, output image is resized
            to the input content image size.
        """
        content_a = self.generator.content_encoder(data['images_content'])
        style_b = self.generator.style_encoder(data['images_style'])

        print(f'data["key"]:= {data["key"]}')
        # print(f"data['images_content']:= {data['images_content']}\n data['images_content'].size():= {data['images_content'].size()}")
        print(f"data['images_content'].size():= {data['images_content'].size()}")
        # print(f"data['images_style']:= {data['images_style']}\n data['images_style'].size():= {data['images_style'].size()}")
        print(f"data['images_style'].size():= {data['images_style'].size()}")
        # print(f'content_a:= {content_a}\n content_a.size():= {content_a.size()}')
        print(f'content_a.size():= {content_a.size()}')
        # print(f'style_b:= {style_b}\n style_b.size():= {style_b.size()}')
        print(f'style_b.size():= {style_b.size()}')
        print(f'keep_original_size:= {keep_original_size}')

        output_images = self.generator.decode_test(content_a, style_b)
        if keep_original_size:
            height = data['original_h_w'][0][0]
            width = data['original_h_w'][0][1]
            # print('( H, W) = ( %d, %d)' % (height, width))
            output_images = torch.nn.functional.interpolate(
                output_images, size=[height, width])
        file_names = data['key']['images_content'][0]
        print(f"file_names:= {file_names}")
        return output_images, file_names

    def inference_style_placeholder(self, data, style_tensor, a2b=True, random_style=True):
        r"""MUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_a (tensor): Images from domain A.
              - images_b (tensor): Images from domain B.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            random_style (bool): If ``True``, samples the style code from the
                prior distribution, otherwise uses the style code encoded from
                the input images in the other domain.
        """
        if a2b:
            input_key = 'images_a'
            content_encode = self.autoencoder_a.content_encoder
            style_encode = self.autoencoder_b.style_encoder
            decode = self.autoencoder_b.decode
        else:
            input_key = 'images_b'
            content_encode = self.autoencoder_b.content_encoder
            style_encode = self.autoencoder_a.style_encoder
            decode = self.autoencoder_a.decode

        content_images = data[input_key]
        content = content_encode(content_images)
        if random_style:
            if style_tensor == 'random':
                style_channels = self.autoencoder_a.style_channels
                style = torch.randn(content.size(0), style_channels, 1, 1,
                                    device=torch.device('cuda'))
                file_names = data['key'][input_key]['filename']
            elif style_tensor == '':
                print(
                    "Style tensor required!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, please check your option for munit style!!")
            else:
                style = style_tensor
                file_names = data['key'][input_key]['filename']

            # print(f'data: {data}')
            # print(f'style: {style}')
            # print(f'file_names: {file_names}')
        else:
            style_key = 'images_b' if a2b else 'images_a'
            assert style_key in data.keys(), \
                "{} must be provided when 'random_style' " \
                "is set to False".format(style_key)
            style_images = data[style_key]
            style = style_encode(style_images)
            file_names = \
                [content_name + '_style_' + style_name
                 for content_name, style_name in
                 zip(data['key'][input_key]['filename'],
                     data['key'][style_key]['filename'])]

        output_images = decode(content, style)
        return output_images, file_names

    def get_style_code_placeholder(self, data, a2b=True, random_style=True):
        if a2b:
            input_key = 'images_a'
            style_encode = self.autoencoder_b.style_encoder
        else:
            input_key = 'images_b'
            style_encode = self.autoencoder_a.style_encoder

        if True:
            style_key = 'images_b' if a2b else 'images_a'
            style_images = data[style_key]
            style = style_encode(style_images)
            style_filenames = data['key'][style_key]['filename']
            style_dirnames = data['key'][style_key]['lmdb_root']
            style_dirname = style_dirnames[0] + '/' + style_key

        return style, style_filenames, style_dirname

    def get_content_and_style_code(self, data, keep_original_size=True):
        r"""COCO-FUNIT inference.

        Args:
            data (dict): Training data at the current iteration.
              - images_content (tensor): Content images.
              - images_style (tensor): Style images.
            a2b (bool): If ``True``, translates images from domain A to B,
                otherwise from B to A.
            keep_original_size (bool): If ``True``, output image is resized
            to the input content image size.
        """
        content = self.generator.content_encoder(data['images_content'])
        style = self.generator.style_encoder(data['images_style'])

        content_filenames = data['key']['images_content'][0]
        style_filenames = data['key']['images_style'][0]

        content_dirname = 'images_content'
        style_dirname = 'images_style'

        return content, content_filenames, content_dirname, style, style_filenames, style_dirname

    def get_content_and_style_code_squeezed(self, data, keep_original_size=True):
        content, content_filenames, content_dirname, style, style_filenames, style_dirname = self.get_content_and_style_code(data, keep_original_size=keep_original_size)
        content, style = self.generator.squeeze_code(content, style)
        return content, content_filenames, content_dirname, style, style_filenames, style_dirname

    def get_contents_and_styles_placeholder(self, data, a2b=True, random_style=True):
        if a2b:
            input_key = 'images_a'
            content_encode = self.autoencoder_a.content_encoder
            style_key = 'images_b'
            style_encode = self.autoencoder_b.style_encoder
        else:
            input_key = 'images_b'
            content_encode = self.autoencoder_b.content_encoder
            style_key = 'images_a'
            style_encode = self.autoencoder_a.style_encoder

        if True:
            content_images = data[input_key]
            content = content_encode(content_images)
            content_filenames = data['key'][input_key]['filename']
            content_dirnames = data['key'][input_key]['lmdb_root']
            content_dirname = content_dirnames[0] + '/' + input_key

        if True:
            style_images = data[style_key]
            style = style_encode(style_images)
            style_filenames = data['key'][style_key]['filename']
            style_dirnames = data['key'][style_key]['lmdb_root']
            style_dirname = style_dirnames[0] + '/' + style_key

        return content_images, content, content_filenames, content_dirname, style_images, style, style_filenames, style_dirname

    def inference_tensor_random_placeholder(self, content, a2b=True, random_style=True):
        if a2b:
            decode = self.autoencoder_b.decode
        else:
            decode = self.autoencoder_a.decode

        style_channels = self.autoencoder_a.style_channels
        style = torch.randn(content.size(0), style_channels, 1, 1,
                            device=torch.device('cuda'))

        output_images = decode(content, style)
        return output_images


class COCOFUNITTranslator(nn.Module):
    r"""COCO-FUNIT Generator architecture.

    Args:
        num_filters (int): Base filter numbers.
        num_filters_mlp (int): Base filter number in the MLP module.
        style_dims (int): Dimension of the style code.
        usb_dims (int): Dimension of the universal style bias code.
        num_res_blocks (int): Number of residual blocks at the end of the
            content encoder.
        num_mlp_blocks (int): Number of layers in the MLP module.
        num_downsamples_content (int): Number of times we reduce
            resolution by 2x2 for the content image.
        num_downsamples_style (int): Number of times we reduce
            resolution by 2x2 for the style image.
        num_image_channels (int): Number of input image channels.
        weight_norm_type (str): Type of weight normalization.
            ``'none'``, ``'spectral'``, or ``'weight'``.
    """

    def __init__(self,
                 num_filters=64,
                 num_filters_mlp=256,
                 style_dims=64,
                 usb_dims=1024,
                 num_res_blocks=2,
                 num_mlp_blocks=3,
                 num_downsamples_style=4,
                 num_downsamples_content=2,
                 num_image_channels=3,
                 weight_norm_type='',
                 **kwargs):
        super().__init__()

        self.style_encoder = StyleEncoder(num_downsamples_style,
                                          num_image_channels,
                                          num_filters,
                                          style_dims,
                                          'reflect',
                                          'none',
                                          weight_norm_type,
                                          'relu')

        self.content_encoder = ContentEncoder(num_downsamples_content,
                                              num_res_blocks,
                                              num_image_channels,
                                              num_filters,
                                              'reflect',
                                              'instance',
                                              weight_norm_type,
                                              'relu')

        self.decoder = Decoder(self.content_encoder.output_dim,
                               num_filters_mlp,
                               num_image_channels,
                               num_downsamples_content,
                               'reflect',
                               weight_norm_type,
                               'relu')

        self.usb = torch.nn.Parameter(torch.randn(1, usb_dims))

        self.mlp = MLP(style_dims,
                       num_filters_mlp,
                       num_filters_mlp,
                       num_mlp_blocks,
                       'none',
                       'relu')

        num_content_mlp_blocks = 2
        num_style_mlp_blocks = 2
        self.mlp_content = MLP(self.content_encoder.output_dim,
                               style_dims,
                               num_filters_mlp,
                               num_content_mlp_blocks,
                               'none',
                               'relu')

        self.mlp_style = MLP(style_dims + usb_dims,
                             style_dims,
                             num_filters_mlp,
                             num_style_mlp_blocks,
                             'none',
                             'relu')

    def forward(self, images):
        r"""Reconstruct the input image by combining the computer content and
        style code.

        Args:
            images (tensor): Input image tensor.
        """
        # reconstruct an image
        content, style = self.encode(images)
        images_recon = self.decode(content, style)
        return images_recon

    def encode(self, images):
        r"""Encoder images to get their content and style codes.

        Args:
            images (tensor): Input image tensor.
        """
        style = self.style_encoder(images)
        content = self.content_encoder(images)
        return content, style

    def decode(self, content, style):
        r"""Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images

    def squeeze_code(self, content, style):
        r"""Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        content_style_code = content.mean(3).mean(2)
        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        coco_style = self.mlp(coco_style)
        # images = self.decoder(content, coco_style)
        # return images
        return content, coco_style



    def decode_test(self, content, style):
        r"""Generate images by combining their content and style codes.

        Args:
            content (tensor): Content code tensor.
            style (tensor): Style code tensor.
        """
        content_style_code = content.mean(3).mean(2)
        # print(f'content.mean(3):= {content.mean(3)}\n content.mean(3).size():= {content.mean(3).size()}')
        print(f'content.mean(3).size():= {content.mean(3).size()}')
        print(f'content.mean(3).mean(2):= {content.mean(3).mean(2)}\n content.mean(3).mean(2).size():= {content.mean(3).mean(2).size()}')
        # print(f'content:= {content}\n content.size():= {content.size()}')
        print(f'content.size():= {content.size()}')

        content_style_code = self.mlp_content(content_style_code)
        batch_size = style.size(0)
        usb = self.usb.repeat(batch_size, 1)
        style = style.view(batch_size, -1)
        print(f'style:= {style}\n style.size():= {style.size()}')
        style_in = self.mlp_style(torch.cat([style, usb], 1))
        coco_style = style_in * content_style_code
        print(f'coco_style:= {coco_style}\n coco_style.size():= {coco_style.size()}')
        coco_style = self.mlp(coco_style)
        images = self.decoder(content, coco_style)
        return images