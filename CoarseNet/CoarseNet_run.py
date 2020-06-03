"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep
  Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks
                 and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""

from __future__ import absolute_import
from __future__ import division

import argparse

from keras import backend as K

from MinutiaeNet_utils import *
from CoarseNet_utils import *
from CoarseNet_model import *

import os


# Configuring Keras y Tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = K.tf.ConfigProto(gpu_options=K.tf.GPUOptions(allow_growth=True))
sess = K.tf.Session(config=config)
K.set_session(sess)


# Path to models
pretrain_dir = '../Models/CoarseNet.h5'
FineNet_dir = '../Models/FineNet.h5'


def main(indir, output_dir, use_fine_net, mode, iext):
    if mode == 'deploy':
        # logging = init_log(output_dir)
        deploy_with_GT(
            indir, output_dir=output_dir,
            model_path=pretrain_dir, FineNet_path=FineNet_dir,
            isHavingFineNet=use_fine_net, iext=iext
        )

        # evaluate_training(model_dir=pretrain_dir,
        #                   test_set=indir, logging=logging)

    elif mode == 'inference':
        # logging = init_log(output_dir)
        inference(
            indir, output_dir=output_dir,
            model_path=pretrain_dir, FineNet_path=FineNet_dir,
            file_ext=iext, isHavingFineNet=use_fine_net)
    else:
        pass


def parse_arguments(argv):
    """Script argument parser"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--odir', type=str,
        help='Path to location where extracted templates should be stored'
    )
    parser.add_argument(
        '--idir', type=str, help='Path to directory containing input images'
    )
    parser.add_argument("--FineNet", "--norotate", required=False,
                        action='store_true',
                        help="Indicates whether to use FineNet")
    parser.add_argument(
        '--mode', type=str, required=False, default='inference',
        help='Run mode (deploy, inference). Default=inference'
    )

    parser.add_argument(
        '--itype', type=str, required=False, default='.bmp',
        help='Image file extension'
    )

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])

    main(args.idir, args.odir, args.FineNet, args.mode, args.itype)
