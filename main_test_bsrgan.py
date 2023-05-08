import os.path
import logging
import torch

from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net
from tqdm import tqdm


"""
Spyder (Python 3.6-3.7)
PyTorch 1.4.0-1.8.1
Windows 10 or Linux
Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/BSRGAN
        https://github.com/cszn/KAIR
If you have any question, please feel free to contact with me.
Kai Zhang (e-mail: cskaizhang@gmail.com)
by Kai Zhang ( March/2020 --> March/2021 --> )
This work was previously submitted to CVPR2021.

# --------------------------------------------
@inproceedings{zhang2021designing,
  title={Designing a Practical Degradation Model for Deep Blind Image Super-Resolution},
  author={Zhang, Kai and Liang, Jingyun and Van Gool, Luc and Timofte, Radu},
  booktitle={arxiv},
  year={2021}
}
# --------------------------------------------

"""


def main():
    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

    testsets = 'testsets'
    testset_L = 'full_img'
    input_path = os.path.join(testsets, testset_L)
    image_paths = os.listdir(input_path)
    model_names = ['ESRGAN']
    device = torch.device('cuda')

    for model_name in model_names:
        if model_name == 'BSRGANx2':
            sf = 2
        else:
            sf = 4
        model_path = os.path.join('model_zoo', model_name+'.pth')          # set model path
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        torch.cuda.set_device(2)
        logger.info('{:>16s} : {:<d}'.format('GPU ID', torch.cuda.current_device()))
        torch.cuda.empty_cache()

        # --------------------------------
        # define network and load model
        # --------------------------------
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)  # define network
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        torch.cuda.empty_cache()

        output_path = os.path.join(testsets, testset_L+'_'+model_name)
        util.mkdir(output_path)

        logger.info('{:>16s} : {:s}'.format('Input Path', input_path))
        logger.info('{:>16s} : {:s}'.format('Output Path', output_path))

        # for idx, filename in enumerate(tqdm(image_paths)):
        for idx, filename in enumerate(image_paths):
            # --------------------------------
            # (1) img_L
            # --------------------------------
            logger.info('{:->4d} --> {:<s} --> x{:<d} --> {:<s}'.format(idx, model_name, sf, filename))
            img = util.imread_uint(os.path.join(input_path, filename), n_channels=3)
            img = util.uint2tensor4(img)
            img = img.to(device)

            # --------------------------------
            # (2) inference
            # --------------------------------
            img = model(img)

            # --------------------------------
            # (3) img
            # --------------------------------
            img = util.tensor2uint(img)
            util.imsave(img, os.path.join(output_path, filename))


if __name__ == '__main__':
    main()
