from srcs.utils.utils_image_kair import tensor2uint, imsave
from srcs.utils.util import instantiate
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
import os,hydra
import logging
import numpy as np


def testing(gpus, config):
    test_worker(gpus, config)


def test_worker(gpus, config):
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)

    # logger & dir setting
    logger = logging.getLogger('test')
    if not os.path.exists(config.outputs_dir):
        os.makedirs(config.outputs_dir)


    # instantiate model
    model = instantiate(config.arch)
    logger.info(model)
    if len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)

    # load weight
    model.load_state_dict(torch.load(config.checkpoint))

    # setup data_loader instances
    data_loader = instantiate(config.test_data_loader)
    kc = config.kc

    # test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log = test(data_loader, kc,  model, device,  None, config)
    logger.info(log)


def test(data_loader, kc, model,  device, metrics, config):
    '''
    test step
    '''

    # init
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (data, kernel, target) in enumerate(tqdm(data_loader, desc='Testing')):
            data, target, kernel = data.to(
                device), target.to(device), kernel.to(device)

            # illum level adjusting
            data = data/kc
            alpha = 40/kc
            alpha = torch.Tensor([alpha]).view(1, 1, 1, 1).to(device)

            # eval
            output = model(data, kernel, alpha)

            # final output
            output = output[-1]

            # save some sample images
            for k, (in_img, out_img, gt_img) in enumerate(zip(data, output, target)):
                in_img = tensor2uint(in_img)
                out_img = tensor2uint(out_img)
                gt_img = tensor2uint(gt_img)

                # crop, for symmetric padding in real test
                H, W = in_img.shape[0]//2, in_img.shape[1]//2
                h = np.int32(H/2)
                w = np.int32(W/2)
                in_img = in_img[h:h+H, w:w+W]
                out_img = out_img[h:h+H, w:w+W]

                imsave(
                    out_img, f'{hydra.utils.get_original_cwd()}/results/physical_reteained_results/deblur_tmp_img{i+1:02d}.png')
