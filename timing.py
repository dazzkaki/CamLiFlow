import cv2
import yaml
import utils
import logging
import argparse
import numpy as np
import torch
import torch.optim
import torch.utils.data
from omegaconf import DictConfig
from factory import model_factory
from utils import copy_to_device, load_fpm, disp2pc
from models.utils import timer


class Worker:
    def __init__(self, device: torch.device, cfgs: DictConfig):
        self.cfgs = cfgs
        self.device = device

        logging.info('Creating model: CamLiFlow')
        self.model = model_factory(self.cfgs.model).to(device=self.device)

        logging.info('Loading checkpoint from %s' % self.cfgs.ckpt.path)
        checkpoint = torch.load(self.cfgs.ckpt.path, self.device)
        self.model.load_state_dict(checkpoint['state_dict'], strict=self.cfgs.ckpt.strict)

    def prepare_images_and_depths(self):
        # load images
        image1 = cv2.imread(args.image1)[..., ::-1]
        image2 = cv2.imread(args.image2)[..., ::-1]

        # load disparity maps
        disp1 = -load_fpm(args.disp1)
        disp2 = -load_fpm(args.disp2)

        # lift disparity maps into point clouds
        pc1 = disp2pc(disp1, args.baseline, args.f, args.cx, args.cy)
        pc2 = disp2pc(disp2, args.baseline, args.f, args.cx, args.cy)

        # apply depth mask
        mask1 = (pc1[..., -1] < args.max_depth)
        mask2 = (pc2[..., -1] < args.max_depth)
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # NaN check
        mask1 = np.logical_not(np.isnan(np.sum(pc1, axis=-1)))
        mask2 = np.logical_not(np.isnan(np.sum(pc2, axis=-1)))
        pc1, pc2 = pc1[mask1], pc2[mask2]

        # random sampling
        indices1 = np.random.choice(pc1.shape[0], size=min(args.n_points, pc1.shape[0]), replace=False)
        indices2 = np.random.choice(pc2.shape[0], size=min(args.n_points, pc2.shape[0]), replace=False)
        pc1, pc2 = pc1[indices1], pc2[indices2]

        return image1, image2, pc1, pc2

    @torch.no_grad()
    def run(self):
        self.model.eval()

        image1, image2, pc1, pc2 = self.prepare_images_and_depths()

        # numpy -> torch
        images = np.concatenate([image1, image2], axis=-1).transpose([2, 0, 1])
        images = torch.from_numpy(images).float().unsqueeze(0)
        pcs = np.concatenate([pc1, pc2], axis=1).transpose()
        pcs = torch.from_numpy(pcs).float().unsqueeze(0)
        intrinsics = torch.as_tensor([args.f, args.cx, args.cy]).unsqueeze(0)

        # inference
        inputs = {'images': images, 'pcs': pcs, 'intrinsics': intrinsics}
        inputs = copy_to_device(inputs, self.device)

        for idx in range(5):
            logging.info('Run %d...' % idx)
            timer.clear_timing_stat()
            self.model(inputs)
            for k, v in timer.get_timing_stat().items():
                logging.info('Function "%s" takes %.1fms' % (k, v))

        logging.info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model weights
    parser.add_argument('--weights', required=True, help='Path to pretrained weights')

    # RGB input
    parser.add_argument('--image1', required=False, default='asserts/demo_image1.png')
    parser.add_argument('--image2', required=False, default='asserts/demo_image2.png')

    # disparity input
    parser.add_argument('--disp1', required=False, default='asserts/demo_disp1.pfm')
    parser.add_argument('--disp2', required=False, default='asserts/demo_disp2.pfm')

    # disparity -> point clouds
    parser.add_argument('--n_points', required=False, default=8192)
    parser.add_argument('--max_depth', required=False, default=35.0)

    # camera intrinsics
    parser.add_argument('--baseline', required=False, default=1.0)
    parser.add_argument('--f', required=False, default=1050.0)
    parser.add_argument('--cx', required=False, default=479.5)
    parser.add_argument('--cy', required=False, default=269.5)

    args = parser.parse_args()

    with open('conf/test/things.yaml', encoding='utf-8') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))
        cfgs.ckpt.path = args.weights

    utils.init_logging()

    if torch.cuda.device_count() == 0:
        device = torch.device('cpu')
        logging.info('No CUDA device detected, using CPU for evaluation')
    else:
        device = torch.device('cuda:0')
        logging.info('Using GPU: %s' % torch.cuda.get_device_name(device))

    worker = Worker(device, cfgs)
    worker.run()
