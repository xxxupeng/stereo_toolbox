import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
import cv2


from stereo_toolbox.datasets import KITTI2015_Dataset, KITTI2012_Dataset, MiddleburyEval3_Dataset, ETH3D_Dataset


def writePFM(file, image, scale=1):
    file = open(file, 'wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write(b'PF\n' if color else b'Pf\n')
    file.write(b'%d %d\n' % (image.shape[1], image.shape[0]))
    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(b'%f\n' % scale)
    image.tofile(file) 


def benchmark_submission(model, device='cuda:0', save_dir='./tmp/'):
    """
    Submission to KITTI 2015, KITTI 2012, Middlebury Eval3, and ETH3D benchmarks.
    """

    model = model.to(device).eval()

    dataset_names = ['kitti2015', 'kitti2012', 'middlebury', 'eth3d']

    for idx, (dataset, split) in enumerate(zip([KITTI2015_Dataset, KITTI2012_Dataset, MiddleburyEval3_Dataset, ETH3D_Dataset],
                                               ['test', 'test', 'testH', 'test'])):
        print(f'Processing {dataset_names[idx]} dataset...')

        testdataloader = DataLoader(dataset(split=split, training=False),
                                    batch_size=1, num_workers=16, shuffle=False, drop_last=False)
        
        for i, data in enumerate(tqdm(testdataloader)):
            left = data['left'].to(device)
            right = data['right'].to(device)
            top_pad = data['top_pad']
            right_pad = data['right_pad']
            left_filename = data['left_filename'][0]

            with torch.no_grad():
                output = model(left, right).squeeze()
            
            assert output.dim() == 2, 'Output should be a 2D disparity map.'
            if right_pad > 0:
                output = output[top_pad:, :-right_pad]
            else:
                output = output[top_pad:, :]

            if idx == 0:
                output = (output * 256).cpu().numpy().astype(np.uint16)
                filename = os.path.join(save_dir, f'kitti2015/disp_0/{left_filename[-13:]}')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, output)

            elif idx == 1:
                output = (output * 256).cpu().numpy().astype(np.uint16)
                filename = os.path.join(save_dir, f'kitti2012/disp_0/{left_filename[-13:]}')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, output)

            elif idx == 2:
                output = output.cpu().numpy().astype(np.float32)
                filename = os.path.join(save_dir, f"middlebury/{left_filename[left_filename.find('testH/'):].replace('im0.png', 'disp0Ours.pfm')}") # middlebury 命名要求为 disp0xxx.pfm
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                writePFM(filename, output)
            
                filename = filename.replace('disp0Ours.pfm', 'timeOurs.txt') # 同时需要名为 timexxx.txt 的时间文件
                with open(filename, 'w') as f:
                    f.write('0.xx')

            elif idx == 3:
                output = output.cpu().numpy().astype(np.float32)
                filename = os.path.join(save_dir, f"eth3d/low_res_two_view/{left_filename.split('/')[-2]}.pfm")
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                writePFM(filename, output)

                filename = filename.replace('.pfm', '.txt')
                with open(filename, 'w') as f:
                    f.write('runtime 0.xx')


    os.system(f'cd {save_dir}/kitti2015/ && zip -r ../kitti2015.zip .')
    os.system(f'cd {save_dir}kitti2012/disp_0/ && zip -r ../../kitti2012.zip .')
    os.system(f'cd {save_dir}/middlebury/ && zip -r ../middlebury.zip .')
    os.system(f'cd {save_dir}/eth3d/ && zip -r ../eth3d.zip .')


                


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
    from datasets import KITTI2015_Dataset, KITTI2012_Dataset, MiddleburyEval3_Dataset, ETH3D_Dataset
    from models import load_checkpoint_flexible, IGEVStereo

    model = load_checkpoint_flexible(IGEVStereo(),
                                 '/home/xp/stereo_toolbox/stereo_toolbox/models/IGEVStereo/sceneflow.pth',
                                 )

    benchmark_submission(model, device='cuda:3')

        
    
