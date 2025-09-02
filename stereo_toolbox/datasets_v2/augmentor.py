import numpy as np
import albumentations as A
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class StereoAugmentor:
    """
    StereoAugmentor is a class designed for performing data augmentation on stereo image pairs. 
    It includes a variety of photometric and spatial transformations to enhance the diversity 
    of training data for stereo vision tasks.
    Attributes:
        crop_size (list): The size of the crop applied to the images (height, width).
        min_scale (float): Minimum scaling factor for spatial transformations.
        max_scale (float): Maximum scaling factor for spatial transformations.
        max_stretch (float): Maximum stretching factor for spatial transformations.
        pad_scale (int): Scale factor for padding the images.
        eraser_size (list): Maximum size of the random eraser (height, width).
        crop_prob (float): Probability of applying cropping.
        color_aug_prob (float): Probability of applying color augmentation.
        color_asym_prob (float): Probability of applying asymmetric color augmentation.
        spatial_aug_prob (float): Probability of applying spatial transformations.
        stretch_prob (float): Probability of applying stretching during spatial transformations.
        v_flip_prob (float): Probability of applying vertical flipping.
        pad_prob (float): Probability of applying padding.
        eraser_prob (float): Probability of applying random erasing.
    Methods:
        pad_image(img):
            Pads the input image to the nearest multiple of `pad_scale`.
            Args:
                img (numpy.ndarray): Input image to be padded.
            Returns:
                tuple: Padded image, top padding size, and right padding size.
        eraser_transform(tgt):
            Applies random erasing to the target image.
            Args:
                tgt (numpy.ndarray): Target image to apply random erasing.
            Returns:
                numpy.ndarray: Transformed target image.
        color_transform(ref, tgt):
            Applies photometric augmentation to the reference and target images.
            Args:
                ref (numpy.ndarray): Reference image.
                tgt (numpy.ndarray): Target image.
            Returns:
                tuple: Transformed reference and target images.
        spatial_transform(ref, tgt, gt_disp=None, noc_mask=None, raw_ref=None, raw_tgt=None):
            Applies spatial transformations such as scaling, flipping, cropping, and padding.
            Args:
                ref (numpy.ndarray): Reference image.
                tgt (numpy.ndarray): Target image.
                gt_disp (numpy.ndarray, optional): Ground truth disparity map.
                noc_mask (numpy.ndarray, optional): Non-occluded mask.
                raw_ref (numpy.ndarray, optional): Raw reference image.
                raw_tgt (numpy.ndarray, optional): Raw target image.
            Returns:
                tuple: Transformed images, disparity map, masks, and padding information.
        __call__(ref, tgt, gt_disp=None, noc_mask=None, raw_ref=None, raw_tgt=None):
            Applies the full augmentation pipeline to the input images and optional data.
            Args:
                ref (numpy.ndarray): Reference image.
                tgt (numpy.ndarray): Target image.
                gt_disp (numpy.ndarray, optional): Ground truth disparity map.
                noc_mask (numpy.ndarray, optional): Non-occluded mask.
                raw_ref (numpy.ndarray, optional): Raw reference image.
                raw_tgt (numpy.ndarray, optional): Raw target image.
            Returns:
                dict: Augmented data including images, disparity map, masks, and padding information.
    """

    def __init__(self,
            crop_size=[288,576],
            min_scale=-0.2, 
            max_scale=0.5, 
            max_stretch=0.2,
            pad_scale=32,
            eraser_size=[50,50],

            crop_prob = 1.0,
            color_aug_prob=1.0,
            color_asym_prob=0.2, 
            spatial_aug_prob=0.8, 
            eraser_prob=0.8,
            stretch_prob=0.2,
            v_flip_prob=0.05,
            pad_prob=0.0,
        ):
        self.crop_size = crop_size
        self.crop_prob = crop_prob
        self.color_aug_prob =color_aug_prob
        self.color_asym_prob = color_asym_prob
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = stretch_prob
        self.max_stretch = max_stretch
        self.v_flip_prob = v_flip_prob
        self.pad_prob = pad_prob
        self.pad_scale = pad_scale
        self.eraser_prob = eraser_prob
        self.eraser_size = eraser_size

        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=0.8),
            A.RandomGamma(p=0.5),
            A.OneOf([
                A.RGBShift(p=1.0),
                A.ChannelDropout(p=1.0),
                A.ChannelShuffle(p=1.0),
                A.ToGray(p=1.0),
            ], p=0.05),
            A.OneOf([
                A.ImageCompression(p=1.0),
                A.GaussNoise(p=1.0),
                A.MotionBlur(p=1.0),
                A.GaussianBlur( p=1.0),
                A.Blur(p=1.0),
                A.MedianBlur(p=1.0),
            ], p=0.05),
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomSnow(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomSunFlare(p=1.0),
            ], p=0.05),
        ])
        

    def pad_image(self, img):
        h, w = img.shape[:2]
        new_h = int(np.ceil(h / self.pad_scale) * self.pad_scale)
        new_w = int(np.ceil(w / self.pad_scale) * self.pad_scale)
        pad_top = new_h - h
        pad_right = new_w - w

        if len(img.shape) == 3:
            padded_img = np.pad(img, ((0, pad_top), (0, pad_right), (0, 0)), mode='constant', constant_values=0)
        elif len(img.shape) == 2:
            padded_img = np.pad(img, ((0, pad_top), (0, pad_right)), mode='constant', constant_values=0)
        else:
            raise ValueError(f"Unsupported image shape for padding. Current shape: {img.shape}")
        
        return padded_img, pad_top, pad_right
    
    
    def eraser_transform(self, tgt):
        """ Random erasing """
        if np.random.rand() < self.eraser_prob:
            h, w = tgt.shape[:2]
            dy = np.random.randint(0, self.eraser_size[0])
            dx = np.random.randint(0, self.eraser_size[1])

            y = np.random.randint(0, h - dy)
            x = np.random.randint(0, w - dx)

            tgt[y:y+dy, x:x+dx, :] = tgt.mean(axis=(0,1))
        return tgt


    def color_transform(self, ref, tgt):
        """ Photometric augmentation """
        if np.random.rand() < self.color_aug_prob:
            # asymmetric
            if np.random.rand() < self.color_asym_prob:
                ref = self.color_aug(image=ref)['image']
                tgt = self.color_aug(image=tgt)['image']
            # symmetric
            else:
                image_stack = np.concatenate([ref, tgt], axis=0)
                image_stack = self.color_aug(image=image_stack)['image']
                ref, tgt = np.split(image_stack, 2, axis=0)
        
        return ref, tgt
        

    def spatial_transform(self, ref, tgt, gt_disp=None, noc_mask=None, raw_ref=None, raw_tgt=None):
        ref_h, ref_w = ref.shape[:2]
        min_scale = np.maximum(
                        (self.crop_size[0] + 8) / float(ref_h), 
                        (self.crop_size[1] + 8) / float(ref_w)
                    )
        
        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        # rescale the images
        if np.random.rand() < self.spatial_aug_prob:
            ref = cv2.resize(ref, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            tgt = cv2.resize(tgt, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            if gt_disp is not None:
                gt_disp = cv2.resize(gt_disp, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST) * scale_x
            if noc_mask is not None:
                noc_mask = cv2.resize(noc_mask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
            if raw_ref is not None:
                raw_ref = cv2.resize(raw_ref, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            if raw_tgt is not None:
                raw_tgt = cv2.resize(raw_tgt, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        # flip the images
        if np.random.rand() < self.v_flip_prob:
            ref = np.flip(ref, axis=0)
            tgt = np.flip(tgt, axis=0)

            if gt_disp is not None:
                gt_disp = np.flip(gt_disp, axis=0)
            if noc_mask is not None:
                noc_mask = np.flip(noc_mask, axis=0)
            if raw_ref is not None:
                raw_ref = np.flip(raw_ref, axis=0)
            if raw_tgt is not None:
                raw_tgt = np.flip(raw_tgt, axis=0)

        # crop images
        if np.random.rand() < self.crop_prob:
            assert (ref.shape[0] > self.crop_size[0]) and (ref.shape[1] > self.crop_size[1])

            y0 = np.random.randint(0, ref.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, ref.shape[1] - self.crop_size[1])

            ref = ref[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            tgt = tgt[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            if gt_disp is not None:
                gt_disp = gt_disp[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            if noc_mask is not None:
                noc_mask = noc_mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            if raw_ref is not None:
                raw_ref = raw_ref[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            if raw_tgt is not None:
                raw_tgt = raw_tgt[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        # random erasing
        tgt = self.eraser_transform(tgt)

        # pad images during test
        top_pad, right_pad = None, None
        if np.random.rand() < self.pad_prob:
            ref, top_pad, right_pad = self.pad_image(ref)
            tgt, _, _ = self.pad_image(tgt)

            if gt_disp is not None:
                gt_disp, _, _ = self.pad_image(gt_disp)
            if noc_mask is not None:
                noc_mask, _, _ = self.pad_image(noc_mask)
            if raw_ref is not None:
                raw_ref, _, _ = self.pad_image(raw_ref)
            if raw_tgt is not None:
                raw_tgt, _, _ = self.pad_image(raw_tgt)

        return ref, tgt, gt_disp, noc_mask, raw_ref, raw_tgt, top_pad, right_pad


    def __call__(self, ref, tgt, gt_disp=None, noc_mask=None, raw_ref=None, raw_tgt=None):
        ref, tgt = self.color_transform(ref, tgt)

        ref, tgt, gt_disp, noc_mask, raw_ref, raw_tgt, top_pad, right_pad = self.spatial_transform(
            ref, tgt, 
            gt_disp=gt_disp, 
            noc_mask=noc_mask, 
            raw_ref=raw_ref, 
            raw_tgt=raw_tgt
        )

        aug_data = {
            'ref': np.ascontiguousarray(ref) / 255.,
            'tgt': np.ascontiguousarray(tgt) / 255.,
        }

        if gt_disp is not None:
            aug_data['gt_disp'] = np.ascontiguousarray(gt_disp)
        if noc_mask is not None:
            aug_data['noc_mask'] = np.ascontiguousarray(noc_mask)
        if raw_ref is not None:
            aug_data['raw_ref'] = np.ascontiguousarray(raw_ref) / 255.
        if raw_tgt is not None:
            aug_data['raw_tgt'] = np.ascontiguousarray(raw_tgt) / 255.
        if top_pad is not None:
            aug_data['top_pad'] = np.ascontiguousarray(top_pad)
        if right_pad is not None:
            aug_data['right_pad'] = np.ascontiguousarray(right_pad)

        return aug_data
