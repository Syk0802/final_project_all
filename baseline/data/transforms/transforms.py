# decompyle3 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.8.13 (default, Aug  1 2025, 09:27:07)
# [GCC 9.4.0]
# Embedded file name: /xxx/extrawork/code/svc_net/data/transforms/transforms.py
# Compiled at: 2026-07-12 14:40:35
# Size of source mod 2**32: 7877 bytes
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""
import math, random, numpy as np, torch

class RandomErasing(object):
    __doc__ = " Randomly selects a rectangle region in an image and erases its pixels.\n        'Random Erasing Data Augmentation' by Zhong et al.\n        See https://arxiv.org/pdf/1708.04896.pdf\n    Args:\n         probability: The probability that the Random Erasing operation will be performed.\n         sl: Minimum proportion of erased area against input image.\n         sh: Maximum proportion of erased area against input image.\n         r1: Minimum aspect ratio of erased area.\n         mean: Erasing value.\n    "

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img
        return img


class RandomOvalErasing(object):
    __doc__ = " Randomly erases one or several rotated oval / irregular regions of an image.\n        'Random Oval Erasing Data Augmentation' by xxx.\n\n    The augmentation is fully configurable:\n        - shape can be a rotated ellipse ('oval'), an irregular blob ('blob') or\n          randomly chosen per region ('random');\n        - each region can be filled with zeros ('zero'), a patch copied from\n          another random location ('patch'), the (ImageNet) mean ('mean'),\n          gaussian noise ('noise') or a random choice of these ('random');\n        - the number of regions per image is random in [min_count, max_count].\n\n    Args:\n         probability: probability that the operation is performed.\n         sl, sh: min / max proportion of a single erased region against the image.\n         r1: min aspect ratio of an erased region.\n         min_count, max_count: min / max number of regions erased per image.\n         fill_mode: 'zero' | 'patch' | 'mean' | 'noise' | 'random'.\n         shape_mode: 'oval' | 'blob' | 'random'.\n         mean: values used by the 'mean' fill mode.\n    "

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, min_count=1, max_count=5, fill_mode='random', shape_mode='random', mean=(0.485, 0.456, 0.406), blob_amp=0.35, blob_nwave=4):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.min_count = min_count
        self.max_count = max_count
        self.fill_mode = fill_mode
        self.shape_mode = shape_mode
        self.mean = mean
        self.blob_amp = blob_amp
        self.blob_nwave = blob_nwave

    def oval_mask(self, a, b, theta=0.0, edge=0.25):
        a = max(a, 1.0)
        b = max(b, 1.0)
        r = int(math.ceil(math.sqrt(a * a + b * b) * (1.0 + edge))) + 1
        (y, x) = np.ogrid[-r:r + 1, -r:r + 1]
        y = y * np.ones((1, 2 * r + 1), dtype=(np.float64))
        x = x * np.ones((2 * r + 1, 1), dtype=(np.float64))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        d = np.sqrt((x_rot / b) ** 2 + (y_rot / a) ** 2)
        mask = np.clip((d - 1.0) / edge, 0.0, 1.0)
        return mask

    def blob_mask(self, a, b, theta=0.0, edge=0.25, amp=0.35, n_wave=3):
        a = max(a, 1.0)
        b = max(b, 1.0)
        r = int(math.ceil(math.sqrt(a * a + b * b) * (1.0 + amp + edge))) + 1
        (y, x) = np.ogrid[-r:r + 1, -r:r + 1]
        y = y * np.ones((1, 2 * r + 1), dtype=(np.float64))
        x = x * np.ones((2 * r + 1, 1), dtype=(np.float64))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_rot = x * cos_t + y * sin_t
        y_rot = -x * sin_t + y * cos_t
        ex = x_rot / b
        ey = y_rot / a
        d = np.sqrt(ex * ex + ey * ey)
        phi = np.arctan2(ey, ex)
        boundary = np.ones_like(d)
        for _ in range(n_wave):
            k = random.randint(2, 5)
            a_k = random.uniform(-amp, amp) / n_wave
            phase = random.uniform(0, 2 * math.pi)
            boundary = boundary + a_k * np.cos(k * phi + phase)

        boundary = np.clip(boundary, 0.4, None)
        mask = np.clip((d - boundary) / edge, 0.0, 1.0)
        return mask

    def _erase_once(self, img):
        mode = self.fill_mode
        if mode == "random":
            mode = random.choice(["zero", "patch", "mean", "noise"])
        shape = self.shape_mode
        if shape == "random":
            shape = random.choice(["rect", "oval", "blob"])
        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            theta = random.uniform(0, math.pi)
            if shape == "blob":
                mask = self.blob_mask((h / 2.0), (w / 2.0), theta, amp=(self.blob_amp),
                  n_wave=(self.blob_nwave))
            elif shape == "rect":
                mask = np.zeros((max(h, 1), max(w, 1)), dtype=np.float64)
            else:
                mask = self.oval_mask(h / 2.0, w / 2.0, theta)
            (mh, mw) = mask.shape
            if mw < img.size()[2] and mh < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - mh)
                y1 = random.randint(0, img.size()[2] - mw)
                mask_t = torch.from_numpy(mask).to(dtype=(img.dtype))
                inv_t = 1.0 - mask_t
                c = img.size()[0] if img.size()[0] == 3 else 1
                if mode == "patch":
                    sx = random.randint(0, img.size()[1] - mh)
                    sy = random.randint(0, img.size()[2] - mw)
                    fill = img[:, sx:sx + mh, sy:sy + mw].clone()
                elif mode == "noise":
                    noise = torch.empty((img.size()[0], mh, mw), dtype=(img.dtype))
                    noise.normal_(0.5, 0.25).clamp_(0.0, 1.0)
                for ch in range(c):
                    region = img[ch, x1:x1 + mh, y1:y1 + mw]
                    if mode == "patch":
                        img[ch, x1:x1 + mh, y1:y1 + mw] = region * mask_t + fill[ch] * inv_t
                    elif mode == "mean":
                        val = self.mean[ch] if c == 3 else self.mean[0]
                        img[ch, x1:x1 + mh, y1:y1 + mw] = region * mask_t + val * inv_t
                    elif mode == "noise":
                        img[ch, x1:x1 + mh, y1:y1 + mw] = region * mask_t + noise[ch] * inv_t
                    else:
                        img[ch, x1:x1 + mh, y1:y1 + mw] = region * mask_t

                return None

    def __call__(self, img):
        if random.uniform(0, 1) >= self.probability:
            return img
        count = random.randint(self.min_count, self.max_count)
        for _ in range(count):
            self._erase_once(img)

        return img
