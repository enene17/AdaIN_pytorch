import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import torch
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def random_crop(img):

    w, h = img.size
    if w > h:
        r = w/h*512
        img = img.resize((int(r), 512))
        w_in = np.random.randint(0, int(r)-256)
        h_in = np.random.randint(0, 256)

        img = img.crop((w_in, h_in, w_in+256, h_in+256))

    else:
        r = h/w*512
        img = img.resize((512, int(r)))
        w_in = np.random.randint(0, 256)
        h_in = np.random.randint(0, int(r)-256)

        img = img.crop((w_in, h_in, w_in+256, h_in+256))

    return img


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0


class DS(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                self.samples.append(path)

        if len(self.samples) == 0:
            raise RuntimeError("Found 0 files in subfolders of: " + root)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_path = self.samples[index]
        sample = Image.open(sample_path).convert('RGB')

        sample = random_crop(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
