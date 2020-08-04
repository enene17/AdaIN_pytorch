import argparse

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
import glob
import os

from model import StyleTransferNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--c_image', type=str,
                    help='content_folder_path.')
parser.add_argument('--s_image', type=str,
                    help='style_image_path.')                    
parser.add_argument('--output', type=str,
                    help='output_folder_path.')
parser.add_argument('--blend_value', type=str, default=1.0,
                    help='content-style trade-off')
parser.add_argument('--checkpoint', type=str,
                    help='The filename of pickle checkpoint.')


def norm(x):
    return 2. * x - 1.  # [0,1] -> [-1,1]

def denorm(x):
    out = (x + 1) / 2  # [-1,1] -> [0,1]
    return out.clamp_(0, 1)


if __name__ == "__main__":
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    st_model = StyleTransferNetwork().to(device)
    st_checkpoint = torch.load(args.checkpoint, map_location=device)
    st_model.load_state_dict(st_checkpoint)
    st_model.eval()

    to_tensor = transforms.ToTensor()

    grid = 256

    s_img = Image.open(args.s_image).convert('RGB')
    s_img = to_tensor(s_img)
    _, s_h, s_w = s_img.shape
    s_img = s_img[:, :s_h//grid*grid, :s_w//grid*grid]
    s_img = s_img.unsqueeze_(0)  # CHW -> BCHW
    s_img = norm(s_img)  # [0,1] -> [-1,1]
    s_img = s_img.to(device)

    files = glob.glob("%s/*" % args.c_image)

    for fname in files:
        name = os.path.split(fname)[1]

        c_img = Image.open(fname).convert('RGB')
        c_img = to_tensor(c_img)
        _, c_h, c_w = c_img.shape
        c_img = c_img[:, :c_h//grid*grid, :c_w//grid*grid]
        c_img = c_img.unsqueeze_(0)  # CHW -> BCHW  
        c_img = norm(c_img)  # [0,1] -> [-1,1]
        c_img = c_img.to(device)
        
        import time
        start_time = time.time()
        _, _, result, _ = st_model(c_img, s_img, args.blend_value, 1)
        print("Done in %.3f seconds!" % (time.time() - start_time))

        save_image(denorm(result), "%s/%s" % (args.output, name))
