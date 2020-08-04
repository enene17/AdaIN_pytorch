import argparse
import os

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import DS, InfiniteSampler
from model import StyleTransferNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--c_root', type=str, default='./content')
parser.add_argument('--s_root', type=str, default='./style')
parser.add_argument('--save_dir', type=str, default='./result')
parser.add_argument('--lr', type=float, default=1e-4, help="adam: learning rate")
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument("--c_weight", type=float, default=10.0)
parser.add_argument("--s_weight", type=float, default=1.0)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--resume', type=int)
args = parser.parse_args()


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

writer = SummaryWriter()

train_tf = transforms.Compose([
    transforms.ToTensor(),
])

c_train_set = DS(args.c_root, train_tf)
s_train_set = DS(args.s_root, train_tf)

c_iterator_train = iter(data.DataLoader(
    c_train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(c_train_set)),
    num_workers=args.n_threads))
print(len(c_train_set))

s_iterator_train = iter(data.DataLoader(
    s_train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(s_train_set)),
    num_workers=args.n_threads))
print(len(s_train_set))

st_model = StyleTransferNetwork().to(device)

content_loss = nn.MSELoss().to(device)

start_iter = 0
st_optimizer = torch.optim.Adam(
    st_model.parameters(), args.lr)

if args.resume:
    st_checkpoint = torch.load("{}/ckpt/ST_{}.pth".format(args.save_dir, args.resume), map_location=device)
    st_model.load_state_dict(st_checkpoint)
    print('Models restored')


for i in tqdm(range(start_iter, args.max_iter)):

    #adjust the learning rate
    lr = args.lr / (1.0 + args.lr_decay * i)
    for param_group in st_optimizer.param_groups:
        param_group['lr'] = lr

    c_img = next(c_iterator_train).to(device)
    s_img = next(s_iterator_train).to(device)

    c_img = 2. * c_img - 1. # [0,1] -> [-1,1]
    s_img = 2. * s_img - 1. # [0,1] -> [-1,1]

    a_feature, r_feature, result, s_loss = st_model(c_img, s_img, 1, args.batch_size)

    c_loss = content_loss(a_feature, r_feature)

    total_loss = args.c_weight * c_loss + args.s_weight * s_loss

    st_optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    st_optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        torch.save(st_model.state_dict(), "{}/ckpt/ST_{}.pth".format(args.save_dir, i + 1))

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('content_loss', c_loss.item(), i + 1)
        writer.add_scalar('style_loss', s_loss.item(), i + 1)
        writer.add_scalar('total_loss', total_loss.item(), i + 1)

    def denorm(x):
        out = (x + 1) / 2 # [-1,1] -> [0,1]
        return out.clamp_(0, 1)
    if (i + 1) % args.vis_interval == 0:
        ims = torch.cat([c_img, s_img, result], dim=3)
        writer.add_images('content_style_result', denorm(ims), i + 1)

writer.close()
