import os, struct, math
import numpy as np
import torch
from glob import glob

import cv2
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_latest_file(root_dir):
    """Returns path to latest file in a directory."""
    list_of_files = glob.glob(os.path.join(root_dir, '*'))
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def parse_comma_separated_integers(string):
    return list(map(int, string.split(',')))


def convert_image(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img.cpu().detach().numpy())

    img = img.squeeze()
    img = img.transpose(1,2,0)
    img += 1.
    img /= 2.
    img *= 2**8 - 1
    img = img.round().clip(0, 2**8-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def write_img(img, path):
    cv2.imwrite(path, img.astype(np.uint8))


def in_out_to_param_count(in_out_tuples):
    return np.sum([np.prod(in_out) + in_out[-1] for in_out in in_out_tuples])

def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx/width * trgt_sidelength
        cy = cy/height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses

def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

def num_divisible_by_2(number):
    i = 0
    while not number%2:
        number = number // 2
        i += 1

    return i

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pose(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def write_image(writer, name, img, iter):
    writer.add_image(name, normalize(img.permute([0,3,1,2])), iter)


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d"%params)


def custom_load(model, path, discriminator=None, overwrite_embeddings=False, overwrite_renderer=False, optimizer=None):
    if os.path.isdir(path):
        checkpoint_path = sorted(glob(os.path.join(path, "*.pth")))[-1]
    else:
        checkpoint_path = path

    whole_dict = torch.load(checkpoint_path)

    if overwrite_embeddings:
        del whole_dict['model']['latent_codes.weight']

    if overwrite_renderer:
        keys_to_remove = [key for key in whole_dict['model'].keys() if 'rendering_net' in key]
        for key in keys_to_remove:
            print(key)
            whole_dict['model'].pop(key, None)

    state = model.state_dict()
    state.update(whole_dict['model'])
    model.load_state_dict(state)

    if discriminator:
        discriminator.load_state_dict(whole_dict['discriminator'])

    if optimizer:
        optimizer.load_state_dict(whole_dict['optimizer'])


def custom_save(model, path, discriminator=None, optimizer=None):
    whole_dict = {'model':model.state_dict()}
    if discriminator:
        whole_dict.update({'discriminator':discriminator.state_dict()})
    if optimizer:
        whole_dict.update({'optimizer':optimizer.state_dict()})

    torch.save(whole_dict, path)


def show_images(images, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    cols = np.ceil(np.sqrt(len(images))).astype(int)

    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        im = a.imshow(image)

        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

        if len(images) < 10:
            divider = make_axes_locatable(a)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')


    plt.tight_layout()

    # fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    return fig

