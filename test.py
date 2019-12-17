import configargparse
import os, time, datetime

import torch
import numpy as np

import dataio
from torch.utils.data import DataLoader
from srns import *
import util

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

# Note: in contrast to training, no multi-resolution!
p.add_argument('--img_sidelength', type=int, default=128, required=False,
               help='Sidelength of test images.')

p.add_argument('--data_root', required=True, help='Path to directory with training data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')
p.add_argument('--batch_size', type=int, default=32, help='Batch size.')
p.add_argument('--preload', action='store_true', default=False, help='Whether to preload data to RAM.')

p.add_argument('--max_num_instances', type=int, default=-1,
               help='If \'data_root\' has more instances, only the first max_num_instances are used')
p.add_argument('--specific_observation_idcs', type=str, default=None,
               help='Only pick a subset of specific observations for each instance.')
p.add_argument('--has_params', action='store_true', default=False,
               help='Whether each object instance already comes with its own parameter vector.')

p.add_argument('--save_out_first_n',type=int, default=250, help='Only saves images of first n object instances.')
p.add_argument('--checkpoint_path', default=None, help='Path to trained model.')

# Model options
p.add_argument('--num_instances', type=int, required=True,
               help='The number of object instances that the model was trained with.')
p.add_argument('--tracing_steps', type=int, default=10, help='Number of steps of intersection tester.')
p.add_argument('--fit_single_srn', action='store_true', required=False,
               help='Only fit a single SRN for a single scene (not a class of SRNs) --> no hypernetwork')
p.add_argument('--use_unet_renderer', action='store_true',
               help='Whether to use a DeepVoxels-style unet as rendering network or a per-pixel 1x1 convnet')
p.add_argument('--embedding_size', type=int, default=256,
               help='Dimensionality of latent embedding.')

opt = p.parse_args()

device = torch.device('cuda')


def test():
    if opt.specific_observation_idcs is not None:
        specific_observation_idcs = list(map(int, opt.specific_observation_idcs.split(',')))
    else:
        specific_observation_idcs = None

    dataset = dataio.SceneClassDataset(root_dir=opt.data_root,
                                       max_num_instances=opt.max_num_instances,
                                       specific_observation_idcs=specific_observation_idcs,
                                       max_observations_per_instance=-1,
                                       samples_per_instance=1,
                                       img_sidelength=opt.img_sidelength)
    dataset = DataLoader(dataset,
                         collate_fn=dataset.collate_fn,
                         batch_size=1,
                         shuffle=False,
                         drop_last=False)

    model = SRNsModel(num_instances=opt.num_instances,
                      latent_dim=opt.embedding_size,
                      has_params=opt.has_params,
                      fit_single_srn=opt.fit_single_srn,
                      use_unet_renderer=opt.use_unet_renderer,
                      tracing_steps=opt.tracing_steps)

    assert (opt.checkpoint_path is not None), "Have to pass checkpoint!"

    print("Loading model from %s" % opt.checkpoint_path)
    util.custom_load(model, path=opt.checkpoint_path, discriminator=None,
                     overwrite_embeddings=False)

    model.eval()
    model.cuda()

    # directory structure: month_day/
    renderings_dir = os.path.join(opt.logging_root, 'renderings')
    gt_comparison_dir = os.path.join(opt.logging_root, 'gt_comparisons')
    util.cond_mkdir(opt.logging_root)
    util.cond_mkdir(gt_comparison_dir)
    util.cond_mkdir(renderings_dir)

    # Save command-line parameters to log directory.
    with open(os.path.join(opt.logging_root, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    print('Beginning evaluation...')
    with torch.no_grad():
        instance_idx = 0
        idx = 0
        psnrs, ssims = list(), list()
        for model_input, ground_truth in dataset:
            model_outputs = model(model_input)
            psnr, ssim = model.get_psnr(model_outputs, ground_truth)

            psnrs.extend(psnr)
            ssims.extend(ssim)

            instance_idcs = model_input['instance_idx']
            print("Object instance %d. Running mean PSNR %0.6f SSIM %0.6f" %
                  (instance_idcs[-1], np.mean(psnrs), np.mean(ssims)))

            if instance_idx < opt.save_out_first_n:
                output_imgs = model.get_output_img(model_outputs).cpu().numpy()
                comparisons = model.get_comparisons(model_input,
                                                    model_outputs,
                                                    ground_truth)
                for i in range(len(output_imgs)):
                    prev_instance_idx = instance_idx
                    instance_idx = instance_idcs[i]

                    if prev_instance_idx != instance_idx:
                        idx = 0

                    img_only_path = os.path.join(renderings_dir, "%06d" % instance_idx)
                    comp_path = os.path.join(gt_comparison_dir, "%06d" % instance_idx)

                    util.cond_mkdir(img_only_path)
                    util.cond_mkdir(comp_path)

                    pred = util.convert_image(output_imgs[i].squeeze())
                    comp = util.convert_image(comparisons[i].squeeze())

                    util.write_img(pred, os.path.join(img_only_path, "%06d.png" % idx))
                    util.write_img(comp, os.path.join(comp_path, "%06d.png" % idx))

                    idx += 1

    with open(os.path.join(opt.logging_root, "results.txt"), "w") as out_file:
        out_file.write("%0.6f, %0.6f" % (np.mean(psnrs), np.mean(ssims)))

    print("Final mean PSNR %0.6f SSIM %0.6f" % (np.mean(psnrs), np.mean(ssims)))


def main():
    test()


if __name__ == '__main__':
    main()
