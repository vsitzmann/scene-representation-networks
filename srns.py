import torch
import torch.nn as nn
import numpy as np

import torchvision
import util

import skimage.measure
from torch.nn import functional as F

from pytorch_prototyping import pytorch_prototyping
import custom_layers
import geometry
import hyperlayers


class SRNsModel(nn.Module):
    def __init__(self,
                 num_instances,
                 latent_dim,
                 tracing_steps,
                 has_params=False,
                 fit_single_srn=False,
                 use_unet_renderer=False,
                 freeze_networks=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.has_params = has_params

        self.num_hidden_units_phi = 256
        self.phi_layers = 4  # includes the in and out layers
        self.rendering_layers = 5  # includes the in and out layers
        self.sphere_trace_steps = tracing_steps
        self.freeze_networks = freeze_networks
        self.fit_single_srn = fit_single_srn

        if self.fit_single_srn:  # Fit a single scene with a single SRN (no hypernetworks)
            self.phi = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                   num_hidden_layers=self.phi_layers - 2,
                                                   in_features=3,
                                                   out_features=self.num_hidden_units_phi)
        else:
            # Auto-decoder: each scene instance gets its own code vector z
            self.latent_codes = nn.Embedding(num_instances, latent_dim).cuda()
            nn.init.normal_(self.latent_codes.weight, mean=0, std=0.01)

            self.hyper_phi = hyperlayers.HyperFC(hyper_in_ch=self.latent_dim,
                                                 hyper_num_hidden_layers=1,
                                                 hyper_hidden_ch=self.latent_dim,
                                                 hidden_ch=self.num_hidden_units_phi,
                                                 num_hidden_layers=self.phi_layers - 2,
                                                 in_ch=3,
                                                 out_ch=self.num_hidden_units_phi)

        self.ray_marcher = custom_layers.Raymarcher(num_feature_channels=self.num_hidden_units_phi,
                                                    raymarch_steps=self.sphere_trace_steps)

        if use_unet_renderer:
            self.pixel_generator = custom_layers.DeepvoxelsRenderer(nf0=32, in_channels=self.num_hidden_units_phi,
                                                                    input_resolution=128, img_sidelength=128)
        else:
            self.pixel_generator = pytorch_prototyping.FCBlock(hidden_ch=self.num_hidden_units_phi,
                                                               num_hidden_layers=self.rendering_layers - 1,
                                                               in_features=self.num_hidden_units_phi,
                                                               out_features=3,
                                                               outermost_linear=True)

        if self.freeze_networks:
            all_network_params = (list(self.pixel_generator.parameters())
                                  + list(self.ray_marcher.parameters())
                                  + list(self.hyper_phi.parameters()))
            for param in all_network_params:
                param.requires_grad = False

        # Losses
        self.l2_loss = nn.MSELoss(reduction="mean")

        # List of logs
        self.logs = list()

        print(self)
        print("Number of parameters:")
        util.print_network(self)

    def get_regularization_loss(self, prediction, ground_truth):
        """Computes regularization loss on final depth map (L_{depth} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: Regularization loss on final depth map.
        """
        _, depth = prediction

        neg_penalty = (torch.min(depth, torch.zeros_like(depth)) ** 2)
        return torch.mean(neg_penalty) * 10000

    def get_image_loss(self, prediction, ground_truth):
        """Computes loss on predicted image (L_{img} in eq. 6 in paper)

        :param prediction (tuple): Output of forward pass.
        :param ground_truth: Ground-truth (unused).
        :return: image reconstruction loss.
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        loss = self.l2_loss(pred_imgs, trgt_imgs)
        return loss

    def get_latent_loss(self):
        """Computes loss on latent code vectors (L_{latent} in eq. 6 in paper)
        :return: Latent loss.
        """
        if self.fit_single_srn:
            self.latent_reg_loss = 0
        else:
            self.latent_reg_loss = torch.mean(self.z ** 2)

        return self.latent_reg_loss

    def get_psnr(self, prediction, ground_truth):
        """Compute PSNR of model image predictions.

        :param prediction: Return value of forward pass.
        :param ground_truth: Ground truth.
        :return: (psnr, ssim): tuple of floats
        """
        pred_imgs, _ = prediction
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()
        batch_size = pred_imgs.shape[0]

        if not isinstance(pred_imgs, np.ndarray):
            pred_imgs = util.lin2img(pred_imgs).detach().cpu().numpy()

        if not isinstance(trgt_imgs, np.ndarray):
            trgt_imgs = util.lin2img(trgt_imgs).detach().cpu().numpy()

        psnrs, ssims = list(), list()
        for i in range(batch_size):
            p = pred_imgs[i].squeeze().transpose(1, 2, 0)
            trgt = trgt_imgs[i].squeeze().transpose(1, 2, 0)

            p = (p / 2.) + 0.5
            p = np.clip(p, a_min=0., a_max=1.)

            trgt = (trgt / 2.) + 0.5

            ssim = skimage.measure.compare_ssim(p, trgt, multichannel=True, data_range=1)
            psnr = skimage.measure.compare_psnr(p, trgt, data_range=1)

            psnrs.append(psnr)
            ssims.append(ssim)

        return psnrs, ssims

    def get_comparisons(self, model_input, prediction, ground_truth=None):
        predictions, depth_maps = prediction

        batch_size = predictions.shape[0]

        # Parse model input.
        intrinsics = model_input["intrinsics"].cuda()
        uv = model_input["uv"].cuda().float()

        x_cam = uv[:, :, 0].view(batch_size, -1)
        y_cam = uv[:, :, 1].view(batch_size, -1)
        z_cam = depth_maps.view(batch_size, -1)

        normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
        normals = F.pad(normals, pad=(1, 1, 1, 1), mode="constant", value=1.)

        predictions = util.lin2img(predictions)

        if ground_truth is not None:
            trgt_imgs = ground_truth["rgb"]
            trgt_imgs = util.lin2img(trgt_imgs)

            return torch.cat((normals.cpu(), predictions.cpu(), trgt_imgs.cpu()), dim=3).numpy()
        else:
            return torch.cat((normals.cpu(), predictions.cpu()), dim=3).numpy()

    def get_output_img(self, prediction):
        pred_imgs, _ = prediction
        return util.lin2img(pred_imgs)

    def write_updates(self, writer, predictions, ground_truth, iter, prefix=""):
        """Writes tensorboard summaries using tensorboardx api.

        :param writer: tensorboardx writer object.
        :param predictions: Output of forward pass.
        :param ground_truth: Ground truth.
        :param iter: Iteration number.
        :param prefix: Every summary will be prefixed with this string.
        """
        predictions, depth_maps = predictions
        trgt_imgs = ground_truth['rgb']

        trgt_imgs = trgt_imgs.cuda()

        batch_size, num_samples, _ = predictions.shape

        # Module"s own log
        for type, name, content, every_n in self.logs:
            name = prefix + name

            if not iter % every_n:
                if type == "image":
                    writer.add_image(name, content.detach().cpu().numpy(), iter)
                    writer.add_scalar(name + "_min", content.min(), iter)
                    writer.add_scalar(name + "_max", content.max(), iter)
                elif type == "figure":
                    writer.add_figure(name, content, iter, close=True)
                elif type == "histogram":
                    writer.add_histogram(name, content.detach().cpu().numpy(), iter)
                elif type == "scalar":
                    writer.add_scalar(name, content.detach().cpu().numpy(), iter)
                elif type == "embedding":
                    writer.add_embedding(mat=content, global_step=iter)

        if not iter % 100:
            output_vs_gt = torch.cat((predictions, trgt_imgs), dim=0)
            output_vs_gt = util.lin2img(output_vs_gt)
            writer.add_image(prefix + "Output_vs_gt",
                             torchvision.utils.make_grid(output_vs_gt,
                                                         scale_each=False,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

            rgb_loss = ((predictions.float().cuda() - trgt_imgs.float().cuda()) ** 2).mean(dim=2, keepdim=True)
            rgb_loss = util.lin2img(rgb_loss)

            fig = util.show_images([rgb_loss[i].detach().cpu().numpy().squeeze()
                                    for i in range(batch_size)])
            writer.add_figure(prefix + "rgb_error_fig",
                              fig,
                              iter,
                              close=True)

            depth_maps_plot = util.lin2img(depth_maps)
            writer.add_image(prefix + "pred_depth",
                             torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                                         scale_each=True,
                                                         normalize=True).cpu().detach().numpy(),
                             iter)

        writer.add_scalar(prefix + "out_min", predictions.min(), iter)
        writer.add_scalar(prefix + "out_max", predictions.max(), iter)

        writer.add_scalar(prefix + "trgt_min", trgt_imgs.min(), iter)
        writer.add_scalar(prefix + "trgt_max", trgt_imgs.max(), iter)

        if iter:
            writer.add_scalar(prefix + "latent_reg_loss", self.latent_reg_loss, iter)

    def forward(self, input, z=None):
        self.logs = list() # log saves tensors that"ll receive summaries when model"s write_updates function is called

        # Parse model input.
        instance_idcs = input["instance_idx"].long().cuda()
        pose = input["pose"].cuda()
        intrinsics = input["intrinsics"].cuda()
        uv = input["uv"].cuda().float()

        if self.fit_single_srn:
            phi = self.phi
        else:
            if self.has_params: # If each instance has a latent parameter vector, we"ll use that one.
                if z is None:
                    self.z = input["param"].cuda()
                else:
                    self.z = z
            else: # Else, we"ll use the embedding.
                self.z = self.latent_codes(instance_idcs)

            phi = self.hyper_phi(self.z) # Forward pass through hypernetwork yields a (callable) SRN.

        # Raymarch SRN phi along rays defined by camera pose, intrinsics and uv coordinates.
        points_xyz, depth_maps, log = self.ray_marcher(cam2world=pose,
                                                       intrinsics=intrinsics,
                                                       uv=uv,
                                                       phi=phi)
        self.logs.extend(log)

        # Sapmle phi a last time at the final ray-marched world coordinates.
        v = phi(points_xyz)

        # Translate features at ray-marched world coordinates to RGB colors.
        novel_views = self.pixel_generator(v)

        # Calculate normal map
        with torch.no_grad():
            batch_size = uv.shape[0]
            x_cam = uv[:, :, 0].view(batch_size, -1)
            y_cam = uv[:, :, 1].view(batch_size, -1)
            z_cam = depth_maps.view(batch_size, -1)

            normals = geometry.compute_normal_map(x_img=x_cam, y_img=y_cam, z=z_cam, intrinsics=intrinsics)
            self.logs.append(("image", "normals",
                              torchvision.utils.make_grid(normals, scale_each=True, normalize=True), 100))

        if not self.fit_single_srn:
            self.logs.append(("embedding", "", self.latent_codes.weight, 500))
            self.logs.append(("scalar", "embed_min", self.z.min(), 1))
            self.logs.append(("scalar", "embed_max", self.z.max(), 1))

        return novel_views, depth_maps
