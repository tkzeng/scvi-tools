from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl

from scvi.external.velovi._constants import VELOVI_REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, FCLayers

import wandb


def compute_penalty(list_, p=2, target=0.):
    penalty = 0
    for m in list_:
        penalty += torch.norm(m - target, p=p) ** p
    return penalty

def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)

def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft

    return y

class GumbelAdjacency(torch.nn.Module):
    """
    Random matrix M used for the mask. Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """
    def __init__(self, num_vars):
        super(GumbelAdjacency, self).__init__()
        self.num_vars = num_vars
        self.log_alpha = torch.nn.Parameter(torch.zeros((num_vars, num_vars)))
        self.uniform = torch.distributions.uniform.Uniform(0, 1)
        self.reset_parameters()

    def forward(self, bs, tau=1, drawhard=True):
        adj = gumbel_sigmoid(self.log_alpha, self.uniform, bs, tau=tau, hard=drawhard)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)

class DecoderVELOVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space).
    n_output
        The dimensionality of the output (data space).
    n_cat_list
        A list containing the number of categories or each category of interest. Each category will
        be included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers.
    n_hidden
        The number of nodes per hidden layer.
    dropout_rate
        Dropout rate to apply to each of the hidden layers.
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    linear_decoder
        Whether to use linear decoder for time.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        linear_decoder: bool = False,
        **kwargs,
    ):
        # print("n_input", n_input)
        # print("n_output", n_output)

        super().__init__()
        self.n_ouput = n_output
        self.linear_decoder = linear_decoder
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden if not linear_decoder else n_output,
            n_cat_list=n_cat_list,
            n_layers=n_layers if not linear_decoder else 1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm if not linear_decoder else False,
            use_activation=not linear_decoder,
            bias=not linear_decoder,
            **kwargs,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # categorical pi
        # 4 states
        self.px_pi_decoder = nn.Linear(n_hidden, 4 * n_output)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(n_output))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(n_output))

        # initialize current adjacency matrix
        self.num_vars = n_output # number of genes
        self.adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars)
        self.gumbel_adjacency = GumbelAdjacency(self.num_vars)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0
        self.num_layers = 0 # number of hidden layers?
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        self.hid_dim = 8

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer
            if i == 0:
                in_dim = self.num_vars

            # last layer
            if i == self.num_layers:
                out_dim = 1 #self.num_params

            # for perfect interv, generate only one MLP per conditional
            self.weights.append(nn.Parameter(torch.zeros(self.num_vars, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(self.num_vars, out_dim)))
            self.numel_weights += self.num_vars * out_dim * in_dim

    def forward(self,
                z: torch.Tensor,
                x: torch.Tensor,
                latent_dim: int = None):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``.
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :class:`~torch.Tensor`
            parameters for the ZINB distribution of expression.

        """
        z_in = z
        if latent_dim is not None:
            mask = torch.zeros_like(z)
            mask[..., latent_dim] = 1
            z_in = z * mask
        # The decoder returns values for the parameters of the ZINB distribution
        rho_first = self.rho_first_decoder(z_in)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp() + self.linear_scaling_tau_intercept
            )

        # cells by genes by 4
        pi_first = self.pi_first_decoder(z)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 4))
        )

        ##### graph params #####
        # x = spliced counts for now as input
        bs = x.size(0)
        num_zero_weights = 0
        # weights torch.Size([95, 1, 95])
        # M torch.Size([256, 95, 95])
        # adj torch.Size([1, 95, 95])
        # x torch.Size([256, 95])
        # biases torch.Size([95, 1])

        for layer in range(self.num_layers + 1):
            # First layer, apply the mask
            if layer == 0:
                # sample the matrix M that will be applied as a mask at the MLP input
                M = self.gumbel_adjacency(bs)
                adj = self.adjacency.unsqueeze(0)

                # print("M",M)
                # print("adj",adj)

                # if not self.intervention:
                # print("weights", self.weights[layer].shape)
                # print("M", M.shape)
                # print("adj", adj.shape)
                # print("x", x.shape)
                #print(x)
                # print("biases", self.biases[layer].shape)
                alpha = torch.einsum("tij,bjt,ljt,bj->bti", self.weights[layer], M, adj, x) + self.biases[layer]
                #print("alpha...", alpha.shape)
                # elif self.intervention_type == "perfect" and self.intervention_knowledge == "known":
                #     # the mask is not applied here, it is applied in the loss term
                #     alpha = torch.einsum("tij,bjt,ljt,bj->bti", self.weights[layer], M, adj, x) + self.biases[layer]

            # 2nd layer and more
            else:
                print("alpha", alpha.shape)
                alpha = torch.einsum("tij,btj->bti", self.weights[layer], alpha) + self.biases[layer]
                print("alpha", alpha.shape)

            # count number of zeros
            num_zero_weights += self.weights[layer].numel() - self.weights[layer].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                alpha = F.leaky_relu(alpha)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)
        print("zero weights ratio", self.zero_weights_ratio)

        #print("alpha", alpha.shape, alpha)

        alpha = torch.clamp(F.softplus(alpha), 0, 50)
        alpha = torch.squeeze(alpha, dim=2)
        #alpha = torch.unbind(alpha, 1)
        #print("alpha", len(alpha), alpha[0].shape, alpha[1].shape)
        #test = torch.nn.Parameter(0 * torch.ones(self.num_vars))
        #print("test", test.shape)

        ##########

        return px_pi, px_rho, px_tau, alpha


# VAE model
class VELOVAE(BaseModuleClass):
    """Variational auto-encoder model.

    This is an implementation of the veloVI model descibed in :cite:p:`GayosoWeiler23`.

    Parameters
    ----------
    n_input
        Number of input genes.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of the following:

        * ``"normal"`` - Isotropic normal.
        * ``"ln"`` - Logistic normal with normal params N(0, 1).
    use_layer_norm
        Whether to use layer norm in layers.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When ``None``, defaults to :func:`~torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        true_time_switch: np.ndarray | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = False,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_observed_lib_size: bool = True,
        var_activation: callable | None = torch.nn.Softplus(),
        model_steady_states: bool = True,
        gamma_unconstr_init: np.ndarray | None = None,
        alpha_unconstr_init: np.ndarray | None = None,
        alpha_1_unconstr_init: np.ndarray | None = None,
        lambda_alpha_unconstr_init: np.ndarray | None = None,
        switch_spliced: np.ndarray | None = None,
        switch_unspliced: np.ndarray | None = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
        dirichlet_concentration: float = 0.25,
        linear_decoder: bool = False,
        time_dep_transcription_rate: bool = False,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale
        self.dirichlet_concentration = dirichlet_concentration
        self.time_dep_transcription_rate = time_dep_transcription_rate

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if gamma_unconstr_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(torch.from_numpy(gamma_unconstr_init))

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        # transcription
        if alpha_unconstr_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(torch.from_numpy(alpha_unconstr_init))

        # TODO: Add `require_grad`
        if alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_1_unconstr = torch.nn.Parameter(torch.from_numpy(alpha_1_unconstr_init))
        self.alpha_1_unconstr.requires_grad = time_dep_transcription_rate

        if lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.lambda_alpha_unconstr = torch.nn.Parameter(
                torch.from_numpy(lambda_alpha_unconstr_init)
            )
        self.lambda_alpha_unconstr.requires_grad = time_dep_transcription_rate

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
            linear_decoder=linear_decoder,
        )

    def _get_inference_input(self, tensors):
        spliced = tensors[VELOVI_REGISTRY_KEYS.X_KEY]
        unspliced = tensors[VELOVI_REGISTRY_KEYS.U_KEY]

        input_dict = {
            "spliced": spliced,
            "unspliced": unspliced,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        x = tensors[VELOVI_REGISTRY_KEYS.X_KEY]
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]

        input_dict = {
            "z": z,
            "x": x,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = {
            MODULE_KEYS.Z_KEY: z,
            MODULE_KEYS.QZM_KEY: qz_m,
            MODULE_KEYS.QZV_KEY: qz_v,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 50)
        if self.time_dep_transcription_rate:
            alpha_1 = torch.clamp(F.softplus(self.alpha_1_unconstr), 0, 50)
            lambda_alpha = torch.clamp(F.softplus(self.lambda_alpha_unconstr), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha


    def _get_w_adj(self):
        return self.decoder.gumbel_adjacency.get_proba() * self.decoder.adjacency


    @auto_move_data
    def generative(self, z, x, gamma, beta, alpha, alpha_1, lambda_alpha, latent_dim=None):
        """Runs the generative model."""
        # lambda_alpha and alpha_1 are only used with time-dependent transcription rates

        alpha_orig = alpha
        decoder_input_1 = z
        decoder_input_2 = x
        # override alpha here
        px_pi_alpha, px_rho, px_tau, alpha = self.decoder(decoder_input_1,
                                                          decoder_input_2,
                                                          latent_dim=latent_dim)
        #alpha = alpha_orig

        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha,
            alpha_1,
            lambda_alpha,
        )

        return {
            "px_pi": px_pi,
            "px_rho": px_rho,
            "px_tau": px_tau,
            "scale": scale,
            "px_pi_alpha": px_pi_alpha,
            "mixture_dist_u": mixture_dist_u,
            "mixture_dist_s": mixture_dist_s,
            "end_penalty": end_penalty,
        }



    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[VELOVI_REGISTRY_KEYS.X_KEY]
        unspliced = tensors[VELOVI_REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs[MODULE_KEYS.QZM_KEY]
        qz_v = inference_outputs[MODULE_KEYS.QZV_KEY]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
        ).sum(dim=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty

        w_adj = self._get_w_adj()
        print("min", torch.min(w_adj[w_adj>0]), torch.max(w_adj[w_adj>0]))
        reg = torch.abs(w_adj).sum() / w_adj.shape[0] ** 2

        reg1 = compute_penalty([w_adj], p=1)
        reg1 /= w_adj.shape[0]**2

        print("reg", reg1)

        loss = loss + 600*reg1

        loss_recoder = LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

        # wandb.log({"loss": loss})

        return loss_recoder

    @auto_move_data
    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) -> torch.Tensor:
        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        #print(beta.shape, gamma.shape, )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = (alpha_1 / beta).expand(n_cells, self.n_input)
            mean_s_ind_steady = (alpha_1 / gamma).expand(n_cells, self.n_input)
        else:
            mean_u_ind_steady = (alpha / beta).expand(n_cells, self.n_input)
            mean_s_ind_steady = (alpha / gamma).expand(n_cells, self.n_input)
        scale_u = scale[: self.n_input, :].expand(n_cells, self.n_input, 4).sqrt()

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_input :, :].expand(n_cells, self.n_input, 4).sqrt()

        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # unspliced
        mean_u = torch.stack(
            (
                mean_u_ind,
                mean_u_ind_steady,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            dim=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        scale_s = torch.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            dim=2,
        )
        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty

    def _get_induction_unspliced_spliced(
        self, alpha, alpha_1, lambda_alpha, beta, gamma, t, eps=1e-6
    ):
        if self.time_dep_transcription_rate:
            unspliced = alpha_1 / beta * (1 - torch.exp(-beta * t)) - (alpha_1 - alpha) / (
                beta - lambda_alpha
            ) * (torch.exp(-lambda_alpha * t) - torch.exp(-beta * t))

            spliced = (
                alpha_1 / gamma * (1 - torch.exp(-gamma * t))
                + alpha_1 / (gamma - beta + eps) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
                - beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - lambda_alpha + eps)
                * (torch.exp(-lambda_alpha * t) - torch.exp(-gamma * t))
                + beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - beta + eps)
                * (torch.exp(-beta * t) - torch.exp(-gamma * t))
            )
        else:
            # print("alpha", alpha.shape) # len(alpha), alpha[0].shape)
            # print("beta", beta.shape)
            # print("t", t.shape)

            unspliced = (alpha / beta) * (1 - torch.exp(-beta * t))
            spliced = (alpha / gamma) * (1 - torch.exp(-gamma * t)) + (
                alpha / ((gamma - beta) + eps)
            ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))

        # print("uns", unspliced.shape)
        # print("spl", spliced.shape)

        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (beta * u_0 / ((gamma - beta) + eps)) * (
            torch.exp(-gamma * t) - torch.exp(-beta * t)
        )
        return unspliced, spliced

    # def sample(
    #     self,
    # ) -> np.ndarray:
    #     """Not implemented."""
    #     raise NotImplementedError

    @torch.inference_mode()
    def sample(self, z, x, n_samples=1):
        """Sample from the generative model."""
        inference_kwargs = {"n_samples": n_samples}

        with torch.inference_mode():
            inference_outputs, generative_outputs = self.forward(
                z, x,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )

        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        spliced_sample = mixture_dist_s.sample().cpu()
        unspliced_sample = mixture_dist_u.sample().cpu()

        return spliced_sample, unspliced_sample

    @torch.no_grad()
    def get_loadings(self) -> np.ndarray:
        """Extract per-gene weights in the linear decoder."""
        # This is BW, where B is diag(b) batch norm, W is weight matrix
        if self.decoder.linear_decoder is False:
            raise ValueError("Model not trained with linear decoder")
        w = self.decoder.rho_first_decoder.fc_layers[0][0].weight
        if self.use_batch_norm_decoder:
            bn = self.decoder.rho_first_decoder.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = w
        loadings = loadings.detach().cpu().numpy()

        return loadings
