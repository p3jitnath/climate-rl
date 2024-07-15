import numpy as np
import torch
from torch.distributions.kl import kl_divergence

# def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
#     torch.nn.init.orthogonal_(layer.weight, std)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer


def fisher_vector_product(actor, obs, x, cg_damping=0.1):
    x.detach()
    pi_new = actor(obs)
    with torch.no_grad():
        pi_old = actor(obs)
    kl = kl_divergence(pi_old, pi_new).mean()
    kl_grads = torch.autograd.grad(
        kl, tuple(actor.parameters()), create_graph=True
    )
    flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])
    kl_grad_p = (flat_kl_grad * x).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, tuple(actor.parameters()))
    flat_kl_hessian_p = torch.cat(
        [grad.contiguous().view(-1) for grad in kl_hessian_p]
    )

    # tricks to stabilize
    # see https://www2.maths.lth.se/matematiklth/vision/publdb/reports/pdf/byrod-eccv-10.pdf
    return flat_kl_hessian_p + cg_damping * x


# Refer to https://en.wikipedia.org/wiki/Conjugate_gradient_method for more details
def conjugate_gradient(actor, obs, b, cg_iters, cg_residual_tol=1e-10):
    """
    Given a linear system Ax = b and an initial guess x0=0, the conjugate gradient method solves the problem
    Ax = b for x without computing A explicitly. Instead, only the computation of the matrix-vector product Ax is needed.
    In TRPO, A is the Fisher information matrix F (the second derivates of KL divergence) and b is the gradient of the loss function.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)
    for _ in range(cg_iters):
        _Ax = fisher_vector_product(actor, obs, p)
        alpha = r_dot_r / torch.dot(p, _Ax)
        x += alpha * p
        r -= alpha * _Ax
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / r_dot_r
        p = r + betta * p
        r_dot_r = new_rdotr
        if r_dot_r < cg_residual_tol:
            break
    return x


def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length
