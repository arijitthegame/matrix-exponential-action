# pylint: skip-file

import numpy as np
from tqdm import trange
import torch

from lanczos import lanczos
from conjugate_gradient import conjugate_grad_torch
from sdd_solver import *

# TODO: ADD SPARSE FROM HAN
def compute_lanczos_matrix_exp(
    A, v, k, use_reorthogonalization=False, return_exp=False
):
    """
    Compute the action of matrix exponential on a vector v using the Lanczos algorithm.
    Can also optionally return the approximate exponential matrix too
    This is figure 4 in https://arxiv.org/abs/1111.1491
    v is assumed to be of shape B x N

    #TODO: ADD NECESSARY CHECKS FOR SHAPES
    """

    if len(v.shape) == 1:
        v = v.unsqueeze(0)

    # compute Q, T via Lanczos, T is the tridiagonal matrix with shape k x k and Q is of shape n x k
    T, Q = lanczos(
        A, num_eig_vec=k, mask=None, use_reorthogonalization=use_reorthogonalization
    )

    D, P = torch.linalg.eigh(T)
    exp_T = torch.bmm(torch.bmm(P, torch.diag_embed(torch.exp(D))), P.transpose(1, 2))

    # compute the action

    exp_A = torch.bmm(torch.bmm(Q, exp_T), Q.transpose(1, 2))

    w = torch.einsum("ijk, ik -> ij", exp_A, v)

    if return_exp is False:
        return w
    else:
        return w, exp_A


def compute_exprational_matrix_exp(
    A, v, k, return_exp=False, method_type="torchsolve", tolerance=1e-6, eps=1e-15
):
    """
    This is the algorithm as described in figure 5 of https://arxiv.org/abs/1111.1491
    SDD solver in the paper is not implemented however we can use CG or Gauss-Siedel or Jacobi methods or a torch solver to solve linear equations.
    So far torch solver is very stable and produces good results.
    Args: A matrix A of shape b x n x n,
        v vector of shape B x n,
        k number of eigenvectors to compute,
        return_exp=True if you want to return the approximate exponential matrix
        method_type = 'cg' or 'jacobi' or 'gauss_seidel' or 'torchsolve'

    A vector u such that ||exp(-A)v - u|| ≤ ε
    """
    Alpha = torch.empty(k + 1, k + 1)
    V = torch.empty(A.shape[0], k + 1)
    V[:, 0] = v.squeeze()
    for i in range(k):
        W = torch.empty(A.shape[0], 1)
        if method_type == "torchsolve":
            w = torch.linalg.solve(A, V[:, i].reshape(-1, 1))
            W = w
        elif method_type == "cg":
            w = conjugate_grad_torch(A, V[:, i].reshape(-1, 1)).reshape(-1, 1)
            W = w
        else:
            raise NotImplementedError("Other methods are not yet implemented")

        temp_vec = torch.zeros(W.shape[0], 1)
        for j in range(i):
            Alpha[j][i] = torch.dot(V[:, j], W.squeeze())
            temp_vec += Alpha[j][i] * V[:, j].reshape(-1, 1)

            # orthogonalize
        w = w - temp_vec
        Alpha[i + 1][i] = torch.norm(W.squeeze()) + eps
        V[:, i + 1] = W.squeeze() / Alpha[i + 1][i]
        for j in range(i + 2, k + 1):
            A[j][i] = 0

    T_hat = 0.5 * (Alpha + Alpha.t())
    B = torch.linalg.matrix_exp(
        k * (torch.eye(T_hat.shape[0]) - torch.linalg.inv(T_hat))
    )

    approx_vec = V @ B[:, 0]
    return approx_vec
