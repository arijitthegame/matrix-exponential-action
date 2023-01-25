# pylint: skip-file
import torch
from lanczos import lanczos
from conjugate_gradient import conjugate_grad_torch
from sdd_solver import *


def is_psd(mat):
    """
    Check if the matrix is positive semi-definite.
    Note in PSD def, we want the matrices are symmetric otherwise it will lead to a lot of issues.
    For example : see https://math.stackexchange.com/questions/83134/does-non-symmetric-positive-definite-matrix-have-positive-eigenvalues
    """
    return bool((mat == mat.T).all() and (torch.linalg.eigvalsh(mat)[0] >= 0))


# TODO: ADD SPARSE FROM HAN
def compute_lanczos_matrix_exp(
    A, v, k, use_reorthogonalization=False, return_exp=False
):
    """
    Compute the action of matrix exponential on a vector v using the Lanczos algorithm.
    Can also optionally return the approximate exponential matrix too
    This is figure 4 in https://arxiv.org/abs/1111.1491
    A is assumed to be of shape B x N x N, a batch of symmetric PSD.
    v is assumed to be of shape B x N
    Compute :
        A vector u that is an approximation to exp(-A)v.
    """

    if len(v.shape) == 1:
        v = v.unsqueeze(0)

    # if len(A.shape) == 3:
    #     for i in range(A.shape[0]):
    #         assert is_psd(A[i]), "All matrices in this batch needs to be PSD"

    # normalize v
    norm_v = torch.linalg.norm(v, dim=1, keepdim=True)
    v = v / norm_v

    # compute Q, T via Lanczos, T is the tridiagonal matrix with shape k x k and Q is of shape n x k
    T, Q = lanczos(
        A, num_eig_vec=k, mask=None, use_reorthogonalization=use_reorthogonalization
    )

    D, P = torch.linalg.eigh(T)
    exp_T = torch.bmm(torch.bmm(P, torch.diag_embed(torch.exp(-D))), P.transpose(1, 2))

    # compute the action

    exp_A = torch.bmm(torch.bmm(Q, exp_T))
    
    # if len(exp_A.squeeze().shape)==2: 
    #     w = exp_A[:,0]*norm_v
    # else :
    #     w = exp_A[: :,0:,0] * norm_v

    w = torch.einsum("ijk, ik -> ij", exp_A, v) * norm_v

    if return_exp is False:
        return w
    else:
        # raise ValueError("Fast exponential does not allow for the materialization of exp")
        return w, exp_A


def compute_exprational_matrix_exp(
    A, v, k, return_exp=False, method_type="torchsolve", tolerance=1e-6, eps=1e-15
):
    """
    This is the algorithm as described in figure 5 of https://arxiv.org/abs/1111.1491
    SDD solver in the paper is not implemented however we can use CG or Gauss-Siedel or Jacobi methods or a torch solver to solve linear equations.
    So far torch solver is very stable and produces good results.
    Args: A PSD matrix A of shape n x n,
        v vector of shape n x 1,
        k number of eigenvectors to compute,
        method_type = 'cg' or 'jacobi' or 'gauss_seidel' or 'torchsolve'

    A vector u such that ||exp(-A)v - u|| ≤ ε
    For this algorithm , v needs to be an unit vector otherwise normalize v
    """

    Alpha = torch.ones(k + 1, k + 1)
    V = torch.zeros(A.shape[0], k + 1)
    norm_v = torch.linalg.norm(v)
    v = v/norm_v
    V[:, 0] = v.squeeze()
    for i in range(k):
        W = torch.empty(A.shape[0], 1)
        if method_type == "torchsolve":
            w = torch.linalg.solve(
                (torch.eye(A.shape[0]) + A / k), V[:, i].reshape(-1, 1)
            )
            W = w
        elif method_type == "cg":
            assert is_psd(A) is True, "This algorithm only works for PSD"
            w = conjugate_grad_torch(
                (torch.eye(A.shape[0]) + A / k), V[:, i].reshape(-1, 1), tolerance=tolerance
            ).reshape(-1, 1)
            W = w
        else:
            raise NotImplementedError("Other methods are not yet implemented")

        temp_vec = torch.zeros(W.shape[0], 1)
        for j in range(i+1):
            Alpha[j][i] = torch.dot(V[:, j], W.squeeze())
            temp_vec += Alpha[j][i] * V[:, j].reshape(-1, 1)

        # orthogonalize
        w = W - temp_vec
        Alpha[i + 1][i] = torch.norm(w.squeeze()) + eps
        V[:, i + 1] = w.squeeze() / Alpha[i + 1][i]
        for j in range(i + 2, k + 1):
            Alpha[j][i] = 0

    T_hat = 0.5 * (Alpha + Alpha.t())
   
    B = torch.linalg.matrix_exp(
        k * (torch.eye(T_hat.shape[0]) - torch.linalg.inv(T_hat))
    )
    approx_vec = torch.matmul(V,B)[:, 0]
    return approx_vec * norm_v