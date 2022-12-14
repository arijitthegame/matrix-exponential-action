import argparse
import torch
from compute_action_matrix_exp import compute_lanczos_matrix_exp, is_psd

parser = argparse.ArgumentParser(
    description="Testing Action of matrix expoentials on vectors via Lanczos method"
)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument(
    "--num_dim", default=16, type=int, help="Dim of the Arnoldi subspace"
)
parser.add_argument(
    "--matrix_dim",
    default=256,
    type=int,
    help="Dim of the matrix for which we want to compute the action",
)


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # generate some random matrices
    A = torch.rand(args.matrix_dim, args.matrix_dim)
    A = A + A.t()  # symmetrize
    print(is_psd(A))  # there is (.5)^{matrix_dim} that it is PSD.
    A1 = A @ A.t()  # this is PSD
    # generate a random vector
    v = torch.rand(args.matrix_dim)
    # compute actions of exp(-A) (resp. exp(-A1)) and test with GT
    w0, exp0 = compute_lanczos_matrix_exp(
        A.unsqueeze(0), v, args.num_dim, return_exp=True
    )
    exp1 = torch.linalg.matrix_exp(-A)
    w1 = torch.matmul(exp1, v)
    print("PRINTING RESULTS FOR A NON PSD MATRIX")
    print("*" * 50)
    print(
        "Difference between the exponentials is ", ((exp1 - exp0.squeeze()) ** 2).sum()
    )
    print(
        "Difference between the GT vector and the estimated vector is ",
        ((w1 - w0.squeeze()) ** 2).sum(),
    )
    del w0, exp0, exp1, w1

    w0, exp0 = compute_lanczos_matrix_exp(
        A1.unsqueeze(0), v, args.num_dim, return_exp=True
    )
    exp1 = torch.linalg.matrix_exp(-A1)
    w1 = torch.matmul(exp1, v)
    print("PRINTING RESULTS FOR A PSD MATRIX")
    print("*" * 50)
    print(
        "Difference between the exponentials is ", ((exp1 - exp0.squeeze()) ** 2).sum()
    )
    print(
        "Difference between the GT vector and the estimated vector is ",
        ((w1 - w0.squeeze()) ** 2).sum(),
    )


if __name__ == "__main__":
    main()
