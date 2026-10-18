import math
from typing import Any

import torch
from torch import Tensor, nn


class TokenNorm(nn.Module):
    def __init__(self, D: int, eps: float = 1e-5, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.eps: float = eps  # use eps to avoid division by zero errors
        self.gamma = nn.Parameter(torch.ones([D, 1]))
        self.beta = nn.Parameter(torch.zeros([D, 1]))

    def forward(self, X: Tensor, dim: int = 0) -> Tensor:
        # X [D, N]
        X_bar = (
            1
            / (torch.sqrt(torch.var(X, dim=0, keepdim=True, correction=0) + self.eps))
            * (X - torch.mean(X, dim=0, keepdim=True))
            * self.gamma
            + self.beta
        )  # [D, N]
        # correction = 0: to divide by D and not (D-1) for variance
        return X_bar


def norm(X: Tensor, dim: int = 0) -> Tensor:
    # X [D, N]
    X = X / torch.sum(X, dim=dim, keepdim=True)
    return X


class AttentionHead(nn.Module):
    def __init__(self, D: int, K: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        # self.U_k = torch.Tensor(size=(K, D))
        # self.U_q = torch.Tensor(size=(K, D))
        # self.V = torch.Tensor(size=(D, D))

        self.K: int = K

        self.U_k: Tensor = nn.Parameter(torch.randn(K, D) * (D**-0.5))
        self.U_q: Tensor = nn.Parameter(torch.randn(K, D) * (D**-0.5))
        self.V: Tensor = nn.Parameter(torch.randn(D, D) * (D**-0.5))

    def forward(self, X_l: Tensor) -> Tensor:
        # X_l = X^(m-1) from the paper
        U_k_X = (self.U_k @ X_l).T  # ([K, D] @ [D, N]).T -> ([K, N]).T -> [N, K]
        U_q_X = self.U_q @ X_l  #      [K, D] @ [D, N]    ->  [K, N]
        KQ = U_k_X @ U_q_X  #              [N, K] @ [K, N]    ->  [N, N]
        # The denominator is the same for all x_n across dim 1
        # ([1 ... N] repeated for N rows)
        # See version with loops below

        # The comment below is a more literal translation from the paper:
        # N = X_l.shape[1]
        # A = torch.exp(KQ / math.sqrt(self.K)) / torch.sum(torch.exp(KQ / math.sqrt(self.K)), dim=0).repeat(N, 1)
        # [N,N] / [N].repeat(N,1)  -> [N,N]
        # Tensor.repeat(size) works as follows:
        # [N].repeat(N,1) -> [N,N]

        # You can see the that A is the dot product of KQ normalized along dim=1
        # A = norm(torch.exp(KQ / math.sqrt(self.K)), dim=0)  # [N,N]

        # The simplest way and numerical most stable way is to use softmax:
        # Divide by K and not D
        # Footnote 8 in paper references "dimensionality of the projected vector" which is K and not D
        A = torch.softmax(KQ / math.sqrt(self.K), dim=0)

        # Y^(m)_h from the paper (no sum across H dimension, yet)
        Y = self.V @ X_l @ A  # [D,D] @ [D,N] @ [N,N] -> [D, N]
        return Y

    def looped_forward(self, X_l: Tensor) -> Tensor:
        # X_l = X^(m-1) from the paper
        N = X_l.shape[1]
        A = torch.zeros(size=(N, N))

        for n_1 in range(N):
            x_n1 = X_l[:, n_1]
            for n_2 in range(N):
                x_n2 = X_l[:, n_2]

                num = torch.exp((self.U_k @ x_n1).T @ (self.U_q @ x_n2) / math.sqrt(self.K))
                denom = 0

                for n_3 in range(N):
                    x_n3 = X_l[:, n_3]
                    # NOTE: The denominator is the same for all x_n2.
                    #       You can either save the denom, normalize A along dim=1, or rearrange the loops:
                    #        for n_2 in range(N): ...
                    #          for n_3 in range(N): ...
                    #            for n_1 in range(N): ...
                    denom += torch.exp((self.U_k @ x_n3).T @ (self.U_q @ x_n2) / math.sqrt(self.K))

                denom = denom.type(dtype=float).item()

                A[n_1, n_2] = num / denom

        # Y^(m)_h from the paper (no sum across H dimension, yet)
        Y = self.V @ X_l @ A  # [D,D] @ [D,N] @ [N,N] -> [D, N]

        return Y


class MultiHeadedSelfAttention(nn.Module):
    # TODO: refactor into nicer code that just has MHSA class
    # and creates U_k, U_q [H,K,D] and V [H,D,D] matrices and finally sums across H
    def __init__(self, D: int, H: int, K: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.attention_heads = nn.ModuleList([AttentionHead(D=D, K=K) for _ in range(H)])

    def forward(self, X_l: Tensor) -> Tensor:
        # X_l = X^(m-1) from the paper
        Y_h = torch.stack([ah.forward(X_l) for ah in self.attention_heads], dim=0)  # [H, D, N]
        Y = torch.sum(Y_h, dim=0)  # [H, D, N] -> [D, N]
        return Y


class MLP(nn.Module):
    def __init__(self, D: int, HL: int, *args: Any, **kwargs: Any):
        # HL is the number is hidden layers
        super().__init__(*args, **kwargs)
        layers = []
        for i in range(HL + 1):  # number of hidden layers + one final output layer
            layers.append(nn.Linear(D, D))
            if i < HL - 1:
                layers.append(nn.ReLU())  # some non-linear activation function
        self.mlp = nn.Sequential(*layers)

    def forward(self, Y: Tensor) -> Tensor:
        # Y [D, N]
        X: Tensor = self.mlp(Y.T).T  # [D, N].T -> [N, D].T -> [D, N]
        return X

    def looped_forward(self, Y):
        X = torch.zeros_like(Y)
        for n in range(Y.shape[1]):
            X[:, n] = self.mlp(Y[:, n])
        return X


class TransformerBlock(nn.Module):
    def __init__(self, D: int, H: int, K: int, HL: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.mhsa: MultiHeadedSelfAttention = MultiHeadedSelfAttention(D=D, K=K, H=H)
        self.mlp: MLP = MLP(D, HL)
        self.token_norm_1: TokenNorm = TokenNorm(D=D)
        self.token_norm_2: TokenNorm = TokenNorm(D=D)

    def forward(self, X_l: Tensor) -> Tensor:
        # X_l = X^(m-1) from the paper
        X_l_bar = self.token_norm_1(X_l)  # TokenNorm
        MHSA_X_l_bar = self.mhsa(X_l_bar)  # MHSA
        Y_m = X_l + MHSA_X_l_bar  # residual connection

        Y_m_bar = self.token_norm_2(Y_m)  # TokenNorm
        MLP_Y_m_bar = self.mlp(Y_m_bar)  # MLP
        X_m = Y_m + MLP_Y_m_bar  # residual connection

        return X_m


class Transformer(nn.Module):
    def __init__(self, D: int, M: int, H: int, K: int, HL: int, incremental: bool = False, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(D=D, H=H, K=K, HL=HL) for _ in range(M)])

    def forward(self, X_0: Tensor) -> Tensor:
        X = X_0
        for m in range(len(self.transformer_blocks)):
            transformer_block_m = self.transformer_blocks[m]
            X = transformer_block_m(X)

        return X


class PositionEmbeddings(nn.Module):
    def __init__(self):
        pass
