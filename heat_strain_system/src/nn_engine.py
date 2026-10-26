"""
A minimal reverse-mode autodiff engine (NumPy-only).

torch could not be installed in this container (disk/RAM constrained), so
the adaptive-fusion / domain-adversarial network is implemented on top of
this tiny engine instead of hand-derived gradients for every layer, which
would be far more bug-prone. It supports exactly the ops the fusion model
needs: matmul+bias (Linear), ReLU, softmax-cross-entropy, dropout, and a
Gradient-Reversal Layer (GRL) for the domain-adversarial branch.
"""
from __future__ import annotations
import numpy as np


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, _prev=(), requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = _prev
        self.requires_grad = requires_grad

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __add__(self, other):
        # supports bias broadcasting (row vector + matrix)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            g = out.grad
            self.grad += g if g.shape == self.data.shape else g.sum(axis=0, keepdims=(self.data.ndim > 1))
            other.grad += g if g.shape == other.data.shape else g.sum(axis=0, keepdims=(other.data.ndim > 1))
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other) if isinstance(other, Tensor) else self + Tensor(np.full_like(self.data, other), requires_grad=False)

    def __mul__(self, scalar):
        out = Tensor(self.data * scalar, (self,))

        def _backward():
            self.grad += scalar * out.grad
        out._backward = _backward
        return out

    __rmul__ = __mul__

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,))

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def dropout(self, p: float, training: bool, rng: np.random.Generator):
        if not training or p <= 0:
            return self
        mask = (rng.random(self.data.shape) > p) / (1.0 - p)
        out = Tensor(self.data * mask, (self,))

        def _backward():
            self.grad += mask * out.grad
        out._backward = _backward
        return out

    def grl(self, lam: float):
        """Gradient Reversal Layer: identity on the forward pass, negates
        (and scales) the gradient on the backward pass."""
        out = Tensor(self.data.copy(), (self,))

        def _backward():
            self.grad += -lam * out.grad
        out._backward = _backward
        return out

    def hcat(self, others):
        tensors = [self] + list(others)
        out = Tensor(np.concatenate([t.data for t in tensors], axis=1), tensors)
        splits = np.cumsum([t.data.shape[1] for t in tensors])[:-1]

        def _backward():
            grads = np.split(out.grad, splits, axis=1)
            for t, g in zip(tensors, grads):
                t.grad += g
        out._backward = _backward
        return out

    def weighted_sum(self, weight_tensor, n_parts: int):
        """self: (N, n_parts*d) stacked embeddings; weight_tensor: (N, n_parts)
        attention weights. Returns (N, d) = sum_k w[:,k] * emb_k."""
        N, total = self.data.shape
        d = total // n_parts
        parts = self.data.reshape(N, n_parts, d)
        w = weight_tensor.data.reshape(N, n_parts, 1)
        out_data = (parts * w).sum(axis=1)
        out = Tensor(out_data, (self, weight_tensor))

        def _backward():
            g = out.grad.reshape(N, 1, d)
            self.grad += (g * w).reshape(N, n_parts * d)
            weight_tensor.grad += (g * parts).sum(axis=2)
        out._backward = _backward
        return out

    def softmax(self):
        z = self.data - self.data.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        out = Tensor(p, (self,))

        def _backward():
            g = out.grad
            dot = np.sum(g * p, axis=1, keepdims=True)
            self.grad += p * (g - dot)
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


def cross_entropy_loss(probs: Tensor, y_onehot: np.ndarray, sample_weight=None):
    """probs: Tensor of softmax probabilities (N,C). Returns scalar Tensor."""
    eps = 1e-9
    p = np.clip(probs.data, eps, 1 - eps)
    per_sample = -np.sum(y_onehot * np.log(p), axis=1)
    if sample_weight is not None:
        per_sample = per_sample * sample_weight
        denom = sample_weight.sum()
    else:
        denom = len(per_sample)
    loss_val = per_sample.sum() / max(denom, 1e-9)
    out = Tensor(loss_val, (probs,))

    def _backward():
        w = sample_weight if sample_weight is not None else np.ones(len(p))
        grad = -(y_onehot / p) * (w[:, None]) / max(denom, 1e-9) * out.grad
        probs.grad += grad
    out._backward = _backward
    return out


def focal_loss(probs: Tensor, y_onehot: np.ndarray, gamma=2.0, alpha=None):
    """Multi-class focal loss on top of softmax probabilities."""
    eps = 1e-9
    p = np.clip(probs.data, eps, 1 - eps)
    pt = np.sum(y_onehot * p, axis=1)
    if alpha is not None:
        a = np.sum(y_onehot * alpha[None, :], axis=1)
    else:
        a = 1.0
    per_sample = -a * (1 - pt) ** gamma * np.log(pt)
    loss_val = per_sample.mean()
    out = Tensor(loss_val, (probs,))

    def _backward():
        # d(loss)/dp_c for the true class c, others ~0 via one-hot mask
        dpt = -a * ((1 - pt) ** gamma / pt - gamma * (1 - pt) ** (gamma - 1) * np.log(pt))
        dpt = dpt / len(pt)
        grad = y_onehot * dpt[:, None]
        probs.grad += grad * out.grad
    out._backward = _backward
    return out


class Linear:
    def __init__(self, d_in, d_out, rng: np.random.Generator):
        limit = np.sqrt(6.0 / (d_in + d_out))
        self.W = Tensor(rng.uniform(-limit, limit, size=(d_in, d_out)))
        self.b = Tensor(np.zeros((1, d_out)))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class Adam:
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self, clip_norm=5.0):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p.data
            if clip_norm is not None:
                n = np.linalg.norm(g)
                if n > clip_norm:
                    g = g * (clip_norm / (n + 1e-9))
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g ** 2)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
