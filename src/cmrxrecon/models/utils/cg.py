import torch


def inner_product(t1, t2=None):
    """
    Inner product between two tensors.
    If t2 is None, the inner product is calculated between t1 and itself.
    """
    if t2 is None:
        if torch.is_complex(t1):
            return torch.sum(t1.flatten() * t1.flatten().conj())
        return torch.sum(t1.flatten() * t1.flatten())
    if torch.is_complex(t2):
        return torch.sum(t1.flatten() * t2.flatten().conj())
    return torch.sum(t1.flatten() * t2.flatten())


def conj_grad(H, b, x=None, niter=4, tol=None):
    """
    Conjugate gradient solver for linear systems of the form Hx = b.

    Parameters
    ----------
    H : function
        Function that performs the matrix-vector product Hx.
    b : torch.Tensor
        Right-hand side of the linear system.
    x : torch.Tensor, optional
        Starting value for the solver. If None, x is initialized to b
    niter : int, default=4
        Maximum number of iterations.
    tol : float, optional
        Tolerance for the residual norm.
    """
    if tol is not None:
        sqnorm_b = inner_product(b)
        tol = tol * sqnorm_b

    if x is None:
        x = torch.clone(b)

    r = b - H(x)
    norm_r_old = inner_product(r)
    p = r.clone()

    for kiter in range(niter):
        d = H(p)
        alpha = norm_r_old / inner_product(p, d)
        x = x + p * alpha
        r = r - d * alpha
        sqnorm_r_new = inner_product(r)

        if tol is not None and sqnorm_r_new.item() < tol:
            # the residual norm is below the tolerance
            break

        beta = sqnorm_r_new / norm_r_old
        norm_r_old = sqnorm_r_new
        p = r + p * beta

    return x
