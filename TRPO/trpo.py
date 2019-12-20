import numpy as np 
import torch
from torch.autograd import Variable
from utils import *


def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b - Avp(x)
    p = r
    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        _Avp = Avp(p)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        if rdotr<residual_tol:
            break
    return x

def linesearch(model, f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=0.1):
    fval = f(True).data
    print("fval before", fval.item())
    for(_n_backtracks, step_frac) in enumerate(0.5**np.arange(max_backtracks)):
        xnew = x + step_frac * fullstep
        set_flat_params_to(model, xnew)
        newfval = f(True).data
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * step_frac
        ratio = actual_improve / expected_improve
        print("a/e/r", actual_improve.item(), expected_improve.item(), ratio.item())

        if ratio.item() > accept_ratio and actual_improve.item() > 0:
            print("fval after", newfval.item())
            return True, xnew
    return False, x

def trpo_step(model, get_loss, get_kl, max_kl, damping):
    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    def Fvp(v):
        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads]).data
        
        kl_v = (flat_grad_kl * Variable(v)).sum()
        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_kl + v * damping

    stepdir = conjugate_gradient(Fvp, -loss_grad, 10)
    shs = 0.5 * (stepdir * Fvp(stepdir)).sum(0, keepdim=True)
    lm = torch.sqrt(shs / max_kl)
    fullstep = stepdir / lm[0]

    neggdot_stepdir = (- loss_grad * stepdir).sum(0, keepdim=True)
    print(("lagrange multiplier: ", lm[0], "gradnorm: ", loss_grad.norm()))

    prev_params = get_flat_grad_from(model)
    success, new_params = linesearch(model, get_loss, prev_params, fullstep, neggdot_stepdir / lm[0])
    set_flat_params_to(model, new_params)

    return loss


