import torch
from torch.autograd import Function, Variable
from torch.nn import Module
from torch.nn.parameter import Parameter

import numpy as np
import numpy.random as npr

from collections import namedtuple

import time

from . import util, mpc
from .pnqp import pnqp

LqrBackOut = namedtuple('lqrBackOut', 'n_total_qp_iter')
LqrForOut = namedtuple(
    'lqrForOut',
    'objs full_du_norm alpha_du_norm mean_alphas costs'
)


class LQRStep(Function):
    """A single step of the box-constrained iLQR solver.

    Required Args:
        n_state, n_ctrl, T
        x_init: The initial state [n_batch, n_state]

    Optional Args:
        u_lower, u_upper: The lower- and upper-bounds on the controls.
            These can either be floats or shaped as [T, n_batch, n_ctrl]
            TODO: Better support automatic expansion of these.
        TODO
    """

    # @profile
    @staticmethod
    def forward(ctx, x_init, C, c, F, f=None,
                n_state=None,
                n_ctrl=None,
                T=None,
                u_lower=None,
                u_upper=None,
                u_zero_I=None,
                delta_u=None,
                linesearch_decay=0.2,
                max_linesearch_iter=10,
                true_cost=None,
                true_dynamics=None,
                delta_space=True,
                current_x=None,
                current_u=None,
                verbose=0,
                back_eps=1e-3,
                no_op_forward=False,      
    ):
        ctx.n_state = n_state
        ctx.n_ctrl = n_ctrl
        ctx.T = T
        ctx.u_lower = util.get_data_maybe(u_lower)
        ctx.u_upper = util.get_data_maybe(u_upper)
        
        # TODO: Better checks for this
        if isinstance(ctx.u_lower, int):
            ctx.u_lower = float(ctx.u_lower)
        if isinstance(ctx.u_upper, int):
            ctx.u_upper = float(ctx.u_upper)
        if isinstance(ctx.u_lower, np.float32):
            ctx.u_lower = u_lower.item()
        if isinstance(ctx.u_upper, np.float32):
            ctx.u_upper = u_upper.item()
        
        ctx.u_zero_I = u_zero_I
        
        ctx.delta_u = delta_u
        
        # if isinstance(delta_u, torch.Tensor):
        #     ctx.delta_u = delta_u
        # elif isinstance(delta_u, (int, float)):
        #     ctx.delta_u = delta_u
        # else:
        #     ctx.delta_u = None
            
        ctx.linesearch_decay = linesearch_decay
        ctx.max_linesearch_iter = max_linesearch_iter
        ctx.true_cost = true_cost
        ctx.true_dynamics = true_dynamics
        ctx.delta_space = delta_space
        ctx.current_x = util.get_data_maybe(current_x)
        ctx.current_u = util.get_data_maybe(current_u)
        ctx.verbose = verbose
        ctx.back_eps = back_eps
        ctx.no_op_forward = no_op_forward
        
        if ctx.no_op_forward:
            ctx.save_for_backward(
                x_init, C, c, F, f, ctx.current_x, ctx.current_u)
            return ctx.current_x, ctx.current_u, None, None

        if ctx.delta_space:
            # Taylor-expand the objective to do the backward pass in
            # the delta space.
            assert ctx.current_x is not None
            assert ctx.current_u is not None
            c_back = []
            for t in range(ctx.T):
                xt = ctx.current_x[t]
                ut = ctx.current_u[t]
                xut = torch.cat((xt, ut), 1)
                c_back.append(util.bmv(C[t], xut) + c[t])
            c_back = torch.stack(c_back)
            f_back = None
        else:
            assert False

        Ks, ks, ctx.back_out = LQRStep.lqr_backward(ctx, C, c_back, F, f_back)
        new_x, new_u, ctx.for_out = LQRStep.lqr_forward(ctx,
            x_init, C, c, F, f, Ks, ks)
        ctx.save_for_backward(x_init, C, c, F, f, new_x, new_u)

        return new_x, new_u, ctx.back_out, ctx.for_out
    
    @staticmethod
    def backward(ctx, dl_dx, dl_du, back_out, for_out):
        start = time.time()
        x_init, C, c, F, f, new_x, new_u = ctx.saved_tensors

        r = []
        for t in range(ctx.T):
            rt = torch.cat((dl_dx[t], dl_du[t]), 1)
            r.append(rt)
        r = torch.stack(r)

        if ctx.u_lower is None:
            I = None
        else:
            I = (torch.abs(new_u - ctx.u_lower) <= 1e-8) | \
                (torch.abs(new_u - ctx.u_upper) <= 1e-8)
        dx_init = Variable(torch.zeros_like(x_init))
        _mpc = mpc.MPC(
            ctx.n_state, ctx.n_ctrl, ctx.T,
            u_zero_I=I,
            u_init=None,
            lqr_iter=1,
            verbose=-1,
            n_batch=C.size(1),
            delta_u=None,
            # exit_unconverged=True, # It's really bad if this doesn't converge.
            exit_unconverged=False, # It's really bad if this doesn't converge.
            eps=ctx.back_eps,
        )
        dx, du, _ = _mpc(dx_init, mpc.QuadCost(C, -r), mpc.LinDx(F, None))

        dx, du = dx.data, du.data
        dxu = torch.cat((dx, du), 2)
        xu = torch.cat((new_x, new_u), 2)

        dC = torch.zeros_like(C)
        for t in range(ctx.T):
            xut = torch.cat((new_x[t], new_u[t]), 1)
            dxut = dxu[t]
            dCt = -0.5*(util.bger(dxut, xut) + util.bger(xut, dxut))
            dC[t] = dCt

        dc = -dxu

        lams = []
        prev_lam = None
        for t in range(ctx.T-1, -1, -1):
            Ct_xx = C[t,:,:ctx.n_state,:ctx.n_state]
            Ct_xu = C[t,:,:ctx.n_state,ctx.n_state:]
            ct_x = c[t,:,:ctx.n_state]
            xt = new_x[t]
            ut = new_u[t]
            lamt = util.bmv(Ct_xx, xt) + util.bmv(Ct_xu, ut) + ct_x
            if prev_lam is not None:
                Fxt = F[t,:,:,:ctx.n_state].transpose(1, 2)
                lamt += util.bmv(Fxt, prev_lam)
            lams.append(lamt)
            prev_lam = lamt
        lams = list(reversed(lams))

        dlams = []
        prev_dlam = None
        for t in range(ctx.T-1, -1, -1):
            dCt_xx = C[t,:,:ctx.n_state,:ctx.n_state]
            dCt_xu = C[t,:,:ctx.n_state,ctx.n_state:]
            drt_x = -r[t,:,:ctx.n_state]
            dxt = dx[t]
            dut = du[t]
            dlamt = util.bmv(dCt_xx, dxt) + util.bmv(dCt_xu, dut) + drt_x
            if prev_dlam is not None:
                Fxt = F[t,:,:,:ctx.n_state].transpose(1, 2)
                dlamt += util.bmv(Fxt, prev_dlam)
            dlams.append(dlamt)
            prev_dlam = dlamt
        dlams = torch.stack(list(reversed(dlams)))

        dF = torch.zeros_like(F)
        for t in range(ctx.T-1):
            xut = xu[t]
            lamt = lams[t+1]

            dxut = dxu[t]
            dlamt = dlams[t+1]

            dF[t] = -(util.bger(dlamt, xut) + util.bger(lamt, dxut))

        if f.nelement() > 0:
            _dlams= dlams[1:]
            assert _dlams.shape == f.shape
            df = -_dlams
        else:
            df = torch.Tensor()

        dx_init = -dlams[0]

        ctx.backward_time = time.time()-start
        return dx_init, dC, dc, dF, df, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    # @profile
    @staticmethod
    def lqr_backward(ctx, C, c, F, f):
        n_batch = C.size(1)

        u = ctx.current_u
        Ks = []
        ks = []
        prev_kt = None
        n_total_qp_iter = 0
        Vtp1 = vtp1 = None
        for t in range(ctx.T-1, -1, -1):
            if t == ctx.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1,2)
                Qt = C[t] + Ft_T.bmm(Vtp1).bmm(Ft)
                if f is None or f.nelement() == 0:
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
                        Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)

            n_state = ctx.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if ctx.u_lower is None:
                if ctx.n_ctrl == 1 and ctx.u_zero_I is None:
                    Kt = -(1./Qt_uu)*Qt_ux
                    kt = -(1./Qt_uu.squeeze(2))*qt_u
                else:
                    if ctx.u_zero_I is None:
                        Qt_uu_inv = [
                            torch.pinverse(Qt_uu[i]) for i in range(Qt_uu.shape[0])
                        ]
                        Qt_uu_inv = torch.stack(Qt_uu_inv)
                        Kt = -Qt_uu_inv.bmm(Qt_ux)
                        kt = util.bmv(-Qt_uu_inv, qt_u)

                        # Qt_uu_LU = Qt_uu.btrifact()
                        # Kt = -Qt_ux.btrisolve(*Qt_uu_LU)
                        # kt = -qt_u.btrisolve(*Qt_uu_LU)
                    else:
                        # Solve with zero constraints on the active controls.
                        I = ctx.u_zero_I[t]
                        # print(f"I dtype: {I.dtype}", flush=True)
                        notI = ~I # 1-I

                        qt_u_ = qt_u.clone()
                        qt_u_[I] = 0

                        Qt_uu_ = Qt_uu.clone()

                        if I.is_cuda:
                            notI_ = notI.float()
                            Qt_uu_I = (1-util.bger(notI_, notI_)).type_as(I)
                        else:
                            Qt_uu_I = (1-util.bger(notI, notI)).type_as(I)
                            # print(f"Qt_uu_I dtype: {Qt_uu_I.dtype}", flush=True)
                        Qt_uu_[Qt_uu_I] = 0.
                        Qt_uu_[util.bdiag(I)] += 1e-8

                        Qt_ux_ = Qt_ux.clone()
                        Qt_ux_[I.unsqueeze(2).repeat(1,1,Qt_ux.size(2))] = 0.

                        if ctx.n_ctrl == 1:
                            Kt = -(1./Qt_uu_)*Qt_ux_
                            kt = -(1./Qt_uu.squeeze(2))*qt_u_
                        else:
                            Qt_uu_LU_ = Qt_uu_.btrifact()
                            Kt = -Qt_ux_.btrisolve(*Qt_uu_LU_)
                            kt = -qt_u_.btrisolve(*Qt_uu_LU_)
            else:
                assert ctx.delta_space
                lb = LQRStep.get_bound(ctx, 'lower', t) - u[t]
                ub = LQRStep.get_bound(ctx, 'upper', t) - u[t]
                if ctx.delta_u is not None:
                    lb[lb < -ctx.delta_u] = -ctx.delta_u
                    ub[ub > ctx.delta_u] = ctx.delta_u
                kt, Qt_uu_free_LU, If, n_qp_iter = pnqp(
                    Qt_uu, qt_u, lb, ub,
                    x_init=prev_kt, n_iter=20)
                if ctx.verbose > 1:
                    print('  + n_qp_iter: ', n_qp_iter+1)
                n_total_qp_iter += 1+n_qp_iter
                prev_kt = kt
                Qt_ux_ = Qt_ux.clone()
                Qt_ux_[(~If).unsqueeze(2).repeat(1,1,Qt_ux.size(2))] = 0  # 1-If
                if ctx.n_ctrl == 1:
                    # Bad naming, Qt_uu_free_LU isn't the LU in this case.
                    Kt = -((1./Qt_uu_free_LU)*Qt_ux_)
                else:
                    Kt = -Qt_ux_.btrisolve(*Qt_uu_free_LU)

            Kt_T = Kt.transpose(1,2)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks, LqrBackOut(n_total_qp_iter=n_total_qp_iter)


    # @profile
    @staticmethod
    def lqr_forward(ctx, x_init, C, c, F, f, Ks, ks):
        x = ctx.current_x
        u = ctx.current_u
        n_batch = C.size(1)

        old_cost = util.get_cost(ctx.T, u, ctx.true_cost, ctx.true_dynamics, x=x)

        current_cost = None
        alphas = torch.ones(n_batch).type_as(C)
        full_du_norm = None

        i = 0
        while (current_cost is None or \
               (old_cost is not None and \
                  torch.any((current_cost > old_cost)).cpu().item() == 1)) and \
              i < ctx.max_linesearch_iter:
            new_u = []
            new_x = [x_init]
            dx = [torch.zeros_like(x_init)]
            objs = []
            for t in range(ctx.T):
                t_rev = ctx.T-1-t
                Kt = Ks[t_rev]
                kt = ks[t_rev]
                new_xt = new_x[t]
                xt = x[t]
                ut = u[t]
                dxt = dx[t]
                new_ut = util.bmv(Kt, dxt) + ut + torch.diag(alphas).mm(kt)

                # Currently unimplemented:
                assert not ((ctx.delta_u is not None) and (ctx.u_lower is None))

                if ctx.u_zero_I is not None:
                    new_ut[ctx.u_zero_I[t]] = 0.

                if ctx.u_lower is not None:
                    lb = LQRStep.get_bound(ctx, 'lower', t)
                    ub = LQRStep.get_bound(ctx, 'upper', t)

                    if ctx.delta_u is not None:
                        lb_limit, ub_limit = lb, ub
                        lb = u[t] - ctx.delta_u
                        ub = u[t] + ctx.delta_u
                        I = lb < lb_limit
                        lb[I] = lb_limit if isinstance(lb_limit, float) else lb_limit[I]
                        I = ub > ub_limit
                        ub[I] = ub_limit if isinstance(lb_limit, float) else ub_limit[I]
                    new_ut = util.eclamp(new_ut, lb, ub)
                new_u.append(new_ut)

                new_xut = torch.cat((new_xt, new_ut), dim=1)
                if t < ctx.T-1:
                    if isinstance(ctx.true_dynamics, mpc.LinDx):
                        F, f = ctx.true_dynamics.F, ctx.true_dynamics.f
                        new_xtp1 = util.bmv(F[t], new_xut)
                        if f is not None and f.nelement() > 0:
                            new_xtp1 += f[t]
                            #if t==0:
                            #    print("In lqr_step: ", f[t])
                    else:
                        new_xtp1 = ctx.true_dynamics(
                            Variable(new_xt), Variable(new_ut)).data

                    new_x.append(new_xtp1)
                    dx.append(new_xtp1 - x[t+1])

                if isinstance(ctx.true_cost, mpc.QuadCost):
                    C, c = ctx.true_cost.C, ctx.true_cost.c
                    obj = 0.5*util.bquad(new_xut, C[t]) + util.bdot(new_xut, c[t])
                else:
                    obj = ctx.true_cost(new_xut)

                objs.append(obj)

            objs = torch.stack(objs)
            current_cost = torch.sum(objs, dim=0)

            new_u = torch.stack(new_u)
            new_x = torch.stack(new_x)
            if full_du_norm is None:
                full_du_norm = (u-new_u).transpose(1,2).contiguous().view(
                    n_batch, -1).norm(2, 1)

            alphas[current_cost > old_cost] *= ctx.linesearch_decay
            i += 1

        # If the iteration limit is hit, some alphas
        # are one step too small.
        alphas[current_cost > old_cost] /= ctx.linesearch_decay
        alpha_du_norm = (u-new_u).transpose(1,2).contiguous().view(
            n_batch, -1).norm(2, 1)

        return new_x, new_u, LqrForOut(
            objs, full_du_norm,
            alpha_du_norm,
            torch.mean(alphas),
            current_cost
        )

    @staticmethod
    def get_bound(ctx, side, t):
        v = getattr(ctx, 'u_'+side)
        if isinstance(v, float):
            return v
        else:
            return v[t]
