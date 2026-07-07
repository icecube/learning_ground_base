import torch
import torch.autograd

import numpy
import subprocess
import os

def get_gpu_mem(extra_str):


    if(torch.cuda.is_available()):
        device_id=0
        """
        os.environ['CUDA_VISIBLE_DEVICES']#torch.cuda.current_device()
        if(device_id is None):
            device_id=0
        else:
            device_id=int(device_id)
        """
        res=subprocess.run(["nvidia-smi"], capture_output=True)
        res=str(res.stdout).split("|   %d" % device_id)[1]

        first=res.find(" / ")

        sec=res.find(" / ", first+1)

        ret_str="GPU USAGE of device %d: %s " % (device_id, res[sec-8:sec]) + extra_str
        return ret_str

    else:
        return "no GPU enabled"

def jacobian(f, x):

    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """
    print("before f(x)")
    print("F", f)
    B, N = x.shape
    y = f(x)
    jacobian = list()
    print(f.parameters())
    print(y)

    print("printing all...")
    for i in y[:,0]:
        print(i)
    

    for i in range(y.shape[1]):
        v = torch.zeros_like(y)
        v[:, i] = 1.

        print("X", x)
        print(x[0])
        print(y[0])
        dy_i_dx = torch.autograd.grad(y[0,i],
                       x,
                       grad_outputs=v,
                       retain_graph=False,
                       create_graph=True,
                       allow_unused=True)[0]  # shape [B, N]

        print("DYDX", dy_i_dx)
        print("DYDX SHAPE", dy_i_dx)

        sys.exit(-1)
        jacobian.append(dy_i_dx)



    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian

