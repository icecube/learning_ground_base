
import torch
from xformers.ops.fmha.common import Inputs
from xformers.ops.fmha import _fMHA, _memory_efficient_attention_forward, _memory_efficient_attention_forward_requires_grad
from xformers.ops import fmha

def memory_efficient_with_lse(query, key, value, attn_bias=None, op=None):

    """
    _memory_efficient_attention(
        Inputs(
            query=query,
            key=key,
            value=value,
            p=p,
            attn_bias=attn_bias,
            scale=scale,
            output_dtype=output_dtype,
        ),
        op=op,
    )
    """

    inp=Inputs(
            query=query,
            key=key,
            value=value,
            p=0.0,
            attn_bias=attn_bias
        )

    res0=fmha.memory_efficient_attention(query, key, value, attn_bias=attn_bias, op=op)

    res1=_memory_efficient_attention_forward(inp, op=op)

    res2=_memory_efficient_attention_forward_requires_grad(inp, op=op)

    """
    output_shape = inp.normalize_bmhk()
    result =_fMHA.apply(
        op, inp.query, inp.key, inp.value, inp.attn_bias, inp.p, inp.scale
    ).reshape(output_shape)
    """
    print("RES ....0")
    print(res0)

    print("RES ....1")
    print(res1)

    print("RES ....2")
    print(res2[1].lse)
    print(torch.nn.functional.softmax(-res2[1].lse, dim=-1))
    
    sys.exit(-1)

    