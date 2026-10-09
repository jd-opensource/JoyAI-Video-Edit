import torch
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """Build an n-dimensional grid from start, stop, and size."""
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # Exclude each axis endpoint.
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")
    grid = torch.stack(grid, dim=0)

    return grid


# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    x: torch.Tensor,
    head_first=False,
):
    """Align RoPE frequencies with the input tensor layout."""
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            if freqs_cis[0].shape == (x.shape[0], x.shape[1], x.shape[-1]):
                shape = [
                    d if i == 0 or i == 1 or i == ndim - 1 else 1
                    for i, d in enumerate(x.shape)
                ]
            else:
                assert freqs_cis[0].shape == (
                    x.shape[1],
                    x.shape[-1],
                ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
                shape = [d if i == 1 or i == ndim -
                         1 else 1 for i, d in enumerate(x.shape)]
            return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [
                d if i == ndim - 2 or i == ndim - 1 else 1
                for i, d in enumerate(x.shape)
            ]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim -
                     1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = (
        x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    )
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query/key tensors."""
    assert xq.shape[-1] % 2 == 0 and xk.shape[-1] % 2 == 0, (
        f"RoPE requires an even head dimension, got q={xq.shape[-1]}, k={xk.shape[-1]}"
    )

    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # Multiply interleaved real/imaginary pairs.
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs = reshape_for_broadcast(freqs_cis, xq_complex, head_first).to(xq.device)
        xq_out = torch.view_as_real(xq_complex * freqs).flatten(-2).type_as(xq)
        xk_out = torch.view_as_real(xk_complex * freqs).flatten(-2).type_as(xk)

    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    txt_rope_size=None,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """Build RoPE frequencies for an n-dimensional token grid."""
    assert all(dim % 2 == 0 for dim in rope_dim_list), (
        f"Each rope dimension must be even, got {rope_dim_list}"
    )

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            grid[i].reshape(-1),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        vis_emb = (cos, sin)
    else:
        vis_emb = torch.cat(embs, dim=1)
    if txt_rope_size is not None:
        embs_txt = []
        vis_max_ids = grid.view(-1).max().item()
        grid_txt = torch.arange(txt_rope_size) + vis_max_ids + 1
        for i in range(len(rope_dim_list)):
            emb = get_1d_rotary_pos_embed(
                rope_dim_list[i],
                grid_txt,
                theta,
                use_real=use_real,
                theta_rescale_factor=theta_rescale_factor[i],
                interpolation_factor=interpolation_factor[i],
            )
            embs_txt.append(emb)
        if use_real:
            cos = torch.cat([emb[0] for emb in embs_txt], dim=1)
            sin = torch.cat([emb[1] for emb in embs_txt], dim=1)
            txt_emb = (cos, sin)
        else:
            txt_emb = torch.cat(embs_txt, dim=1)
    else:
        txt_emb = None
    return vis_emb, txt_emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Build one-dimensional RoPE frequencies with optional scaling."""
    assert dim % 2 == 0, f"RoPE dimension must be even, got {dim}"
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    # NTK-aware frequency scaling (bloc97).
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    pos_device = pos.device if torch.is_tensor(pos) else torch.device("cpu")
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=pos_device)[: (dim // 2)].float() / dim)
    )
    freqs = torch.outer(pos * interpolation_factor, freqs)
    if use_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1)
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1)
        return freqs_cos, freqs_sin
    else:
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )
        return freqs_cis
