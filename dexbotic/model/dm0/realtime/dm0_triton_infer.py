import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def rmsnorm_factor_kernel(
    inp_ptr, factor_ptr,
    rows: tl.constexpr, features: tl.constexpr,
    eps: tl.constexpr = 1e-6,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(axis=0)
    psize = tl.num_programs(axis=0)
    for i in range(pid, rows, psize):
        sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in range(0, features, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            x = tl.load(inp_ptr + i * features + offs, mask=offs < features, other=0.0)
            sum_x += x * x
        factor = 1.0 / tl.sqrt(tl.sum(sum_x) / features + eps)
        tl.store(factor_ptr + i, factor)

@triton.jit
def matmul_small_bias_silu(
    inp_ptr, weight_ptr, out_ptr, bias_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            acc = tl.dot(x, w, acc)
        acc = acc * tl.sigmoid(acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_bias(
    inp_ptr, weight_ptr, out_ptr, bias_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc += tl.load(
            bias_ptr + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0
        )
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def euler_step_kernel(
    x_ptr, norm_factor_ptr, weight_ptr, bias_ptr, noise_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, out_dim: tl.constexpr,
    BLOCK_K: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    j_offs = tl.arange(0, 32)
    for row in range(pid, seq_len, tl.num_programs(0)):
        factor = tl.load(norm_factor_ptr + row)
        acc = tl.load(bias_ptr + j_offs).to(tl.float32)
        for k in range(0, features, BLOCK_K):
            k_offs = k + tl.arange(0, BLOCK_K)
            k_mask = k_offs < features
            x = tl.load(x_ptr + row * features + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            x = x * factor
            w = tl.load(weight_ptr + k_offs[:, None] * out_dim + j_offs[None, :],
                        mask=k_mask[:, None],
                        other=0.0).to(tl.float32)
            acc += tl.sum(x[:, None] * w, axis=0)
        n = tl.load(noise_ptr + row * out_dim + j_offs).to(tl.float32)
        tl.store(noise_ptr + row * out_dim + j_offs, (n + acc).to(tl.bfloat16))

@triton.jit
def matmul_small_res(
    inp_ptr, weight_ptr, out_ptr, res_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.load(
            res_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
            other=0.0
        ).to(tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def matmul_small_plain(
    inp_ptr, weight_ptr, out_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w = tl.load(
                weight_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            acc = tl.dot(x, w, acc)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            acc.to(tl.bfloat16),
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def scaled_matmul_gate_silu_fixed(
    inp_ptr, inp_norm_factor_ptr, weight1_ptr, weight2_ptr, out_ptr,
    seq_len: tl.constexpr, features: tl.constexpr, hidden: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    grid_i = tl.cdiv(seq_len, BLOCK_SIZE_N)
    grid_j = tl.cdiv(hidden, BLOCK_SIZE_M)
    for p in range(pid, grid_i * grid_j, psize):
        i = (p // grid_j) * BLOCK_SIZE_N
        j = (p % grid_j) * BLOCK_SIZE_M
        factor = tl.load(
            inp_norm_factor_ptr + i + tl.arange(0, BLOCK_SIZE_N),
            mask=i + tl.arange(0, BLOCK_SIZE_N) < seq_len, other=1.0
        ).to(tl.float32)
        acc1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
        for k in range(0, features, BLOCK_SIZE_K):
            x = tl.load(
                inp_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * features + (k + tl.arange(0, BLOCK_SIZE_K))[None, :],
                mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((k + tl.arange(0, BLOCK_SIZE_K))[None, :] < features),
                other=0.0
            )
            w1 = tl.load(
                weight1_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            w2 = tl.load(
                weight2_ptr + (k + tl.arange(0, BLOCK_SIZE_K))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
                mask=((k + tl.arange(0, BLOCK_SIZE_K))[:, None] < features) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden),
                other=0.0
            )
            acc1 = tl.dot(x, w1, acc1)
            acc2 = tl.dot(x, w2, acc2)
        acc1 = acc1 * factor[:, None]
        acc2 = acc2 * factor[:, None]
        result = (acc1 * tl.sigmoid(acc1) * acc2).to(tl.bfloat16)
        tl.store(
            out_ptr + (i + tl.arange(0, BLOCK_SIZE_N))[:, None] * hidden + (j + tl.arange(0, BLOCK_SIZE_M))[None, :],
            result,
            mask=((i + tl.arange(0, BLOCK_SIZE_N))[:, None] < seq_len) & ((j + tl.arange(0, BLOCK_SIZE_M))[None, :] < hidden)
        )

@triton.jit
def prefix_scaled_qknorm_rope_split_kernel(
    norm_factor_ptr,
    qkv_ptr,
    q_norm_w_ptr, k_norm_w_ptr,
    rope_ptr,
    q_ptr, k_ptr, v_ptr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    total_out_dim: tl.constexpr,
    q_stride_row: tl.constexpr,
    k_stride_head: tl.constexpr, k_stride_row: tl.constexpr,
    v_stride_head: tl.constexpr, v_stride_row: tl.constexpr,
):
    pid = tl.program_id(0)
    total_heads = num_q_heads + num_kv_heads + num_kv_heads
    half_dim: tl.constexpr = head_dim // 2

    total_work = seq_len * total_heads
    for idx in range(pid, total_work, tl.num_programs(0)):
        row = idx // total_heads
        head_idx = idx % total_heads

        rms_factor = tl.load(norm_factor_ptr + row).to(tl.float32)

        head_offset = head_idx * head_dim
        offs = tl.arange(0, head_dim)
        val = tl.load(qkv_ptr + row * total_out_dim + head_offset + offs).to(tl.float32)
        val = val * rms_factor

        if head_idx < num_q_heads:
            norm_w = tl.load(q_norm_w_ptr + offs)
            rms_sq = tl.sum(val * val) / head_dim
            rms_inv = 1.0 / tl.sqrt(rms_sq + 1e-6)
            val = val * rms_inv * norm_w

            half_offs = tl.arange(0, half_dim)
            even = half_offs * 2
            odd = even + 1
            x0 = tl.load(qkv_ptr + row * total_out_dim + head_offset + even).to(tl.float32)
            x1 = tl.load(qkv_ptr + row * total_out_dim + head_offset + odd).to(tl.float32)
            x0 = x0 * rms_factor * rms_inv * tl.load(q_norm_w_ptr + even)
            x1 = x1 * rms_factor * rms_inv * tl.load(q_norm_w_ptr + odd)
            r_cos = tl.load(rope_ptr + row * head_dim + even).to(tl.float32)
            r_sin = tl.load(rope_ptr + row * head_dim + odd).to(tl.float32)
            tl.store(
                q_ptr + row * q_stride_row + head_idx * head_dim + even,
                (x0 * r_cos - x1 * r_sin).to(tl.bfloat16),
            )
            tl.store(
                q_ptr + row * q_stride_row + head_idx * head_dim + odd,
                (x1 * r_cos + x0 * r_sin).to(tl.bfloat16),
            )

        elif head_idx < num_q_heads + num_kv_heads:
            norm_w = tl.load(k_norm_w_ptr + offs)
            rms_sq = tl.sum(val * val) / head_dim
            rms_inv = 1.0 / tl.sqrt(rms_sq + 1e-6)
            val = val * rms_inv * norm_w

            kv_head = head_idx - num_q_heads
            half_offs = tl.arange(0, half_dim)
            even = half_offs * 2
            odd = even + 1
            x0 = tl.load(qkv_ptr + row * total_out_dim + head_offset + even).to(tl.float32)
            x1 = tl.load(qkv_ptr + row * total_out_dim + head_offset + odd).to(tl.float32)
            x0 = x0 * rms_factor * rms_inv * tl.load(k_norm_w_ptr + even)
            x1 = x1 * rms_factor * rms_inv * tl.load(k_norm_w_ptr + odd)
            r_cos = tl.load(rope_ptr + row * head_dim + even).to(tl.float32)
            r_sin = tl.load(rope_ptr + row * head_dim + odd).to(tl.float32)
            tl.store(
                k_ptr + kv_head * k_stride_head + row * k_stride_row + even,
                (x0 * r_cos - x1 * r_sin).to(tl.bfloat16),
            )
            tl.store(
                k_ptr + kv_head * k_stride_head + row * k_stride_row + odd,
                (x1 * r_cos + x0 * r_sin).to(tl.bfloat16),
            )
        else:
            kv_head = head_idx - num_q_heads - num_kv_heads
            tl.store(v_ptr + kv_head * v_stride_head + row * v_stride_row + offs,
                     val.to(tl.bfloat16))

@triton.jit
def prefix_mm_kernel(
    inp_ptr, weight_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    GROUP_M: tl.constexpr = 8
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % (GROUP_M * num_n)) // group_sz

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)

    a_ptrs = inp_ptr + m_offs[:, None] * K + k_offs[None, :]
    b_ptrs = weight_ptr + k_offs[:, None] * N + n_offs[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offs) < K
        a = tl.load(a_ptrs, mask=(m_offs[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (n_offs[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16),
             mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))

@triton.jit
def prefix_addres_kernel(
    inp_ptr, weight_ptr, res_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    GROUP_M: tl.constexpr = 8
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % (GROUP_M * num_n)) // group_sz

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    m_mask = m_offs < M
    n_mask = n_offs < N

    r_ptrs = res_ptr + m_offs[:, None] * N + n_offs[None, :]
    acc = tl.load(r_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    a_ptrs = inp_ptr + m_offs[:, None] * K + k_offs[None, :]
    b_ptrs = weight_ptr + k_offs[:, None] * N + n_offs[None, :]

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offs) < K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    tl.store(r_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def prefix_gate_silu_kernel(
    norm_factor_ptr, inp_ptr, gate_w_ptr, up_w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    GROUP_M: tl.constexpr = 8
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % (GROUP_M * num_n)) // group_sz

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    m_mask = m_offs < M
    n_mask = n_offs < N

    factor = tl.load(norm_factor_ptr + m_offs, mask=m_mask, other=1.0).to(tl.float32)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = inp_ptr + m_offs[:, None] * K + k_offs[None, :]
    g_ptrs = gate_w_ptr + k_offs[:, None] * N + n_offs[None, :]
    u_ptrs = up_w_ptr + k_offs[:, None] * N + n_offs[None, :]

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offs) < K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        g = tl.load(g_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        u = tl.load(u_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc_gate = tl.dot(a, g, acc_gate)
        acc_up = tl.dot(a, u, acc_up)
        a_ptrs += BLOCK_K
        g_ptrs += BLOCK_K * N
        u_ptrs += BLOCK_K * N

    acc_gate = acc_gate * factor[:, None]
    acc_up = acc_up * factor[:, None]
    result = acc_gate * tl.sigmoid(acc_gate) * acc_up

    c_ptrs = out_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, result.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def gelu_approx(x):
    GELU_COEFF: tl.constexpr = 0.044715
    SCALE: tl.constexpr = 1.5957691216057308
    x_sq = x * x
    inner = SCALE * x * (1.0 + GELU_COEFF * x_sq)
    return x * tl.sigmoid(inner)

@triton.jit
def vit_layernorm_stats_kernel(
    inp_ptr, mean_ptr, rstd_ptr, rows,
    FEATURES: tl.constexpr = 1024,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    x = tl.load(inp_ptr + pid * FEATURES + offs, mask=offs < FEATURES, other=0.0).to(tl.float32)
    mean = tl.sum(x) / FEATURES
    var = tl.sum(x * x) / FEATURES - mean * mean
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, 1.0 / tl.sqrt(var + 1e-5))

@triton.jit
def vit_ln_mm_kernel(
    a_ptr, b_ptr, c_ptr,
    mean_ptr, rstd_ptr, col_sum_ptr, bias_ptr,
    M, N, K,  HAS_GELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    GROUP_M: tl.constexpr = 8
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % (GROUP_M * num_n)) // group_sz
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    m_mask = m_offs < M
    n_mask = n_offs < N
    a_ptrs = a_ptr + m_offs[:, None] * K + k_offs[None, :]
    b_ptrs = b_ptr + k_offs[:, None] * N + n_offs[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offs) < K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    mean = tl.load(mean_ptr + m_offs, mask=m_mask, other=0.0)
    rstd = tl.load(rstd_ptr + m_offs, mask=m_mask, other=1.0)
    col_sum = tl.load(col_sum_ptr + n_offs, mask=n_mask, other=0.0)
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc = rstd[:, None] * (acc - mean[:, None] * col_sum[None, :]) + bias[None, :]
    if HAS_GELU:
        acc = gelu_approx(acc)
    c_ptrs = c_ptr + m_offs[:, None] * N + n_offs[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def vit_bias_addres_kernel(
    inp_ptr, weight_ptr, bias_ptr, res_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    GROUP_M: tl.constexpr = 8
    group_id = pid // (GROUP_M * num_n)
    first_m = group_id * GROUP_M
    group_sz = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_sz)
    pid_n = (pid % (GROUP_M * num_n)) // group_sz
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N
    mn_mask = m_mask[:, None] & n_mask[None, :]
    r_ptrs = res_ptr + m_offs[:, None] * N + n_offs[None, :]
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc = tl.load(r_ptrs, mask=mn_mask, other=0.0).to(tl.float32) + bias[None, :]
    k_offs = tl.arange(0, BLOCK_K)
    a_ptrs = inp_ptr + m_offs[:, None] * K + k_offs[None, :]
    b_ptrs = weight_ptr + k_offs[:, None] * N + n_offs[None, :]
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offs) < K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    tl.store(r_ptrs, acc.to(tl.bfloat16), mask=mn_mask)

@triton.jit
def im2col_kernel_nhwc(
    input_ptr, col_ptr,
    batch_size, img_h, img_w, in_c,
    out_h, out_w, kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    k_total = kernel_size * kernel_size * in_c
    total_elements = batch_size * out_h * out_w * k_total
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    k = offs % k_total
    rem = offs // k_total
    out_x = rem % out_w
    rem = rem // out_w
    out_y = rem % out_h
    batch_idx = rem // out_h
    ic = k % in_c
    k_rem = k // in_c
    kx = k_rem % kernel_size
    ky = k_rem // kernel_size
    in_y = out_y * stride - padding + ky
    in_x = out_x * stride - padding + kx
    in_idx = ((batch_idx * img_h + in_y) * img_w + in_x) * in_c + ic
    in_mask = mask & (in_y >= 0) & (in_y < img_h) & (in_x >= 0) & (in_x < img_w)
    val = tl.load(input_ptr + in_idx, mask=in_mask, other=0.0)
    tl.store(col_ptr + offs, val, mask=mask)

@triton.jit
def _layernorm_inplace_kernel(
    x_ptr, mean_ptr, rstd_ptr, w_ptr, b_ptr,
    rows, FEATURES: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    psize = tl.num_programs(0)
    for row in range(pid, rows, psize):
        mean = tl.load(mean_ptr + row)
        rstd = tl.load(rstd_ptr + row)
        for j in range(0, FEATURES, BLOCK_SIZE):
            offs = j + tl.arange(0, BLOCK_SIZE)
            mask = offs < FEATURES
            x = tl.load(x_ptr + row * FEATURES + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(x_ptr + row * FEATURES + offs, y.to(tl.bfloat16), mask=mask)

@torch.compile
def _apply_rotary_emb_vision_precomputed(cos, sin, t):
    rot_dim = cos.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * cos) + (_rotate_half_vision(t_rot) * sin)
    return torch.cat([t_rot, t_pass], dim=-1).to(t.dtype)


@torch.compile
def vit_qkv_attention(proj, freqs_cos, freqs_sin, num_images, num_heads, head_dim):
    q, k, v = proj.chunk(3, dim=-1)
    seq_len = proj.shape[1]
    q = q.view(num_images, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(num_images, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(num_images, seq_len, num_heads, head_dim).transpose(1, 2)
    q = _apply_rotary_emb_vision_precomputed(freqs_cos, freqs_sin, q)
    k = _apply_rotary_emb_vision_precomputed(freqs_cos, freqs_sin, k)
    attn = F.scaled_dot_product_attention(q, k, v, scale=head_dim ** -0.5)
    return attn.transpose(1, 2).contiguous().view(num_images, seq_len, -1)

def vit_fused_ln_mm(x, w_fused, col_sum, bias_fused, out_buf, mean_buf, rstd_buf, has_gelu=False):
    M, K = x.shape
    N = w_fused.shape[1]
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    vit_layernorm_stats_kernel[(M,)](x, mean_buf, rstd_buf, M)
    vit_ln_mm_kernel[grid](x, w_fused, out_buf, mean_buf, rstd_buf, col_sum, bias_fused,
                           M, N, K, HAS_GELU=has_gelu, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

def vit_bias_addres(inp, weight, bias, res):
    M, K = inp.shape
    N = weight.shape[1]
    cfg = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=3, num_warps=4)
    grid = (triton.cdiv(M, cfg['BLOCK_M']) * triton.cdiv(N, cfg['BLOCK_N']),)
    vit_bias_addres_kernel[grid](inp, weight, bias, res, M, N, K, **cfg)

def _triton_mm(a, b, out=None):
    M, K = a.shape
    N = b.shape[1]
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    if out is None:
        out = torch.empty(M, N, dtype=a.dtype, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    prefix_mm_kernel[grid](a, b, out, M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return out

def conv2d_im2col_gemm(input_nhwc, weight_hwio, bias, stride, padding, col_buf=None, mm_buf=None):
    batch, img_h, img_w, in_c = input_nhwc.shape
    k_h, k_w, _, out_c = weight_hwio.shape
    out_h = (img_h + 2 * padding - k_h) // stride + 1
    out_w = (img_w + 2 * padding - k_w) // stride + 1
    k_total = k_h * k_w * in_c
    m = batch * out_h * out_w
    if col_buf is not None:
        col = col_buf[:m, :k_total]
    else:
        col = torch.empty(m, k_total, dtype=input_nhwc.dtype, device=input_nhwc.device)
    total = m * k_total
    block = 1024
    im2col_kernel_nhwc[(triton.cdiv(total, block),)](
        input_nhwc, col,
        batch, img_h, img_w, in_c,
        out_h, out_w, k_h, stride, padding,
        BLOCK_SIZE=block,
    )
    w_2d = weight_hwio.reshape(k_total, out_c)
    if mm_buf is not None:
        output = mm_buf[:m, :out_c]
        _triton_mm(col, w_2d, out=output)
    else:
        output = _triton_mm(col, w_2d)
    if bias is not None:
        output.add_(bias)
    return output.reshape(batch, out_h, out_w, out_c)

def _apply_layernorm_inplace(x_2d, mean_buf, rstd_buf, w, b):
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    _layernorm_inplace_kernel[(M,)](x_2d, mean_buf, rstd_buf, w, b, M, K, BLOCK_SIZE=1024)

def _rotate_half_vision(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

def compute_2d_freqs_cache(grid_h=52, grid_w=52, head_dim=64, theta=10000.0, device='cuda'):
    dim_per_axis = head_dim // 2
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, dim_per_axis, 2, device=device)[: (dim_per_axis // 2)].float() / dim_per_axis
    ))
    gh = torch.arange(grid_h, dtype=torch.float, device=device) + 1
    gw = torch.arange(grid_w, dtype=torch.float, device=device) + 1
    def _f(t):
        f = torch.einsum("..., f -> ... f", t, inv_freq)
        return f.repeat_interleave(2, dim=-1)
    fh = _f(gh)[:, None].expand(grid_h, grid_w, -1)
    fw = _f(gw)[None, :].expand(grid_h, grid_w, -1)
    freqs = torch.cat([fw, fh], dim=-1).reshape(grid_h * grid_w, -1)
    freqs = torch.cat([torch.zeros(1, freqs.shape[-1], device=device), freqs], dim=0)
    freqs = freqs[None, None, ...].to(torch.bfloat16)
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    return freqs_cos, freqs_sin

def run_vision_forward(weights, images, input_ids, freqs_cos, freqs_sin, vit_bufs, output_buf=None):
    num_images = images.shape[0]
    grid_h, grid_w, width = 52, 52, 1024
    num_heads, head_dim = 16, 64
    seq_per_img = grid_h * grid_w + 1
    x = images.reshape(num_images, 3, grid_h, 14, grid_w, 14)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(num_images * grid_h * grid_w, -1)
    x = _triton_mm(x, weights['vision_conv1_w_t'],
                   out=vit_bufs.get('conv1_buf'))
    x = x.reshape(num_images, grid_h * grid_w, width)
    x_3d = vit_bufs['x_2d'].view(num_images, seq_per_img, width)
    x_3d[:, 0, :] = weights['vision_class_embedding']
    x_3d[:, 1:, :] = x
    pos_emb = weights['vision_pos_emb'][:seq_per_img, :]
    x_3d.add_(pos_emb.unsqueeze(0))
    x_2d = x_3d.view(-1, width)
    M = num_images * seq_per_img
    mean_buf = vit_bufs['mean']
    rstd_buf = vit_bufs['rstd']
    vit_layernorm_stats_kernel[(M,)](x_2d, mean_buf, rstd_buf, M)
    ln_w = weights['vision_ln_pre_w']
    ln_b = weights['vision_ln_pre_b']
    _apply_layernorm_inplace(x_2d, mean_buf, rstd_buf, ln_w, ln_b)

    qkv_buf = vit_bufs['qkv']
    fc_buf = vit_bufs['fc']

    for i in range(23):
        vit_fused_ln_mm(x_2d, weights['vision_fused_qkv_w'][i],
                        weights['vision_qkv_col_sum'][i],
                        weights['vision_fused_qkv_b'][i],
                        qkv_buf, mean_buf, rstd_buf)
        proj = qkv_buf.view(num_images, -1, 3072)
        attn = vit_qkv_attention(proj, freqs_cos, freqs_sin, num_images, num_heads, head_dim)
        attn_2d = attn.reshape(-1, width)
        vit_bias_addres(attn_2d, weights['vision_out_proj_w'][i],
                        weights['vision_out_proj_b'][i], x_2d)
        vit_fused_ln_mm(x_2d, weights['vision_fused_fc_w'][i],
                        weights['vision_fc_col_sum'][i],
                        weights['vision_fused_fc_b'][i],
                        fc_buf, mean_buf, rstd_buf, has_gelu=True)
        vit_bias_addres(fc_buf, weights['vision_proj_w'][i],
                        weights['vision_proj_b'][i], x_2d)

    x = x_2d.reshape(num_images, -1, width)
    x = x[:, 1:, :].contiguous()
    x = x.reshape(num_images, grid_h, grid_w, width)
    x = conv2d_im2col_gemm(x, weights['vision_ds1_w'], weights['vision_ds1_b'],
                           stride=2, padding=1, col_buf=vit_bufs.get('col1'),
                           mm_buf=vit_bufs.get('mm1'))
    x = conv2d_im2col_gemm(x, weights['vision_ds2_w'], weights['vision_ds2_b'],
                           stride=2, padding=1, col_buf=vit_bufs.get('col2'),
                           mm_buf=vit_bufs.get('mm2'))
    proj_input = x.reshape(-1, 4096)
    proj_w_t = weights['vision_projector_w_t']
    if 'proj_buf' in vit_bufs:
        image_tokens = _triton_mm(proj_input, proj_w_t, out=vit_bufs['proj_buf'][:proj_input.shape[0]])
    else:
        image_tokens = _triton_mm(proj_input, proj_w_t)

    if output_buf is not None:
        n_img = image_tokens.shape[0]
        output_buf[:n_img].copy_(image_tokens)
        if input_ids is not None:
            lang_tokens = F.embedding(input_ids, weights['vision_embed_tokens_w'])
            output_buf[n_img:n_img + lang_tokens.shape[0]].copy_(lang_tokens)
        return output_buf

    if input_ids is not None:
        lang_tokens = F.embedding(input_ids, weights['vision_embed_tokens_w'])
        return torch.cat([image_tokens, lang_tokens], dim=0)
    return image_tokens

def embed_suffix_step(weights, buffers, step):
    seq_len = buffers['diffusion_noise'].shape[0]
    matmul_small_bias_silu[((seq_len + 31) // 32) * (1024 // 32),](
        buffers['diffusion_noise'],
        weights['decoder_action_fused_in_proj_w'],
        buffers['decoder_x_buf'],
        weights['decoder_action_fused_time_biases'][step],
        seq_len=seq_len, features=32, hidden=1024,
        BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=32
    )
    matmul_small_bias[((seq_len + 15) // 16) * (1024 // 32),](
        buffers['decoder_x_buf'],
        weights['decoder_action_mlp_w'],
        buffers['decoder_x'],
        weights['decoder_action_mlp_b'],
        seq_len=seq_len, features=1024, hidden=1024,
        BLOCK_SIZE_N=16, BLOCK_SIZE_M=32, BLOCK_SIZE_K=256
    )

def decoder_layer(weights, buffers, layer_idx, prefix_len, total_keys):
    seq_len = buffers['decoder_x'].shape[0]
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    total_heads = num_q_heads + 2 * num_kv_heads
    hidden = 1024
    intermediate = 1536
    total_qkv = total_heads * head_dim
    rmsnorm_factor_kernel[(seq_len,)](
        buffers['decoder_x'], buffers['decoder_norm_factor'],
        seq_len, hidden,         eps=1e-6, BLOCK_SIZE=1024
    )
    GRID_QKV = ((seq_len + 31) // 32) * ((total_qkv + 63) // 64)
    matmul_small_plain[(GRID_QKV,)](
        buffers['decoder_x'], weights['decoder_attn_qkv_w'][layer_idx],
        buffers['decoder_qkv'],
        seq_len=seq_len, features=hidden, hidden=total_qkv,
        BLOCK_SIZE_N=32, BLOCK_SIZE_M=64, BLOCK_SIZE_K=128,
    )
    prefix_scaled_qknorm_rope_split_kernel[(seq_len * total_heads,)](
        buffers['decoder_norm_factor'],
        buffers['decoder_qkv'],
        weights['decoder_q_norm_w'][layer_idx],
        weights['decoder_k_norm_w'][layer_idx],
        buffers['decoder_rope_weights'],
        buffers['decoder_Q'],
        buffers['kv_K'][layer_idx, :, prefix_len:prefix_len + seq_len, :],
        buffers['kv_V'][layer_idx, :, prefix_len:prefix_len + seq_len, :],
        seq_len=seq_len,
        head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        total_out_dim=total_qkv,
        q_stride_row=num_q_heads * head_dim,
        k_stride_head=total_keys * head_dim, k_stride_row=head_dim,
        v_stride_head=total_keys * head_dim,         v_stride_row=head_dim,
    )
    Q_4d = buffers['decoder_Q'].view(seq_len, num_q_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
    K_full = buffers['kv_K'][layer_idx].unsqueeze(0)
    V_full = buffers['kv_V'][layer_idx].unsqueeze(0)
    gqa_factor = num_q_heads // num_kv_heads
    K_full = K_full.repeat_interleave(gqa_factor, dim=1)
    V_full = V_full.repeat_interleave(gqa_factor, dim=1)
    attn = F.scaled_dot_product_attention(
        Q_4d, K_full, V_full, attn_mask=buffers['decoder_attn_mask']
    )
    attn_flat = attn.squeeze(0).permute(1, 0, 2).reshape(seq_len, -1)
    GRID_O = ((seq_len + 15) // 16) * ((hidden + 31) // 32)
    matmul_small_res[(GRID_O,)](
        attn_flat, weights['decoder_attn_o_w'][layer_idx],
        buffers['decoder_x'], buffers['decoder_x'],
        seq_len=seq_len, features=num_q_heads * head_dim, hidden=hidden,
        BLOCK_SIZE_N=16, BLOCK_SIZE_M=32, BLOCK_SIZE_K=256,
    )
    rmsnorm_factor_kernel[(seq_len,)](
        buffers['decoder_x'], buffers['decoder_norm_factor'],
        seq_len, hidden,         eps=1e-6, BLOCK_SIZE=1024
    )
    GRID_GS = ((seq_len + 15) // 16) * ((intermediate + 63) // 64)
    scaled_matmul_gate_silu_fixed[(GRID_GS,)](
        buffers['decoder_x'],
        buffers['decoder_norm_factor'],
        weights['decoder_ffn_gate_w'][layer_idx],
        weights['decoder_ffn_up_w'][layer_idx],
        buffers['decoder_hidden'],
        seq_len=seq_len, features=hidden, hidden=intermediate,
        BLOCK_SIZE_N=16, BLOCK_SIZE_M=64, BLOCK_SIZE_K=64,
    )
    GRID_D = ((seq_len + 31) // 32) * ((hidden + 31) // 32)
    matmul_small_res[(GRID_D,)](
        buffers['decoder_hidden'], weights['decoder_ffn_down_w'][layer_idx],
        buffers['decoder_x'], buffers['decoder_x'],
        seq_len=seq_len, features=intermediate, hidden=hidden,
        BLOCK_SIZE_N=32, BLOCK_SIZE_M=32, BLOCK_SIZE_K=128,
    )

def transformer_decoder(weights, buffers, prefix_len, total_keys, num_layers=28, diffusion_steps=10):
    seq_len = buffers['decoder_x'].shape[0]
    rmsnorm_grid = min(seq_len, 170)
    for step in range(diffusion_steps):
        embed_suffix_step(weights, buffers, step)

        for layer_idx in range(num_layers):
            decoder_layer(weights, buffers, layer_idx, prefix_len, total_keys)

        rmsnorm_factor_kernel[(rmsnorm_grid,)](
            buffers['decoder_x'], buffers['decoder_norm_factor'],
            seq_len, 1024, eps=1e-6, BLOCK_SIZE=1024
        )
        euler_step_kernel[(seq_len,)](
            buffers['decoder_x'], buffers['decoder_norm_factor'],
            weights['decoder_action_fused_out_proj_w'],
            weights['decoder_action_fused_out_proj_b'],
            buffers['diffusion_noise'],
            seq_len=seq_len, features=1024, out_dim=32, BLOCK_K=512,
        )

def prefix_layer_forward(weights, buffers, layer_idx, prefix_len, total_keys):
    seq_len = prefix_len
    head_dim = 128
    num_q_heads = 16
    num_kv_heads = 8
    total_heads = num_q_heads + 2 * num_kv_heads
    llm_hidden = 2048
    llm_intermediate = 6144
    total_qkv = total_heads * head_dim
    rmsnorm_factor_kernel[(seq_len,)](
        buffers['prefix_x'], buffers['prefix_norm_factor'],
        seq_len, llm_hidden, eps=1e-6, BLOCK_SIZE=1024
    )

    cfg_mm = dict(BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_stages=4, num_warps=4)
    grid_qkv = (triton.cdiv(seq_len, cfg_mm['BLOCK_M']) * triton.cdiv(total_qkv, cfg_mm['BLOCK_N']),)
    prefix_mm_kernel[grid_qkv](
        buffers['prefix_x'], weights['llm_attn_qkv_w'][layer_idx],
        buffers['prefix_qkv'],
        seq_len, total_qkv, llm_hidden, **cfg_mm
    )

    num_programs = min(seq_len * total_heads, 1024)
    prefix_scaled_qknorm_rope_split_kernel[(num_programs,)](
        buffers['prefix_norm_factor'],
        buffers['prefix_qkv'],
        weights['llm_q_norm_w'][layer_idx],
        weights['llm_k_norm_w'][layer_idx],
        buffers['prefix_rope_weights'],
        buffers['prefix_Q'],
        buffers['kv_K'][layer_idx, :, :prefix_len, :],
        buffers['kv_V'][layer_idx, :, :prefix_len, :],
        seq_len=seq_len,
        head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        total_out_dim=total_qkv,
        q_stride_row=num_q_heads * head_dim,
        k_stride_head=total_keys * head_dim, k_stride_row=head_dim,
        v_stride_head=total_keys * head_dim, v_stride_row=head_dim,
    )

    Q = buffers['prefix_Q'].view(seq_len, num_q_heads, head_dim).permute(1, 0, 2).unsqueeze(0)
    K_layer = buffers['kv_K'][layer_idx, :, :prefix_len, :].unsqueeze(0)
    V_layer = buffers['kv_V'][layer_idx, :, :prefix_len, :].unsqueeze(0)
    gqa_factor = num_q_heads // num_kv_heads
    K_layer = K_layer.repeat_interleave(gqa_factor, dim=1)
    V_layer = V_layer.repeat_interleave(gqa_factor, dim=1)
    attn = F.scaled_dot_product_attention(
        Q, K_layer, V_layer, attn_mask=buffers['prefix_attn_mask']
    )
    attn_flat = attn.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, -1)

    cfg_ar = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_stages=3, num_warps=4)
    grid_o = (triton.cdiv(seq_len, cfg_ar['BLOCK_M']) * triton.cdiv(llm_hidden, cfg_ar['BLOCK_N']),)
    prefix_addres_kernel[grid_o](
        attn_flat, weights['llm_attn_o_w'][layer_idx],
        buffers['prefix_x'],
        seq_len, llm_hidden, llm_hidden, **cfg_ar
    )

    rmsnorm_factor_kernel[(seq_len,)](
        buffers['prefix_x'], buffers['prefix_norm_factor'],
        seq_len, llm_hidden, eps=1e-6, BLOCK_SIZE=1024
    )

    cfg_gs = dict(BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, num_stages=3, num_warps=8)
    grid_gs = (triton.cdiv(seq_len, cfg_gs['BLOCK_M']) * triton.cdiv(llm_intermediate, cfg_gs['BLOCK_N']),)
    prefix_gate_silu_kernel[grid_gs](
        buffers['prefix_norm_factor'],
        buffers['prefix_x'],
        weights['llm_ffn_gate_w'][layer_idx],
        weights['llm_ffn_up_w'][layer_idx],
        buffers['prefix_hidden'],
        seq_len, llm_intermediate, llm_hidden, **cfg_gs
    )

    grid_d = (triton.cdiv(seq_len, cfg_ar['BLOCK_M']) * triton.cdiv(llm_hidden, cfg_ar['BLOCK_N']),)
    prefix_addres_kernel[grid_d](
        buffers['prefix_hidden'], weights['llm_ffn_down_w'][layer_idx],
        buffers['prefix_x'],
        seq_len, llm_hidden, llm_intermediate, **cfg_ar
    )

def run_prefix_forward(weights, buffers, prefix_len, total_keys, num_layers=28):
    for layer_idx in range(num_layers):
        prefix_layer_forward(weights, buffers, layer_idx, prefix_len, total_keys)

class DM0RealtimeKernelSpec:
    image_size: int = 728
    patch_size: int = 14
    llm_vocab_size: int = 152701
    llm_rope_theta: float = 1000000.0
    chunk_size: int = 50
    diffusion_steps: int = 10

class DM0Inference:
    def __init__(
        self,
        checkpoint,
        num_images=3,
        max_lang_len=100,
        diffusion_steps=10,
        device="cuda",
    ):
        device_obj = torch.device(device)
        if device_obj.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                "DM0 realtime backend requires a CUDA GPU because it uses "
                "Triton kernels and CUDA Graph capture."
            )
        spec = DM0RealtimeKernelSpec()
        self.device = str(device_obj)
        self.num_images = num_images
        self.max_lang_len = max_lang_len
        self.chunk_size = spec.chunk_size
        self.diffusion_steps = int(diffusion_steps or spec.diffusion_steps)
        if self.diffusion_steps <= 0:
            raise ValueError(f"diffusion_steps must be positive, got {self.diffusion_steps}")
        grid_size = spec.image_size // spec.patch_size
        tokens_per_image = (grid_size // 4) ** 2
        self.prefix_len = num_images * tokens_per_image + max_lang_len
        self.total_keys = self.prefix_len + self.chunk_size
        self.tokens_per_image = tokens_per_image
        M_vit = num_images * 2705
        col1_m = num_images * 676
        col2_m = num_images * 169
        conv1_m = num_images * 2704
        ds1_out_c = 2048
        ds2_out_c = 4096
        ds1_k = 3 * 3 * 1024
        ds2_k = 3 * 3 * ds1_out_c
        self.weights = {
            'decoder_attn_qkv_w':        torch.empty(28, 1024, 4096, dtype=torch.bfloat16, device=device),
            'decoder_q_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
            'decoder_k_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
            'decoder_attn_o_w':          torch.empty(28, 2048, 1024,       dtype=torch.bfloat16, device=device),
            'decoder_ffn_gate_w':        torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
            'decoder_ffn_up_w':          torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
            'decoder_ffn_down_w':        torch.empty(28, 1536, 1024,       dtype=torch.bfloat16, device=device),
            'decoder_action_fused_in_proj_w': torch.empty(32, 1024,         dtype=torch.bfloat16, device=device),
            'decoder_action_fused_time_biases': torch.empty(self.diffusion_steps, 1024, dtype=torch.bfloat16, device=device),
            'decoder_action_mlp_w':       torch.empty(1024, 1024,           dtype=torch.bfloat16, device=device),
            'decoder_action_mlp_b':       torch.empty(1024,                 dtype=torch.bfloat16, device=device),
            'decoder_action_fused_out_proj_w': torch.empty(1024, 32,        dtype=torch.bfloat16, device=device),
            'decoder_action_fused_out_proj_b': torch.empty(32,               dtype=torch.bfloat16, device=device),
            'llm_attn_qkv_w':            torch.empty(28, 2048, 4096,       dtype=torch.bfloat16, device=device),
            'llm_q_norm_w':              torch.empty(28, 128,               dtype=torch.bfloat16, device=device),
            'llm_k_norm_w':              torch.empty(28, 128,               dtype=torch.bfloat16, device=device),
            'llm_attn_o_w':              torch.empty(28, 2048, 2048,        dtype=torch.bfloat16, device=device),
            'llm_ffn_gate_w':            torch.empty(28, 2048, 6144,        dtype=torch.bfloat16, device=device),
            'llm_ffn_up_w':              torch.empty(28, 2048, 6144,        dtype=torch.bfloat16, device=device),
            'llm_ffn_down_w':            torch.empty(28, 6144, 2048,       dtype=torch.bfloat16, device=device),
            'vision_conv1_w_t':          torch.empty(588, 1024,            dtype=torch.bfloat16, device=device),
            'vision_class_embedding':     torch.empty(1024,                dtype=torch.bfloat16, device=device),
            'vision_pos_emb':             torch.empty(2705, 1024,            dtype=torch.bfloat16, device=device),
            'vision_ln_pre_w':            torch.empty(1024,                dtype=torch.bfloat16, device=device),
            'vision_ln_pre_b':            torch.empty(1024,                dtype=torch.bfloat16, device=device),
            'vision_fused_qkv_w':        torch.empty(23, 1024, 3072,       dtype=torch.bfloat16, device=device),
            'vision_fused_qkv_b':        torch.empty(23, 3072,             dtype=torch.float32, device=device),
            'vision_qkv_col_sum':        torch.empty(23, 3072,             dtype=torch.float32, device=device),
            'vision_out_proj_w':         torch.empty(23, 1024, 1024,         dtype=torch.bfloat16, device=device),
            'vision_out_proj_b':          torch.empty(23, 1024,             dtype=torch.bfloat16, device=device),
            'vision_fused_fc_w':         torch.empty(23, 1024, 4096,       dtype=torch.bfloat16, device=device),
            'vision_fused_fc_b':         torch.empty(23, 4096,              dtype=torch.float32, device=device),
            'vision_fc_col_sum':          torch.empty(23, 4096,              dtype=torch.float32, device=device),
            'vision_proj_w':              torch.empty(23, 4096, 1024,        dtype=torch.bfloat16, device=device),
            'vision_proj_b':              torch.empty(23, 1024,              dtype=torch.bfloat16, device=device),
            'vision_ds1_w':               torch.empty(3, 3, 1024, 2048, dtype=torch.bfloat16, device=device),
            'vision_ds1_b':               torch.empty(2048,            dtype=torch.bfloat16, device=device),
            'vision_ds2_w':               torch.empty(3, 3, 2048, 4096, dtype=torch.bfloat16, device=device),
            'vision_ds2_b':               torch.empty(4096,            dtype=torch.bfloat16, device=device),
            'vision_projector_w_t':       torch.empty(4096, 2048,           dtype=torch.bfloat16, device=device),
            'vision_embed_tokens_w':      torch.empty(spec.llm_vocab_size, 2048, dtype=torch.bfloat16, device=device),
        }
        self.buffers = {
            'images':                    torch.zeros(num_images, 3, 728, 728, dtype=torch.bfloat16, device=device),
            'input_ids':                 torch.zeros(max_lang_len, dtype=torch.long, device=device),
            'diffusion_noise':            torch.zeros(50, 32, dtype=torch.bfloat16, device=device),
            'vision_x_2d':                torch.empty(M_vit, 1024, dtype=torch.bfloat16, device=device),
            'vision_mean':                torch.empty(M_vit, dtype=torch.float32, device=device),
            'vision_rstd':                torch.empty(M_vit, dtype=torch.float32, device=device),
            'vision_qkv':                 torch.empty(M_vit, 3072, dtype=torch.bfloat16, device=device),
            'vision_fc':                  torch.empty(M_vit, 4096, dtype=torch.bfloat16, device=device),
            'vision_conv1_buf':           torch.empty(conv1_m, 1024, dtype=torch.bfloat16, device=device),
            'vision_col1':                torch.empty(col1_m, ds1_k, dtype=torch.bfloat16, device=device),
            'vision_col2':                torch.empty(col2_m, ds2_k, dtype=torch.bfloat16, device=device),
            'vision_mm1':                 torch.empty(col1_m, ds1_out_c, dtype=torch.bfloat16, device=device),
            'vision_mm2':                 torch.empty(col2_m, ds2_out_c, dtype=torch.bfloat16, device=device),
            'vision_proj_buf':            torch.empty(col2_m, 2048, dtype=torch.bfloat16, device=device),
            'prefix_x':                   torch.empty(self.prefix_len, 2048, dtype=torch.bfloat16, device=device),
            'prefix_qkv':                 torch.empty(self.prefix_len, 4096, dtype=torch.bfloat16, device=device),
            'prefix_Q':                   torch.empty(self.prefix_len, 2048, dtype=torch.bfloat16, device=device),
            'prefix_hidden':               torch.empty(self.prefix_len, 6144, dtype=torch.bfloat16, device=device),
            'prefix_rope_weights':         torch.empty(self.prefix_len, 128, dtype=torch.bfloat16, device=device),
            'prefix_padding_mask':          torch.empty(self.prefix_len, dtype=torch.bool, device=device),
            'prefix_attn_mask':             torch.empty(1, 1, self.prefix_len, self.prefix_len, dtype=torch.bfloat16, device=device),
            'prefix_norm_factor':          torch.empty(self.prefix_len, dtype=torch.float32, device=device),
            'decoder_x':                   torch.empty(50, 1024, dtype=torch.bfloat16, device=device),
            'decoder_x_buf':               torch.empty(50, 1024, dtype=torch.bfloat16, device=device),
            'decoder_Q':                   torch.empty(50, 2048, dtype=torch.bfloat16, device=device),
            'decoder_hidden':              torch.empty(50, 1536, dtype=torch.bfloat16, device=device),
            'decoder_norm_factor':         torch.empty(50, dtype=torch.float32, device=device),
            'decoder_rope_weights':        torch.empty(50, 128, dtype=torch.bfloat16, device=device),
            'decoder_attn_mask':            torch.empty(1, 1, 50, self.total_keys, dtype=torch.bfloat16, device=device),
            'decoder_qkv':                 torch.empty(50, 4096, dtype=torch.bfloat16, device=device),
            'kv_K':                       torch.empty(28, 8, self.total_keys, 128, dtype=torch.bfloat16, device=device),
            'kv_V':                       torch.empty(28, 8, self.total_keys, 128, dtype=torch.bfloat16, device=device),
            'output':                     torch.empty(50, 32, dtype=torch.bfloat16, device=device),
        }
        self.vit_bufs = {
            'mean': self.buffers['vision_mean'],
            'rstd': self.buffers['vision_rstd'],
            'qkv': self.buffers['vision_qkv'],
            'fc': self.buffers['vision_fc'],
            'col1': self.buffers['vision_col1'],
            'col2': self.buffers['vision_col2'],
            'mm1': self.buffers['vision_mm1'],
            'mm2': self.buffers['vision_mm2'],
            'proj_buf': self.buffers['vision_proj_buf'],
            'conv1_buf': self.buffers['vision_conv1_buf'],
            'x_2d': self.buffers['vision_x_2d'],
        }
        for k, v in checkpoint.items():
            if k not in self.weights:
                continue
            if tuple(v.shape) != tuple(self.weights[k].shape):
                raise ValueError(
                    f"Realtime weight {k!r} has shape {tuple(v.shape)}, "
                    f"expected {tuple(self.weights[k].shape)}. "
                    "Check that realtime_diffusion_steps matches the converted checkpoint."
                )
            self.weights[k].copy_(v)
        self.inv_freq = 1.0 / (
            spec.llm_rope_theta
            ** (torch.arange(0, 128, 2, dtype=torch.float32, device=self.device) / 128)
        )
        self._set_default_masks()
        self._compute_rope_weights()
        self.freqs_cos, self.freqs_sin = compute_2d_freqs_cache(device=device)
        self.graph = torch.cuda.CUDAGraph()
        self._record_graph()

    def _update_rope_weights(self, prefix_positions, decoder_positions):
        prefix_positions = prefix_positions.to(device=self.device, dtype=torch.float32)
        decoder_positions = decoder_positions.to(device=self.device, dtype=torch.float32)
        angles = prefix_positions[:, None] * self.inv_freq[None, :]
        cos_vals = torch.cos(angles).to(torch.bfloat16)
        sin_vals = torch.sin(angles).to(torch.bfloat16)
        rope = torch.stack([cos_vals, sin_vals], dim=2).reshape(self.prefix_len, 128)
        self.buffers['prefix_rope_weights'].copy_(rope)

        angles = decoder_positions[:, None] * self.inv_freq[None, :]
        cos_vals = torch.cos(angles).to(torch.bfloat16)
        sin_vals = torch.sin(angles).to(torch.bfloat16)
        rope = torch.stack([cos_vals, sin_vals], dim=2).reshape(self.chunk_size, 128)
        self.buffers['decoder_rope_weights'].copy_(rope)

    def _compute_rope_weights(self):
        positions = torch.arange(self.prefix_len, dtype=torch.float32, device=self.device)
        decoder_positions = torch.arange(
            self.prefix_len,
            self.prefix_len + self.chunk_size,
            dtype=torch.float32,
            device=self.device,
        )
        self._update_rope_weights(positions, decoder_positions)

    def _set_default_masks(self):
        prefix_valid = torch.ones(self.prefix_len, dtype=torch.bool, device=self.device)
        self.buffers['prefix_padding_mask'].copy_(prefix_valid)
        self._update_attention_masks(prefix_valid)

    def _update_attention_masks(self, prefix_valid):
        prefix_valid = prefix_valid.to(device=self.device, dtype=torch.bool)
        self.buffers['prefix_padding_mask'].copy_(prefix_valid)

        prefix_ar = torch.ones(self.prefix_len, dtype=torch.int32, device=self.device)
        prefix_cumsum = torch.cumsum(prefix_ar, dim=0)
        prefix_attend = prefix_cumsum[None, :] <= prefix_cumsum[:, None]
        prefix_attend = prefix_attend & prefix_valid[None, :]
        prefix_attend = prefix_attend | torch.eye(
            self.prefix_len, dtype=torch.bool, device=self.device
        )
        prefix_attn_bias = torch.where(prefix_attend, 0.0, -2.3819763e38).to(
            torch.bfloat16
        )
        self.buffers['prefix_attn_mask'].copy_(prefix_attn_bias[None, None, :, :])

        suffix_valid = torch.ones(self.chunk_size, dtype=torch.bool, device=self.device)
        combined_valid = torch.cat([prefix_valid, suffix_valid], dim=0)
        suffix_attend = combined_valid[None, :].expand(self.chunk_size, self.total_keys)
        decoder_attn_bias = torch.where(suffix_attend, 0.0, -2.3819763e38).to(
            torch.bfloat16
        )
        self.buffers['decoder_attn_mask'].copy_(decoder_attn_bias[None, None, :, :])

        prefix_positions = torch.cumsum(prefix_valid.to(torch.long), dim=0) - 1
        prefix_positions = torch.clamp(prefix_positions, min=0)
        prefix_offset = torch.sum(prefix_valid.to(torch.long))
        decoder_positions = prefix_offset + torch.arange(
            self.chunk_size, device=self.device, dtype=torch.long
        )
        self._update_rope_weights(prefix_positions, decoder_positions)

    def _build_prefix_valid_mask(self, image_masks, attention_mask):
        image_masks = image_masks.to(device=self.device, dtype=torch.bool).flatten()
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool).flatten()
        if image_masks.numel() != self.num_images:
            raise ValueError(
                f"image_masks must contain {self.num_images} item(s), got {image_masks.numel()}"
            )
        if attention_mask.numel() != self.max_lang_len:
            raise ValueError(
                f"attention_mask must contain {self.max_lang_len} item(s), got {attention_mask.numel()}"
            )
        image_token_masks = image_masks[:, None].expand(
            self.num_images, self.tokens_per_image
        ).reshape(-1)
        return torch.cat([image_token_masks, attention_mask], dim=0)

    def _compute_rope_weights_legacy(self, theta):
        inv_freq = 1.0 / (theta ** (torch.arange(0, 128, 2, dtype=torch.float32, device=self.device) / 128))
        positions = torch.arange(self.prefix_len, dtype=torch.float32, device=self.device)
        angles = positions[:, None] * inv_freq[None, :]
        cos_vals = torch.cos(angles).to(torch.bfloat16)
        sin_vals = torch.sin(angles).to(torch.bfloat16)
        rope = torch.stack([cos_vals, sin_vals], dim=2).reshape(self.prefix_len, 128)
        self.buffers['prefix_rope_weights'].copy_(rope)
        positions = torch.arange(self.prefix_len, self.prefix_len + self.chunk_size, dtype=torch.float32, device=self.device)
        angles = positions[:, None] * inv_freq[None, :]
        cos_vals = torch.cos(angles).to(torch.bfloat16)
        sin_vals = torch.sin(angles).to(torch.bfloat16)
        rope = torch.stack([cos_vals, sin_vals], dim=2).reshape(self.chunk_size, 128)
        self.buffers['decoder_rope_weights'].copy_(rope)

    def _record_run(self):
        run_vision_forward(self.weights, self.buffers['images'], self.buffers['input_ids'], self.freqs_cos, self.freqs_sin, self.vit_bufs, output_buf=self.buffers['prefix_x'])
        run_prefix_forward(self.weights, self.buffers, self.prefix_len, self.total_keys, 28)
        transformer_decoder(
            self.weights,
            self.buffers,
            self.prefix_len,
            self.total_keys,
            28,
            self.diffusion_steps,
        )

    def _record_graph(self):
        for _ in range(15):
            self._record_run()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            self._record_run()
            self.graph.capture_end()
        torch.cuda.synchronize()

    def forward(self, images, input_ids, noise, image_masks=None, attention_mask=None):
        self.buffers['images'].copy_(images)
        self.buffers['input_ids'].copy_(input_ids)
        self.buffers['diffusion_noise'].copy_(noise)
        if image_masks is None and attention_mask is None:
            self._set_default_masks()
        else:
            if image_masks is None:
                image_masks = torch.ones(self.num_images, dtype=torch.bool, device=self.device)
            if attention_mask is None:
                attention_mask = torch.ones(self.max_lang_len, dtype=torch.bool, device=self.device)
            prefix_valid = self._build_prefix_valid_mask(image_masks, attention_mask)
            self._update_attention_masks(prefix_valid)
        self.graph.replay()
        return self.buffers['diffusion_noise']
