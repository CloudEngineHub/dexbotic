import math
import argparse
import torch
import torch.nn.functional as F

def _posemb_sincos(time_val, dim, device="cpu"):
    dtype = torch.float64
    time = torch.tensor([time_val], dtype=dtype, device=device)
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=dtype, device=device)
    min_period = 4e-3
    max_period = 4.0
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).squeeze(0)

def _to_interleaved(tensor, num_heads, head_dim):
    shape = tensor.shape
    tensor = tensor.view(*shape[:-1], num_heads, 2, head_dim // 2)
    tensor = tensor.transpose(-1, -2).reshape(shape)
    return tensor.contiguous()

def convert_weights(weights, model, device="cuda", diffusion_steps=None):
    config = model.config
    action_config = config.action_config
    llm_config = config.llm_config
    diffusion_steps = int(
        diffusion_steps
        if diffusion_steps is not None
        else getattr(config, "diffusion_steps", 10)
    )
    if diffusion_steps <= 0:
        raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
    num_layers = getattr(config, "action_num_layers", action_config.num_hidden_layers)
    hidden = getattr(config, "action_hidden_size", action_config.hidden_size)
    num_q_heads = getattr(config, "action_num_heads", action_config.num_attention_heads)
    num_kv_heads = getattr(
        config, "action_num_kv_heads", action_config.num_key_value_heads
    )
    head_dim = getattr(config, "action_head_dim", action_config.head_dim)
    dt = -1.0 / diffusion_steps
    action_in = model.model.action_in_proj
    time_mlp_in = model.model.action_time_mlp_in
    w_action = time_mlp_in.weight[:, :hidden]
    w_time = time_mlp_in.weight[:, hidden:]
    fused_action_weight = w_action @ action_in.weight
    fused_action_bias = time_mlp_in.bias.clone()
    if action_in.bias is not None:
        fused_action_bias = fused_action_bias + F.linear(action_in.bias, w_action, None)

    weights['decoder_action_fused_in_proj_w'].copy_(
        fused_action_weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
    fused_time_w_cpu = w_time.data.float().cpu()
    fused_action_bias_cpu = fused_action_bias.data.float().cpu()
    time_biases = torch.zeros(diffusion_steps, hidden, dtype=torch.float32)
    for step in range(diffusion_steps):
        time_val = 1.0 - step / diffusion_steps
        time_emb = _posemb_sincos(time_val, hidden, device="cpu").float()
        time_proj = F.linear(time_emb, fused_time_w_cpu)
        time_biases[step] = fused_action_bias_cpu + time_proj
    weights['decoder_action_fused_time_biases'].copy_(time_biases.to(torch.bfloat16).to(device))
    weights['decoder_action_mlp_w'].copy_(
        model.model.action_time_mlp_out.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
    weights['decoder_action_mlp_b'].copy_(
        model.model.action_time_mlp_out.bias.data.float().to(torch.bfloat16).to(device))
    final_norm_w = model.model.action_expert.model.norm.weight.data.float()
    out_proj_w = model.model.action_out_proj.weight.data.float().T.contiguous()
    out_proj_b = model.model.action_out_proj.bias.data.float()
    weights['decoder_action_fused_out_proj_w'].copy_(
        (out_proj_w * final_norm_w[:, None] * dt).to(torch.bfloat16).to(device))
    weights['decoder_action_fused_out_proj_b'].copy_(
        (out_proj_b * dt).to(torch.bfloat16).to(device))
    decoder_attn_qkv_w, decoder_q_norm_w, decoder_k_norm_w = [], [], []
    decoder_attn_o_w = []
    decoder_ffn_gate_w, decoder_ffn_up_w, decoder_ffn_down_w = [], [], []

    for i in range(num_layers):
        layer = model.model.action_expert.model.layers[i]
        input_norm_w = layer.input_layernorm.weight.data.float()
        q_w = layer.self_attn.q_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        k_w = layer.self_attn.k_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        v_w = layer.self_attn.v_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        q_w = _to_interleaved(q_w, num_q_heads, head_dim)
        k_w = _to_interleaved(k_w, num_kv_heads, head_dim)
        decoder_attn_qkv_w.append(torch.cat([q_w, k_w, v_w], dim=1).to(torch.bfloat16).to(device))
        q_norm = layer.self_attn.q_norm.weight.data.float()
        k_norm = layer.self_attn.k_norm.weight.data.float()
        decoder_q_norm_w.append(q_norm.view(2, head_dim // 2).T.reshape(head_dim).to(torch.bfloat16).to(device))
        decoder_k_norm_w.append(k_norm.view(2, head_dim // 2).T.reshape(head_dim).to(torch.bfloat16).to(device))

        decoder_attn_o_w.append(layer.self_attn.o_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))
        post_norm_w = layer.post_attention_layernorm.weight.data.float()
        gate_w = layer.mlp.gate_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        up_w = layer.mlp.up_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        decoder_ffn_gate_w.append(gate_w.to(torch.bfloat16).to(device))
        decoder_ffn_up_w.append(up_w.to(torch.bfloat16).to(device))
        decoder_ffn_down_w.append(layer.mlp.down_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

    weights['decoder_attn_qkv_w'].copy_(torch.stack(decoder_attn_qkv_w))
    weights['decoder_q_norm_w'].copy_(torch.stack(decoder_q_norm_w))
    weights['decoder_k_norm_w'].copy_(torch.stack(decoder_k_norm_w))
    weights['decoder_attn_o_w'].copy_(torch.stack(decoder_attn_o_w))
    weights['decoder_ffn_gate_w'].copy_(torch.stack(decoder_ffn_gate_w))
    weights['decoder_ffn_up_w'].copy_(torch.stack(decoder_ffn_up_w))
    weights['decoder_ffn_down_w'].copy_(torch.stack(decoder_ffn_down_w))
    llm_hidden = getattr(config, "llm_hidden_size", llm_config.hidden_size)
    llm_num_layers = getattr(config, "llm_num_layers", llm_config.num_hidden_layers)
    llm_num_q_heads = getattr(config, "llm_num_heads", llm_config.num_attention_heads)
    llm_num_kv_heads = getattr(
        config, "llm_num_kv_heads", llm_config.num_key_value_heads
    )
    llm_head_dim = getattr(config, "llm_head_dim", llm_config.head_dim)

    llm_attn_qkv_w, llm_q_norm_w, llm_k_norm_w = [], [], []
    llm_attn_o_w = []
    llm_ffn_gate_w, llm_ffn_up_w, llm_ffn_down_w = [], [], []

    for i in range(llm_num_layers):
        layer = model.model.llm.layers[i]
        input_norm_w = layer.input_layernorm.weight.data.float()

        q_w = layer.self_attn.q_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        k_w = layer.self_attn.k_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        v_w = layer.self_attn.v_proj.weight.data.float().T.contiguous() * input_norm_w[:, None]
        q_w = _to_interleaved(q_w, llm_num_q_heads, llm_head_dim)
        k_w = _to_interleaved(k_w, llm_num_kv_heads, llm_head_dim)
        llm_attn_qkv_w.append(torch.cat([q_w, k_w, v_w], dim=1).to(torch.bfloat16).to(device))

        q_norm = layer.self_attn.q_norm.weight.data.float()
        k_norm = layer.self_attn.k_norm.weight.data.float()
        llm_q_norm_w.append(q_norm.view(2, llm_head_dim // 2).T.reshape(llm_head_dim).to(torch.bfloat16).to(device))
        llm_k_norm_w.append(k_norm.view(2, llm_head_dim // 2).T.reshape(llm_head_dim).to(torch.bfloat16).to(device))

        llm_attn_o_w.append(layer.self_attn.o_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

        post_norm_w = layer.post_attention_layernorm.weight.data.float()
        gate_w = layer.mlp.gate_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        up_w = layer.mlp.up_proj.weight.data.float().T.contiguous() * post_norm_w[:, None]
        llm_ffn_gate_w.append(gate_w.to(torch.bfloat16).to(device))
        llm_ffn_up_w.append(up_w.to(torch.bfloat16).to(device))
        llm_ffn_down_w.append(layer.mlp.down_proj.weight.data.float().T.contiguous().to(torch.bfloat16).to(device))

    weights['llm_attn_qkv_w'].copy_(torch.stack(llm_attn_qkv_w))
    weights['llm_q_norm_w'].copy_(torch.stack(llm_q_norm_w))
    weights['llm_k_norm_w'].copy_(torch.stack(llm_k_norm_w))
    weights['llm_attn_o_w'].copy_(torch.stack(llm_attn_o_w))
    weights['llm_ffn_gate_w'].copy_(torch.stack(llm_ffn_gate_w))
    weights['llm_ffn_up_w'].copy_(torch.stack(llm_ffn_up_w))
    weights['llm_ffn_down_w'].copy_(torch.stack(llm_ffn_down_w))
    sd = model.state_dict()
    vp = 'model.mm_vision_tower.vision_tower'
    weights['vision_conv1_w_t'].copy_(
        sd[f'{vp}.conv1.weight'].float().reshape(1024, -1).T.contiguous().to(torch.bfloat16).to(device))
    weights['vision_class_embedding'].copy_(sd[f'{vp}.class_embedding'].to(torch.bfloat16).to(device))
    weights['vision_pos_emb'].copy_(sd[f'{vp}.positional_embedding'].to(torch.bfloat16).to(device))
    weights['vision_ln_pre_w'].copy_(sd[f'{vp}.ln_pre.weight'].to(torch.bfloat16).to(device))
    weights['vision_ln_pre_b'].copy_(sd[f'{vp}.ln_pre.bias'].to(torch.bfloat16).to(device))
    v_fused_qkv_w, v_fused_qkv_b, v_qkv_col_sum = [], [], []
    v_out_proj_w, v_out_proj_b = [], []
    v_fused_fc_w, v_fused_fc_b, v_fc_col_sum = [], [], []
    v_proj_w, v_proj_b = [], []

    for i in range(23):
        bp = f'{vp}.transformer.resblocks.{i}'
        ln1_w = sd[f'{bp}.ln_1.weight'].float()
        ln1_b = sd[f'{bp}.ln_1.bias'].float()
        in_proj_w = sd[f'{bp}.attn.in_proj_weight'].float().T.contiguous()
        in_proj_b = sd[f'{bp}.attn.in_proj_bias'].float()
        qkv_fused = ln1_w[:, None] * in_proj_w
        v_fused_qkv_w.append(qkv_fused.to(torch.bfloat16).to(device))
        v_fused_qkv_b.append((torch.matmul(ln1_b, in_proj_w) + in_proj_b).to(torch.float32).to(device))
        v_qkv_col_sum.append(qkv_fused.sum(dim=0).to(torch.float32).to(device))

        ls1 = sd[f'{bp}.ls_1.gamma'].float()
        ow = sd[f'{bp}.attn.out_proj.weight'].float().T.contiguous() * ls1
        ob = sd[f'{bp}.attn.out_proj.bias'].float() * ls1
        v_out_proj_w.append(ow.to(torch.bfloat16).to(device))
        v_out_proj_b.append(ob.to(torch.bfloat16).to(device))

        ln2_w = sd[f'{bp}.ln_2.weight'].float()
        ln2_b = sd[f'{bp}.ln_2.bias'].float()
        fc_w = sd[f'{bp}.mlp.c_fc.weight'].float().T.contiguous()
        fc_b = sd[f'{bp}.mlp.c_fc.bias'].float()
        fc_fused = ln2_w[:, None] * fc_w
        v_fused_fc_w.append(fc_fused.to(torch.bfloat16).to(device))
        v_fused_fc_b.append((torch.matmul(ln2_b, fc_w) + fc_b).to(torch.float32).to(device))
        v_fc_col_sum.append(fc_fused.sum(dim=0).to(torch.float32).to(device))

        ls2 = sd[f'{bp}.ls_2.gamma'].float()
        pw = sd[f'{bp}.mlp.c_proj.weight'].float().T.contiguous() * ls2
        pb = sd[f'{bp}.mlp.c_proj.bias'].float() * ls2
        v_proj_w.append(pw.to(torch.bfloat16).to(device))
        v_proj_b.append(pb.to(torch.bfloat16).to(device))

    weights['vision_fused_qkv_w'].copy_(torch.stack(v_fused_qkv_w))
    weights['vision_fused_qkv_b'].copy_(torch.stack(v_fused_qkv_b))
    weights['vision_qkv_col_sum'].copy_(torch.stack(v_qkv_col_sum))
    weights['vision_out_proj_w'].copy_(torch.stack(v_out_proj_w))
    weights['vision_out_proj_b'].copy_(torch.stack(v_out_proj_b))
    weights['vision_fused_fc_w'].copy_(torch.stack(v_fused_fc_w))
    weights['vision_fused_fc_b'].copy_(torch.stack(v_fused_fc_b))
    weights['vision_fc_col_sum'].copy_(torch.stack(v_fc_col_sum))
    weights['vision_proj_w'].copy_(torch.stack(v_proj_w))
    weights['vision_proj_b'].copy_(torch.stack(v_proj_b))
    weights['vision_ds1_w'].copy_(
        sd[f'{vp}.vit_downsampler1.weight'].permute(2, 3, 1, 0).contiguous().to(torch.bfloat16).to(device))
    weights['vision_ds1_b'].copy_(
        sd[f'{vp}.vit_downsampler1.bias'].to(torch.bfloat16).to(device))
    weights['vision_ds2_w'].copy_(
        sd[f'{vp}.vit_downsampler2.weight'].permute(2, 3, 1, 0).contiguous().to(torch.bfloat16).to(device))
    weights['vision_ds2_b'].copy_(
        sd[f'{vp}.vit_downsampler2.bias'].to(torch.bfloat16).to(device))
    weights['vision_projector_w_t'].copy_(
        sd['model.mm_projector.weight'].T.contiguous().to(torch.bfloat16).to(device))
    weights['vision_embed_tokens_w'].copy_(
        sd['model.llm.embed_tokens.weight'].to(torch.bfloat16).to(device))

def load_dm0_model(model_path, device="cuda"):
    from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM

    model = DM0ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": device},
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model

def create_weights_dict(config, device="cuda", diffusion_steps=None):
    if not hasattr(config, "llm_vocab_size"):
        config.llm_vocab_size = config.llm_config.vocab_size
    diffusion_steps = int(
        diffusion_steps
        if diffusion_steps is not None
        else getattr(config, "diffusion_steps", 10)
    )
    if diffusion_steps <= 0:
        raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
    ds1_out_c = 2048
    ds2_out_c = 4096
    return {
        'decoder_attn_qkv_w':        torch.empty(28, 1024, 4096, dtype=torch.bfloat16, device=device),
        'decoder_q_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
        'decoder_k_norm_w':          torch.empty(28, 128,              dtype=torch.bfloat16, device=device),
        'decoder_attn_o_w':          torch.empty(28, 2048, 1024,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_gate_w':        torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_up_w':          torch.empty(28, 1024, 1536,       dtype=torch.bfloat16, device=device),
        'decoder_ffn_down_w':        torch.empty(28, 1536, 1024,       dtype=torch.bfloat16, device=device),
        'decoder_action_fused_in_proj_w': torch.empty(32, 1024,         dtype=torch.bfloat16, device=device),
        'decoder_action_fused_time_biases': torch.empty(diffusion_steps, 1024, dtype=torch.bfloat16, device=device),
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
        'vision_ds1_w':               torch.empty(3, 3, 1024, ds1_out_c, dtype=torch.bfloat16, device=device),
        'vision_ds1_b':               torch.empty(ds1_out_c,            dtype=torch.bfloat16, device=device),
        'vision_ds2_w':               torch.empty(3, 3, ds1_out_c, ds2_out_c, dtype=torch.bfloat16, device=device),
        'vision_ds2_b':               torch.empty(ds2_out_c,            dtype=torch.bfloat16, device=device),
        'vision_projector_w_t':       torch.empty(4096, 2048,           dtype=torch.bfloat16, device=device),
        'vision_embed_tokens_w':      torch.empty(config.llm_vocab_size, 2048, dtype=torch.bfloat16, device=device),
    }

def convert_checkpoint(model_path, output, device="cuda", diffusion_steps=None):
    model = load_dm0_model(model_path, device=device)
    weights = create_weights_dict(model.config, device=device, diffusion_steps=diffusion_steps)
    convert_weights(weights, model, device=device, diffusion_steps=diffusion_steps)
    torch.save(weights, output)
    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="dm0_triton_weights.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffusion_steps", type=int, default=None)
    args = parser.parse_args()

    weights = convert_checkpoint(
        args.model_path,
        args.output,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )
    print(f"\nSaved to {args.output}")
    total = sum(t.numel() for t in weights.values())
    print(f"Total: {total/1e6:.1f}M params")
