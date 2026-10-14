"""Check the sglang FlashMap draft against a plain PyTorch reference of the map.

Builds ``FlashMapDraftModel`` from an export (``python -m
specforge.export.flashmap``) with the target's embedding and LM head, and
drives it as sglang's DSPARK worker does, with a stand-in KV pool and forward
batch: two requests' context is written with ``write_target_hidden_kv`` (one
prefill; one prefill plus a verify window whose uncommitted columns carry
garbage), then both are drafted in one ``forward``. The drafted logits must
match a direct implementation of the map (every attention over the whole
context, no folding, no fused kernels) on the same noise:

* eager (``FLASHMAP_COMPILE=0``), the served path's own noise is replayed into
  the reference;
* compiled, the drafts are compared noiseless (torch.compile draws its noise
  differently), which still runs every layer;
* the Cholesky factor the noise is drawn with must reproduce ``E^T E``.

Needs a GPU and the sglang environment:

    python scripts/gates/check_flashmap_sglang.py --draft <export> --target Qwen/Qwen3-8B
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig

DEVICE = torch.device("cuda")


class FakePool:
    """The slice of sglang's MHA KV pool the draft uses."""

    def __init__(self, *, layers, slots, heads, head_dim):
        shape = (slots, heads, head_dim)
        self.keys = [
            torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(layers)
        ]
        self.values = [
            torch.zeros(shape, dtype=torch.bfloat16, device=DEVICE)
            for _ in range(layers)
        ]

    def get_key_buffer(self, layer_id):
        return self.keys[layer_id]

    def get_value_buffer(self, layer_id):
        return self.values[layer_id]


def target_vocab_weights(target: str, device: str = "cuda"):
    """The target's (embed_tokens, lm_head) weights from its safetensors. A
    target that ties them (``tie_word_embeddings``, e.g. Qwen3-4B) may store
    no ``lm_head.weight``: its embedding is the LM head."""
    if not os.path.isdir(target):
        from huggingface_hub import snapshot_download

        target = snapshot_download(target, allow_patterns=["*.json", "*.safetensors"])
    with open(os.path.join(target, "config.json")) as handle:
        config = json.load(handle)
    tied = bool(config.get("tie_word_embeddings", False))
    index_path = os.path.join(target, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as handle:
            files = json.load(handle)["weight_map"]
    else:  # a single-file checkpoint
        with safe_open(os.path.join(target, "model.safetensors"), "pt") as handle:
            files = {name: "model.safetensors" for name in handle.keys()}
    embed = "model.embed_tokens.weight"
    head = embed if tied or "lm_head.weight" not in files else "lm_head.weight"
    out = []
    for name in (embed, head):
        with safe_open(os.path.join(target, files[name]), "pt", device=device) as f:
            out.append(f.get_tensor(name))
    return out


def rms_norm(x, weight, eps):
    mean_sq = x.float().pow(2).mean(-1, keepdim=True)
    return x * (weight.float() * torch.rsqrt(mean_sq + eps)).to(x.dtype)


def rope(x, positions, base):
    """Interleaved-pair RoPE of ``x (T, heads, head_dim)`` at ``positions (T,)``."""
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=x.device).float() / head_dim)
    )
    angles = positions.float()[:, None, None] * inv_freq
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    out = torch.stack(
        [
            pairs[..., 0] * angles.cos() - pairs[..., 1] * angles.sin(),
            pairs[..., 1] * angles.cos() + pairs[..., 0] * angles.sin(),
        ],
        dim=-1,
    )
    return out.flatten(-2).to(x.dtype)


@torch.no_grad()
def reference_logits(draft, embed, lm_head, hidden, anchor, noise_embed, start, end):
    """The map's logits for one block after ``hidden (T, features)`` and the
    anchor token at position T, from the embedded noise ``noise_embed (K, H)``
    (= x @ E), for a jump from ``start`` to ``end``: written out directly."""
    fm = draft.config.flashmap_config
    eps, qk_eps, ln_eps = fm["rms_norm_eps"], fm["qk_norm_eps"], fm["layer_norm_eps"]
    heads, theta = fm["num_attention_heads"], fm["rope_theta"]
    length, block = hidden.shape[0], draft.block_size
    context = torch.cat(
        [
            rms_norm(draft.fc(hidden), draft.hidden_norm.weight, eps),
            rms_norm(
                draft.anchor_proj(embed[anchor].view(1, -1)),
                draft.anchor_norm.weight,
                eps,
            ),
        ]
    )
    ctx_positions = torch.arange(length + 1, device=DEVICE)
    positions = length + 1 + torch.arange(block, device=DEVICE)
    vocab = embed.shape[0]
    uniform = torch.full((1, vocab), 1.0 / vocab, device=DEVICE).to(embed.dtype) @ embed
    gate = draft.noise_gate * (end - start)
    state = rms_norm(uniform, draft.state_norm.weight, eps) + gate * rms_norm(
        noise_embed, draft.noise_norm.weight, eps
    )
    times = torch.tensor([start, end], device=DEVICE) * fm["time_scale"]
    cond = draft.time_embed(times[:1]) + draft.jump_embed(times[1:])

    def modulate(x, adaln):
        scale, shift, gate = adaln(cond)
        return F.layer_norm(x, (x.shape[-1],), eps=ln_eps) * (1 + scale) + shift, gate

    def attention(attn, x, kv, q_positions, kv_positions):
        dim = x.shape[-1]
        q = attn.q_proj(x).view(-1, heads, dim // heads)
        k = attn.k_proj(kv).view(-1, heads, dim // heads)
        v = attn.v_proj(kv).view(-1, heads, dim // heads)
        q = rope(rms_norm(q, attn.q_norm.weight, qk_eps), q_positions, theta)
        k = rope(rms_norm(k, attn.k_norm.weight, qk_eps), kv_positions, theta)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return attn.o_proj(out.transpose(0, 1).reshape(-1, dim))

    h = draft.x_embed(state)
    for layer in draft.layers:
        normed, gate = modulate(h, layer.self_attn_adaln)
        h = h + gate * attention(layer.self_attn, normed, normed, positions, positions)
        normed, gate = modulate(h, layer.cross_attn_adaln)
        h = h + gate * attention(
            layer.cross_attn, normed, context, positions, ctx_positions
        )
        normed, gate = modulate(h, layer.mlp_adaln)
        h = h + gate * layer.mlp.fc2(F.gelu(layer.mlp.fc1(normed)))
    normed, _ = modulate(h, draft.final_adaln)
    return draft.head(normed) @ lm_head.T


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--draft", required=True, help="sglang export directory")
    parser.add_argument("--target", required=True, help="target model path or hub id")
    parser.add_argument("--context-lens", default="150,220")
    args = parser.parse_args()
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    from specforge.serving.sglang.flashmap import FlashMapDraftModel

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(args.draft)
    embed, lm_head = target_vocab_weights(args.target)
    with safe_open(os.path.join(args.draft, "model.safetensors"), "pt") as handle:
        weights = {name: handle.get_tensor(name) for name in handle.keys()}
    lens = [int(x) for x in args.context_lens.split(",")]
    features = len(config.target_layer_ids) * int(config.hidden_size)
    hiddens = [
        torch.randn(n + 8, features, device=DEVICE, dtype=torch.bfloat16) for n in lens
    ]

    for compiled in (False, True):
        os.environ["FLASHMAP_COMPILE"] = "1" if compiled else "0"
        torch.set_default_dtype(torch.bfloat16)
        with torch.device(DEVICE):
            draft = FlashMapDraftModel(config)
        torch.set_default_dtype(torch.float32)
        draft.load_weights(iter(weights.items()))
        draft.attach_shared_modules(
            embed_tokens=SimpleNamespace(weight=embed),
            lm_head=SimpleNamespace(weight=lm_head),
        )
        if not compiled:
            gram = embed.double().T @ embed.double()
            factor = draft._constants[2].double()
            err = ((factor.T @ factor - gram).norm() / gram.norm()).item()
            print(f"Cholesky factor: |F^T F - E^T E| / |E^T E| = {err:.2e}")
            assert err < 1e-2, err
        block = draft.block_size
        pool = FakePool(
            layers=draft.num_stages,
            slots=4096,
            heads=draft.num_heads,
            head_dim=draft.head_dim,
        )
        slots = 1 + torch.randperm(4095, device=DEVICE)  # slot 0 is sglang's dummy slot
        req_to_token = slots[: len(lens) * 1024].view(len(lens), 1024).to(torch.int32)
        for req, (hidden, length) in enumerate(zip(hiddens, lens)):
            locs = req_to_token[req].long()
            if req == 0:  # one prefill
                draft.write_target_hidden_kv(
                    target_hidden=hidden[:length],
                    pool=pool,
                    positions=torch.arange(length, device=DEVICE),
                    cache_loc=locs[:length],
                )
                continue
            # a prefill, then a verify window with 5 of 8 columns committed
            split, width, commit = length - 5, 8, 5
            draft.write_target_hidden_kv(
                target_hidden=hidden[:split],
                pool=pool,
                positions=torch.arange(split, device=DEVICE),
                cache_loc=locs[:split],
            )
            window = torch.cat(
                [hidden[split:length], 100 * torch.randn_like(hidden[: width - commit])]
            )
            draft.write_target_hidden_kv(
                target_hidden=window,
                pool=pool,
                positions=torch.arange(split, split + width, device=DEVICE),
                cache_loc=locs[split : split + width],
                cache_loc_2d=locs[split : split + width].view(1, -1),
                commit_lens=torch.tensor([commit], device=DEVICE),
            )
            assert all(
                not pool.keys[i][locs[length : split + width]].any()
                for i in range(draft.num_stages)
            ), "uncommitted verify columns were written"

        anchors = torch.randint(1000, 50000, (len(lens),), device=DEVICE)
        seq_lens = torch.tensor(lens, device=DEVICE)
        input_ids = torch.full((len(lens), block), 151669, device=DEVICE)
        input_ids[:, 0] = anchors
        positions = (seq_lens[:, None] + torch.arange(block, device=DEVICE)).flatten()
        out_cache_loc = torch.stack(
            [req_to_token[r, lens[r] : lens[r] + block] for r in range(len(lens))]
        ).long()
        batch = SimpleNamespace(
            batch_size=len(lens),
            seq_lens=seq_lens,
            req_pool_indices=torch.arange(len(lens), device=DEVICE),
            out_cache_loc=out_cache_loc.flatten(),
        )
        context = ForwardContext(
            attn_backend=SimpleNamespace(
                token_to_kv_pool=pool,
                req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            )
        )
        draft.noise_temperature = 0.0 if compiled else 1.0
        torch.manual_seed(1)
        with forward_context(context), torch.no_grad():
            hidden = draft.forward(input_ids.flatten(), positions, batch).hidden_states
            served = draft.compute_base_logits(hidden)[0].view(len(lens), block, -1)
        torch.manual_seed(1)  # replay the served path's noise draw
        z = draft.noise_temperature * torch.randn(
            len(lens), block, embed.shape[1], device=DEVICE, dtype=torch.bfloat16
        )
        noise_embed = z @ draft._constants[2]
        start, end = draft.grid[0], draft.grid[1]
        hot = (
            draft.hot_ids
            if draft.hot_ids is not None
            else torch.arange(embed.shape[0], device=DEVICE)
        )
        for req, length in enumerate(lens):
            ref = reference_logits(
                draft,
                embed,
                lm_head,
                hiddens[req][:length],
                anchors[req],
                noise_embed[req],
                start,
                end,
            )
            got = served[req].float()
            ref_hot = ref[:, hot].float()
            diff = (got - ref_hot).abs().max().item() / ref_hot.abs().max().item()
            agree = (got.argmax(-1) == ref_hot.argmax(-1)).float().mean().item()
            full_agree = (hot[got.argmax(-1)] == ref.argmax(-1)).float().mean().item()
            print(
                f"{'compiled' if compiled else 'eager':8s} request {req}: max |served - reference| / max |reference| = "
                f"{diff:.2e}, argmax agreement {agree:.0%} (vs full-vocabulary reference {full_agree:.0%})"
            )
            assert diff < 5e-2 and agree >= 0.9, (diff, agree)
        assert math.isfinite(served.float().sum().item())
    print("ok")


if __name__ == "__main__":
    main()
