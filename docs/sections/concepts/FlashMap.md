# FlashMap

FlashMap is a block-parallel drafter. Like DFlash it drafts the 16 tokens
after the last verified token (the anchor) in one pass, conditioned on the
target's hidden states. Its block comes from a distilled discrete flow map
(Flow Map Language Models, arXiv:2602.16813): one network evaluation jumps
Gaussian noise over the target vocabulary straight to per-slot token
distributions. Two jumps cost twice as much and are more accurate.

| Boundary | FlashMap value |
| --- | --- |
| Training strategy | `training.strategy: flashmap` |
| Stages | `training.flashmap_stage: flow`, then `map` |
| Draft architecture | `FlashMapDraftModel` |
| Online server capture | `--spec-capture-method dflash` |
| Offline feature schema | DFlash `input_ids`, `loss_mask`, `hidden_states` |
| Export target | `specforge export --to hf`, then `python -m specforge.export.flashmap` for SGLang |
| SGLang serving algorithm | `DSPARK` with SpecForge's external FlashMap model ([guide](../basic_usage/flashmap_sglang.md)) |

FlashMap reads the same target features as DFlash, so capture servers,
offline feature files and the collator are DFlash's, and DFlash's anchor
sampling, block labels and slot-decay loss weights are reused unchanged.

## Architecture

The draft is a DiT (6 blocks, width 2048, 16 heads) over the 16 draft slots.
Each block runs AdaLN-zero-modulated self-attention within the block,
cross-attention onto the context, and an MLP. The context is the target's
hidden states at `target_layer_ids`, fused by `fc`, strictly before the
anchor, plus the anchor token's embedding projected by `anchor_proj`. The
target's embedding and LM head are frozen and borrowed, never draft weights:
the embedding featurizes the noisy state, and the LM head turns the DiT
output into logits.

A block's state is a point `x` in the vocabulary space. The DiT reads it at
time `s` through `softmax(s x / (1 - s)^2) @ E`, the posterior over the clean
token, normalized by `state_norm`. The map also reads `x @ E` itself through
`noise_norm`, gated by `t - s`, because at `s = 0` the posterior is uniform
whatever the noise. AdaLN conditions on embeddings of the start time `s` and,
for the map, the end time `t` (`jump_embed`).

## Two training stages

**Flow (`flashmap_stage: flow`).** The DiT is trained as the denoiser of the
path `I_t = (1 - t) z + t onehot(y)`: cross entropy against the block's clean
tokens. Every anchor block is noised to its own time `t(tau)`, `tau ~ U[0, 1]`
under the decoding-error schedule (Flow Map Language Models, eq. 25). The
`flashmap/ce_loss@tau{0..3}` metrics split the loss by quartile of `tau`.

**Map (`flashmap_stage: map`).** The flow at `model.flashmap_flow_checkpoint`
is frozen and the map starts as a copy of it. Its end-time embedding's output
layer starts at zero and its noise read-out is gated off on the diagonal
`s = t`, so on the diagonal the initial map is exactly the flow's denoiser
(jumps `s < t` already read their noise). Training uses the semigroup objective
(eq. 23, Algorithm 7) with the official implementation's batch split:

- `flashmap_diagonal_fraction` (0.5) of the blocks train `delta_{t,t}` against
  the frozen flow's denoiser;
- of the rest, `flashmap_boundary_prob` (0.0625) take the drafting jump
  `(s, t) = (0, 1)`, and the others take a jump `s -> t` with midpoint `u`;
- off-diagonal blocks are trained toward the map's own two half-jumps,
  `gamma delta_{s,u} + (1 - gamma) delta_{u,t}`, under stop-gradient;
- times are drawn in the schedule's `tau` and stratified over the batch
  (`flashmap_antithetic_sampling`);
- categories and times are drawn for every anchor slot of the padded batch
  (`batch_size * num_anchors`), so sequences with few valid anchors still get
  drafting-jump blocks.

The loss is soft-target cross entropy, which has the same gradients as the
KL the paper uses. `map_kl` reports the KL, overall and for the
`diag`/`off`/`boundary` groups. `flashmap/acc@boundary` is the argmax
accuracy of the jump that drafting uses. Each map step evaluates the network
three times on vocabulary-sized states, so the map draws fewer anchors per
sequence than the flow (`num_anchors` 128 versus 512; the map stage warns when
it is left at the shared default 512). For FlashMap, `loss_decay_gamma: null`
means the recipe's 7; set 0 to disable the slot decay. Each data-parallel rank
draws its own anchors, times and noise (the training device's generator is
seeded with `seed + rank` and checkpointed per rank).

## Train

Both stages use the checked-in draft config, which matches the released
FlashMap checkpoints. The stage selects the network (`flow_map` in the
config), so a stage-1 flow has no `jump_embed`, `noise_norm` or `noise_gate`.

```bash
# features (offline), with the DFlash capture
torchrun --nproc_per_node=8 scripts/prepare_hidden_states.py \
    --strategy flashmap --target-model-path Qwen/Qwen3-8B \
    --draft-model-config configs/qwen3-8b-flashmap.json \
    --data-path ./cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
    --output-path ./cache/hidden_states/qwen3-8b-flashmap-perfectblend \
    --chat-template qwen --max-length 4096

# stage 1: the flow
specforge train --config examples/configs/offline/colocated/qwen3-8b-flashmap-flow-offline.yaml

# stage 2: distil the flow into the map
specforge train --config examples/configs/offline/colocated/qwen3-8b-flashmap-map-offline.yaml
```

`model.flashmap_flow_checkpoint` accepts a stage-1 run directory, checkpoint
directory or `training_state.pt`, or an exported model directory. The map
starts from that flow unless `model.draft_checkpoint_path` warm-starts it from
an existing map, and resuming a map run (`training.resume_from`) still needs
the flow path for its teacher. Online recipes,
`examples/configs/online/disaggregated/external/qwen3-4b-flashmap-{flow,map}-online.yaml`,
train both stages from a DFlash capture server.

A draft config can also be derived from the target by omitting
`model.draft_model_config`. The default is the 6-block, width-2048 DiT over five
target layers, chosen as DFlash chooses them. FlashMap trains over the full
target vocabulary: the serving export can restrict the drafting vocabulary,
so `model.vocab_mapping_path` is rejected.

The recipes follow the released models' training, which used 500k PerfectBlend
conversations with regenerated target answers and a global batch of 256
sequences. The flow trained 20 epochs at learning rate 6e-4, with a cosine
schedule, AdamW `adam_beta2: 0.95` and `weight_decay: 0.1`; the decay applies to
weight matrices only, and it removed loss spikes at that scale. The map trained
12 epochs at 5e-5, with `lr_scheduler: wsd` (warmup 0.04, then the peak rate, then
an anneal over the last `lr_decay_ratio: 0.2`), `adam_beta2: 0.95` and no weight
decay. Both used slot decay `gamma = 7` and gradient clipping at 1.

The recipes train data parallel with `fsdp_sharding: NO_SHARD` (DDP), which suits
a draft of about 0.5B parameters. The sharded modes also work: the DiT is one
FSDP unit, because its cross-attention context is projected once per step and
read by every evaluation.

## Export and serve

A map checkpoint exports to the released FlashMap layout: `config.json` with
`model_type: flashmap`, and the draft's own weights without the target's
embedding and LM head.

```bash
specforge export --to hf \
    --checkpoint ./outputs/qwen3-8b-flashmap-map-offline \
    --draft-config configs/qwen3-8b-flashmap.json \
    --output-dir ./exports/FlashMap-Qwen3-8B

python -m specforge.export.flashmap \
    --src ./exports/FlashMap-Qwen3-8B --output-dir ./exports/FlashMap-Qwen3-8B-sglang
```

Then serve the second directory as described in
[Serving FlashMap with SGLang](../basic_usage/flashmap_sglang.md).

The same command exports a stage-1 flow: the exporter reads from the weights
whether a checkpoint holds a flow or a map, so the map's draft config serves
for both, and the exported `config.json` records `flow_map`. An exported
directory loads back into SpecForge (`model.draft_checkpoint_path` or
`model.flashmap_flow_checkpoint`) as it is. To load it with plain transformers
(`trust_remote_code=True`), copy `configuration_flashmap.py` and
`modeling_flashmap.py` from a released FlashMap model folder into it and add
`"auto_map": {"AutoConfig": "configuration_flashmap.FlashMapConfig",
"AutoModel": "modeling_flashmap.FlashMapDraftModel"}` to its `config.json`.
