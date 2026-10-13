"""Built-in FlashMap registration and executable providers.

FlashMap trains in two stages under one strategy, selected by
``training.flashmap_stage``: ``flow`` trains the flow, ``map`` distils the
flow at ``model.flashmap_flow_checkpoint`` into the flow map that is served.
Both consume DFlash's features (the target's hidden states at
``target_layer_ids``), so capture, offline storage and collation are DFlash's.
"""

from __future__ import annotations

from functools import partial

from specforge.algorithms.common.defaults import (
    empty_options,
    no_missing_checkpoint_keys,
    online_needs_input_tools,
)
from specforge.algorithms.common.hidden_states_data import (
    NORMALIZER_ID,
    build_collator,
    build_offline_normalizer,
    build_offline_reader,
)
from specforge.algorithms.common.providers import (
    AlgorithmProviders,
    DraftConfigProvider,
    ModelProvider,
    OfflineCaptureLayout,
    OfflineDataProvider,
    ServerCaptureLayout,
    ServerStreamingProvider,
    StepProvider,
    TargetDerivedDraftDefaults,
    make_registration,
)
from specforge.algorithms.contracts import (
    AlgorithmCapabilities,
    AlgorithmSpec,
    DraftRequirement,
    FeatureContract,
    FeatureMode,
    OfflineStorageContract,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

ALGORITHM_NAME = "flashmap"
DRAFT_ARCHITECTURE = "FlashMapDraftModel"
# Draft keys a map has and its flow does not; they start from their init.
MAP_ONLY_KEY_PREFIXES = ("jump_embed.", "noise_norm.", "noise_gate")
# Target-architecture fields the generic draft-config generator copies, which
# describe the target rather than the DiT.
_TARGET_ONLY_FIELDS = (
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "rms_norm_eps",
    "hidden_act",
    "attention_bias",
    "attention_dropout",
    "max_position_embeddings",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
    "initializer_range",
)


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import FlashMapTrainStrategy

    return FlashMapTrainStrategy(wrapped_model)


def resume_contract(config, draft_model, training_model):
    """Persist the resolved FlashMap architecture, stage and objective."""

    contract = {
        "flashmap_stage": str(training_model.stage),
        "flashmap_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "flashmap_block_size": int(training_model.block_size),
        "flashmap_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "flashmap_hidden_size": int(draft_model.config.hidden_size),
        "flashmap_attention_backend": str(training_model.attention_backend),
        "flashmap_num_anchors": int(training_model.num_anchors),
        "flashmap_loss_decay_gamma": training_model.loss_decay_gamma,
    }
    if training_model.stage == "map":
        contract.update(
            {
                "flashmap_diagonal_fraction": training_model.diagonal_fraction,
                "flashmap_boundary_prob": training_model.boundary_prob,
                "flashmap_antithetic_sampling": training_model.antithetic_sampling,
            }
        )
    return contract


def populate_target_defaults(payload, target_config, config):
    """Derive a FlashMap draft config from the target's: the paper's DiT
    (6 blocks, width 2048, 16 heads) over five target layers."""
    from specforge.modeling.draft.dflash import build_target_layer_ids

    num_target_layers = int(target_config.num_hidden_layers)
    payload["target_hidden_size"] = payload.pop("hidden_size")
    for field in _TARGET_ONLY_FIELDS:
        payload.pop(field, None)
    payload["target_model"] = config.model.target_model_path
    payload["target_layer_ids"] = build_target_layer_ids(num_target_layers, 5)
    payload["flow_map"] = config.training.flashmap_stage == "map"


def apply_draft_overrides(config, draft_config):
    # One draft config serves both stages: the stage selects the network.
    draft_config.flow_map = config.training.flashmap_stage == "map"


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def _load_flow(config, draft_config):
    """The frozen stage-1 flow from a SpecForge checkpoint or HF directory."""
    import copy

    from specforge.algorithms.model_providers import _device, _torch_dtype
    from specforge.modeling.draft.flashmap import FlashMapDraftModel
    from specforge.training.model_loading import warm_start_draft_model

    source = config.model.flashmap_flow_checkpoint
    if not source:
        raise ValueError(
            "training.flashmap_stage=map distils a trained flow; set "
            "model.flashmap_flow_checkpoint to the flow stage's checkpoint"
        )
    flow_config = copy.deepcopy(draft_config)
    flow_config.flow_map = False
    flow = FlashMapDraftModel(flow_config)
    warm_start_draft_model(
        flow,
        source,
        draft_config=flow_config,
        strategy=config.training.strategy,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    flow = flow.to(device=_device(), dtype=_torch_dtype(config))
    return flow.requires_grad_(False).eval()


def initialize_map_from_flow(draft_model, flow) -> None:
    """Start the map as its flow: every shared weight is copied; the end-time
    embedding (zero output) and the gated noise read-out (zero on the
    diagonal) keep their init."""
    result = draft_model.load_state_dict(flow.state_dict(), strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [
        key for key in result.missing_keys if not key.startswith(MAP_ONLY_KEY_PREFIXES)
    ]
    if unexpected or missing:
        raise ValueError(
            "flow checkpoint does not match the map architecture: "
            f"unexpected={sorted(unexpected)}, missing={sorted(missing)}"
        )


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.flashmap.model import OnlineFlashMapModel
    from specforge.algorithms.model_providers import (
        AlgorithmModelParts,
        _device,
        _torch_dtype,
    )
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    del target_config, tokenizer
    if config.model.vocab_mapping_path:
        raise ValueError(
            "FlashMap trains over the full target vocabulary; restrict the "
            "drafting vocabulary at serving time instead"
        )
    stage = config.training.flashmap_stage
    flow = None
    if stage == "map":
        flow = _load_flow(config, draft_config)
        # A warm start (or, later, a resume) already holds map weights.
        if not config.model.draft_checkpoint_path:
            initialize_map_from_flow(draft_model, flow)
    elif config.model.flashmap_flow_checkpoint:
        raise ValueError(
            "model.flashmap_flow_checkpoint is the map stage's teacher; "
            "warm-start a flow with model.draft_checkpoint_path"
        )
    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        config.model.target_model_path,
        embed_key=config.model.embedding_key,
        lm_head_key=config.model.lm_head_key,
        cache_dir=config.model.cache_dir,
        device=_device().type,
        dtype=_torch_dtype(config),
        trust_remote_code=config.model.trust_remote_code,
    )
    training = config.training
    model = OnlineFlashMapModel(
        draft_model,
        target_parts.lm_head,
        target_parts.embed_tokens,
        stage=stage,
        flow=flow,
        attention_backend=training.attention_backend,
        num_anchors=training.num_anchors,
        loss_decay_gamma=training.loss_decay_gamma,
        objective_chunk_blocks=training.objective_chunk_blocks,
        diagonal_fraction=training.flashmap_diagonal_fraction,
        boundary_prob=training.flashmap_boundary_prob,
        antithetic_sampling=training.flashmap_antithetic_sampling,
    ).to(device=_device(), dtype=_torch_dtype(config))
    return AlgorithmModelParts(
        model=model, capture_layers=list(draft_model.target_layer_ids)
    )


def resolve_capture_layers(config, draft_config, target_config):
    del config, target_config
    layers = (
        draft_config.get("target_layer_ids")
        if isinstance(draft_config, dict)
        else getattr(draft_config, "target_layer_ids", None)
    )
    if not layers:
        raise ValueError("draft config does not define target_layer_ids")
    return list(layers)


def minimum_loss_tokens(_config, _draft_config):
    # An anchor and its first draft token are both supervised.
    return 2


def algorithm_spec() -> AlgorithmSpec:
    ready = {"input_ids", "loss_mask", "hidden_states"}
    return AlgorithmSpec(
        name=ALGORITHM_NAME,
        draft=DraftRequirement(
            compatible_architectures=frozenset({DRAFT_ARCHITECTURE}),
            default_architecture=DRAFT_ARCHITECTURE,
            supported_overrides={"num_hidden_layers", "block_size"},
        ),
        feature_contracts=(
            FeatureContract(
                mode=FeatureMode.OFFLINE,
                modality="text",
                required_tensors=ready,
                storage=OfflineStorageContract(
                    format="specforge_hidden_states_v1",
                    required_tensors=ready,
                    normalizer=NORMALIZER_ID,
                ),
            ),
            FeatureContract(
                mode=FeatureMode.STREAMING,
                modality="text",
                required_tensors=ready,
            ),
        ),
        capabilities=AlgorithmCapabilities(
            attention_backends={"sdpa", "flex_attention"},
        ),
    )


def algorithm_providers() -> AlgorithmProviders:
    return AlgorithmProviders(
        algorithm_name=ALGORITHM_NAME,
        step=StepProvider(
            build=build_step,
            options=empty_options,
            resume_contract=resume_contract,
            allowed_missing_checkpoint_keys=no_missing_checkpoint_keys,
            uses_external_target_head=False,
        ),
        model=ModelProvider(
            draft_config=DraftConfigProvider(
                architecture=DRAFT_ARCHITECTURE,
                target_defaults=TargetDerivedDraftDefaults(
                    model_type="flashmap",
                    num_hidden_layers=6,
                    populate=populate_target_defaults,
                ),
                apply_overrides=apply_draft_overrides,
            ),
            build_draft=build_draft,
            build_training_model=build_training_model,
            resolve_capture_layers=resolve_capture_layers,
            minimum_loss_tokens=minimum_loss_tokens,
            needs_input_tools=online_needs_input_tools,
            default_dataloader_num_workers=8,
            loss_mask_filter=has_consecutive_supervised_tokens,
        ),
        offline=(
            OfflineDataProvider(
                modality="text",
                normalizer_id=NORMALIZER_ID,
                capture_layout=OfflineCaptureLayout(
                    capture_method="dflash",
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids"),
                        ("loss_mask", "loss_mask"),
                    ),
                ),
                build_reader=partial(build_offline_reader, ALGORITHM_NAME),
                build_normalizer=build_offline_normalizer,
                build_collator=build_collator,
            ),
        ),
        server_streaming=(
            ServerStreamingProvider(
                modality="text",
                capture_method="dflash",
                target_representation=None,
                layout=ServerCaptureLayout(
                    aux_feature="hidden_states",
                    last_hidden_feature=None,
                    passthrough=(
                        ("input_ids", "input_ids", ()),
                        ("loss_mask", "loss_mask", ()),
                    ),
                ),
                build_collator=build_collator,
            ),
        ),
    )


def create_registration():
    return make_registration(algorithm_spec(), algorithm_providers())


__all__ = ["algorithm_providers", "algorithm_spec", "create_registration"]
