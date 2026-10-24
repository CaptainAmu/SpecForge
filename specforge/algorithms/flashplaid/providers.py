"""Built-in FlashPlaid registration and executable providers.

FlashPlaid consumes DFlash's features (the target's hidden states at
``target_layer_ids``), like FlashMap, so capture, offline storage and
collation are DFlash's; existing DFlash/FlashMap hidden-state directories with
the same layers are usable as they are. Stage ``teacher`` (the only one so far)
trains the conditional latent diffusion with Plaid's NELBO.
"""

from __future__ import annotations

import logging
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
from specforge.algorithms.flashmap.providers import (
    _TARGET_ONLY_FIELDS,
    _decorrelate_ranks,
    minimum_loss_tokens,
    resolve_capture_layers,
)
from specforge.data.loss_mask import has_consecutive_supervised_tokens

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "flashplaid"
DRAFT_ARCHITECTURE = "FlashPlaidDraftModel"
# DFlash/FlashMap's slot decay; ``loss_decay_gamma: null`` means this gamma.
DEFAULT_LOSS_DECAY_GAMMA = 7.0
# Plaid: weight decay of the schedule endpoints (the table and the interior
# network take none).
ENDPOINT_WEIGHT_DECAY = 1e-3


def build_step(wrapped_model, *, target_head=None, **_options):
    del target_head
    from specforge.training.strategies.base import FlashPlaidTrainStrategy

    return FlashPlaidTrainStrategy(wrapped_model)


def resume_contract(config, draft_model, training_model):
    """Persist the resolved FlashPlaid architecture, stage and objective."""

    return {
        "flashplaid_stage": str(training_model.stage),
        "flashplaid_target_layer_ids": tuple(
            int(layer_id) for layer_id in draft_model.target_layer_ids
        ),
        "flashplaid_block_size": int(training_model.block_size),
        "flashplaid_latent_dim": int(training_model.latent_dim),
        "flashplaid_num_hidden_layers": int(draft_model.config.num_hidden_layers),
        "flashplaid_hidden_size": int(draft_model.config.hidden_size),
        "flashplaid_attention_backend": str(training_model.attention_backend),
        "flashplaid_num_anchors": int(training_model.num_anchors),
        "flashplaid_loss_decay_gamma": training_model.loss_decay_gamma,
        "flashplaid_adaptive_recon": bool(training_model.adaptive_recon),
    }


def populate_target_defaults(payload, target_config, config):
    """Derive a FlashPlaid draft config from the target's: a Plaid-sized DiT
    (6 blocks, width 768, 12 heads) over five target layers."""
    from specforge.modeling.draft.dflash import build_target_layer_ids

    num_target_layers = int(target_config.num_hidden_layers)
    payload["target_hidden_size"] = payload.pop("hidden_size")
    for field in _TARGET_ONLY_FIELDS:
        payload.pop(field, None)
    payload["target_model"] = config.model.target_model_path
    payload["target_layer_ids"] = build_target_layer_ids(num_target_layers, 5)


def build_draft(config, draft_config):
    from specforge.algorithms.model_providers import build_registered_draft

    return build_registered_draft(config, draft_config)


def initialize_latent_from_target(draft_model, target_embedding) -> None:
    """Start the latent table as the target embedding's top principal
    subspace: ``E`` centred, projected on the leading ``latent_dim``
    eigenvectors of its covariance, rows normalized. Deterministic (float64
    ``eigh`` on the CPU), so every rank computes the same table."""
    import torch

    with torch.no_grad():
        weight = target_embedding.float()
        centered = weight - weight.mean(dim=0, keepdim=True)
        covariance = (centered.T @ centered).double().cpu()
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -draft_model.latent_dim :].flip(-1)
        projected = centered @ basis.to(device=weight.device, dtype=torch.float32)
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-8)
        draft_model.latent_embed.weight.copy_(projected.to(draft_model.latent_embed.weight))


def tag_aux_parameters(draft_model, learning_rate: float) -> None:
    """Give the embedding table and the schedule their own optimizer group
    (``_specforge_lr`` / ``_specforge_weight_decay``, read by BF16Optimizer)."""
    for name, parameter in draft_model.aux_parameters().items():
        parameter._specforge_lr = float(learning_rate)
        is_endpoint = name.endswith(("gamma_0", "gamma_1"))
        parameter._specforge_weight_decay = ENDPOINT_WEIGHT_DECAY if is_endpoint else 0.0


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    from specforge.algorithms.flashplaid.model import OnlineFlashPlaidModel
    from specforge.algorithms.model_providers import (
        AlgorithmModelParts,
        _device,
        _torch_dtype,
    )
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    del draft_config, target_config, tokenizer
    if config.model.vocab_mapping_path:
        raise ValueError(
            "FlashPlaid trains over the full target vocabulary; restrict the "
            "drafting vocabulary at serving time instead"
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
    gamma = training.loss_decay_gamma
    if gamma is None:
        gamma = DEFAULT_LOSS_DECAY_GAMMA
        logger.info("FlashPlaid: training.loss_decay_gamma=null -> %s", gamma)
    model = OnlineFlashPlaidModel(
        draft_model,
        target_parts.lm_head,
        target_parts.embed_tokens,
        stage=training.flashplaid_stage,
        attention_backend=training.attention_backend,
        num_anchors=int(training.num_anchors),
        loss_decay_gamma=gamma,
        objective_chunk_blocks=training.objective_chunk_blocks,
        adaptive_recon=training.flashplaid_adaptive_recon,
        recon_fraction=training.flashplaid_recon_fraction,
    ).to(device=_device(), dtype=_torch_dtype(config))
    draft_model.restore_fp32_modules()
    if not config.model.draft_checkpoint_path:
        # A fresh run (a resume restores the trained table over this).
        initialize_latent_from_target(draft_model, target_parts.embed_tokens.weight)
        logger.info("FlashPlaid: latent table initialized from the target embedding PCA")
    if not training.flashplaid_learn_endpoints:
        schedule = draft_model.noise_schedule
        schedule.gamma_0.requires_grad_(False)
        schedule.gamma_1.requires_grad_(False)
        logger.info(
            "FlashPlaid: schedule endpoints frozen at gamma_0=%.3f, gamma_1=%.3f",
            float(schedule.gamma_0), float(schedule.gamma_1),
        )
    tag_aux_parameters(draft_model, training.flashplaid_aux_lr)
    _decorrelate_ranks(int(training.seed))
    return AlgorithmModelParts(model=model, capture_layers=list(draft_model.target_layer_ids))


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
                    model_type="flashplaid",
                    num_hidden_layers=6,
                    populate=populate_target_defaults,
                ),
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


__all__ = [
    "algorithm_providers",
    "algorithm_spec",
    "create_registration",
    "initialize_latent_from_target",
    "tag_aux_parameters",
]
