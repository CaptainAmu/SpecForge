"""SGLang model implementations for SpecForge drafts.

``SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang`` makes sglang import
every module here and register their ``EntryClass`` models, so an exported
draft is served by architecture name like a built-in one:

* :mod:`.flashmap`: ``FlashMapDraftModel``, served by the DSPARK worker
  (``--speculative-algorithm DSPARK``); export with ``specforge.export.flashmap``.

The models use sglang internals of one release (``SUPPORTED_SGLANG``). sglang
logs and skips a model module that fails to import, which surfaces only later as
an unsupported architecture, so the version is also checked here, where a
mismatch stops the server with the reason.
"""

import os

SUPPORTED_SGLANG = "0.5.20"


def check_sglang_version() -> None:
    """Raise unless the installed sglang is the release these models target
    (``SPECFORGE_SGLANG_ANY_VERSION=1`` skips the check)."""
    from importlib.metadata import version

    found = version("sglang")
    if (
        found == SUPPORTED_SGLANG
        or os.environ.get("SPECFORGE_SGLANG_ANY_VERSION") == "1"
    ):
        return
    raise RuntimeError(
        f"specforge.serving.sglang needs sglang=={SUPPORTED_SGLANG} (its DSPARK "
        f"worker internals), found {found}. Install it alongside SpecForge without "
        f"SpecForge's sglang pin downgrading it: pip install sglang=={SUPPORTED_SGLANG}"
        " && pip install --no-deps -e <SpecForge checkout>"
    )


if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == __name__:
    check_sglang_version()
