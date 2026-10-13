"""SGLang model implementations for SpecForge drafts.

``SGLANG_EXTERNAL_MODEL_PACKAGE=specforge.serving.sglang`` makes sglang import
every module here and register their ``EntryClass`` models, so an exported
draft is served by architecture name like a built-in one:

* :mod:`.flashmap`: ``FlashMapDraftModel``, served by the DSPARK worker
  (``--speculative-algorithm DSPARK``); export with ``specforge.export.flashmap``.
"""
