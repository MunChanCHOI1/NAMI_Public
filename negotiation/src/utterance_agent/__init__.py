"""Utterance-reading agent for the synthetic buyer-side negotiation experiment.

NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only. No real-model
answer to this package's requests exists. Design: negotiation/docs/UTTERANCE_AGENT_DESIGN.md.

The package lives next to `negotiation_agent` on purpose: none of its files is matched by the
source globs that the tuning runner hashes, so adding it does not change that runner's inputs.
"""

PROTOCOL_VERSION = "utterance-2026-11-18.v1"
GPU_NOTICE = "NOT EXECUTED ON A GPU as of 2026-11-18: authored and verified on CPU only."
