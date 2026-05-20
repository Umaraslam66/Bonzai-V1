"""Sub-bar 3 perplexity-gap evaluation harness (skeleton).

This package ships the gap-eval shell ahead of the training scaffold; it does
NOT depend on a trained model. The model_forward callable is injected at gap
computation time (see perplexity_gap.py).
"""

from __future__ import annotations
