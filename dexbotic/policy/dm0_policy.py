from __future__ import annotations

from dexbotic.policy.pi0_policy import Pi0Policy


class DM0Policy(Pi0Policy):
    """
    B-family policy for DM0.
    Identical to Pi0Policy except the tokenization message format uses
    {"from": "human", "value": text} (DM0Tokenization convention).
    """

    state_used = False
    state_required = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dim = None

    def _build_message(self, text: str) -> dict:
        return {"from": "human", "value": text}
