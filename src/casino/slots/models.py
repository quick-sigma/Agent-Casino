from pydantic import BaseModel, ConfigDict, Field, model_validator
from casino.slots.spec import MIN_BET, OutcomeKind, Symbol


class SpinRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_seed: str = Field(strict=True, min_length=1, max_length=64)
    generation: int = Field(strict=True, ge=0)
    turn: int = Field(strict=True, ge=0)
    bet: int = Field(strict=True, ge=MIN_BET)


class SpinResult(SpinRequest):
    reels: tuple[Symbol, Symbol, Symbol]
    kind: OutcomeKind
    multiplier: int = Field(strict=True, ge=0)
    payout: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def _payout_matches_arithmetic(self) -> "SpinResult":
        if self.payout != self.bet * self.multiplier:
            raise ValueError("payout debe ser exactamente bet * multiplier")
        return self


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: OutcomeKind
    multiplier: int = Field(strict=True, ge=0)