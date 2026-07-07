from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - optional during unit tests
    class TrainerCallback:  # type: ignore[override]
        pass


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize_reward_logs(
    reward_logs: list[dict[str, Any]],
    prefix: str = "grpo_reward",
) -> dict[str, float]:
    if not reward_logs:
        return {}

    metrics: dict[str, float] = {
        f"{prefix}/samples": float(len(reward_logs)),
    }

    rewards = [float(item["reward"]) for item in reward_logs if "reward" in item]
    if rewards:
        metrics[f"{prefix}/reward_mean"] = _mean(rewards)
        metrics[f"{prefix}/reward_min"] = float(min(rewards))
        metrics[f"{prefix}/reward_max"] = float(max(rewards))

    component_counts = Counter(str(item.get("component", "unknown")) for item in reward_logs)
    for component, count in component_counts.items():
        metrics[f"{prefix}/component/{component}_count"] = float(count)

    component_values: dict[str, list[float]] = {}
    for item in reward_logs:
        reward_components = item.get("reward_components") or {}
        for key, value in reward_components.items():
            numeric_value = _as_float(value)
            if numeric_value is None:
                continue
            component_values.setdefault(str(key), []).append(numeric_value)

    for key, values in component_values.items():
        metrics[f"{prefix}/{key}_mean"] = _mean(values)
        metrics[f"{prefix}/{key}_min"] = float(min(values))
        metrics[f"{prefix}/{key}_max"] = float(max(values))

    return metrics


class WandbRewardLoggerCallback(TrainerCallback):
    def __init__(self, reward_fn, prefix: str = "grpo_reward") -> None:
        self.reward_fn = reward_fn
        self.prefix = prefix
        self._last_logged_step: int | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._last_logged_step == state.global_step:
            return control

        reward_logs = getattr(self.reward_fn, "last_logs", None) or []
        metrics = summarize_reward_logs(reward_logs, prefix=self.prefix)
        if not metrics:
            return control

        try:
            import wandb
        except ImportError:
            return control

        if getattr(wandb, "run", None) is None:
            return control

        wandb.log(metrics, step=state.global_step, commit=False)
        self._last_logged_step = state.global_step
        return control
