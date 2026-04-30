from typing import Any, Callable

import torch


class HybridOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("optimizers must not be empty")

        self.optimizers = optimizers
        param_groups: list[dict[str, Any]] = []
        seen_params: set[int] = set()

        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    param_id = id(param)
                    if param_id in seen_params:
                        raise ValueError(
                            "the same parameter is present in more than one optimizer"
                        )
                    seen_params.add(param_id)

                param_groups.append(group)

        super().__init__(param_groups, defaults={})

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        for optimizer in self.optimizers:
            current_loss = optimizer.step(closure)
            if current_loss is not None:
                loss = current_loss
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        optimizer_states = state_dict["optimizers"]
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError("optimizer count mismatch")

        for optimizer, optimizer_state in zip(
            self.optimizers,
            optimizer_states,
            strict=True,
        ):
            optimizer.load_state_dict(optimizer_state)
