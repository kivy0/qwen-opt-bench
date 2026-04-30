import logging
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler

from src.logger import HardwareMonitor, MetricsLogger

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        train_dataloader: Any,
        metrics_logger: MetricsLogger,
        num_steps: int = 1000,
        grad_accum_steps: int = 1,
        max_grad_norm: float | None = None,
        fp16: bool = False,
        bf16: bool = False,
        use_backward: bool = True,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.metrics_logger = metrics_logger
        self.num_steps = num_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_backward = use_backward

        self.use_amp = fp16 or bf16
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.scaler = torch.amp.GradScaler(
            enabled=fp16 and self.device == "cuda",
        )
        self.hw_monitor = HardwareMonitor(self.device)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """Perform forward pass and return loss."""
        with torch.amp.autocast(
            device_type=self.device,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            outputs = self.model(**batch)
            loss = outputs.loss
        return loss

    def backward(self, loss: torch.Tensor) -> None:
        """Perform backward pass with optional gradient scaling."""
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def training_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """Execute one training step: forward + backward (if enabled)."""
        loss = self.forward(batch)

        if self.use_backward:
            self.backward(loss / self.grad_accum_steps)

        return loss.detach()

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping and scaling."""
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)

        if self.max_grad_norm is not None:
            clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        if self.scaler.is_enabled():
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    def train(self) -> None:
        logger.info("Training started")
        self.model.train()
        self.optimizer.zero_grad()

        global_step = 0
        micro_step = 0
        seen_examples = 0
        step_tokens = 0

        accumulated_loss = torch.tensor(0.0, device=self.device)
        dataloader_iterator = iter(self.train_dataloader)

        self.hw_monitor.reset()

        while global_step < self.num_steps:
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.train_dataloader)
                batch = next(dataloader_iterator)

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            current_batch_size = len(next(iter(batch.values())))
            seen_examples += current_batch_size

            if "attention_mask" in batch:
                step_tokens += int(batch["attention_mask"].sum().item())
            elif "input_ids" in batch:
                step_tokens += batch["input_ids"].numel()

            loss = self.training_step(batch)

            accumulated_loss += loss.detach()
            micro_step += 1

            if micro_step % self.grad_accum_steps == 0:
                self.optimizer_step()
                global_step += 1

                hw_metrics = self.hw_monitor.collect()

                avg_loss = accumulated_loss / self.grad_accum_steps

                metrics = self._collect_train_metrics(
                    loss=avg_loss,
                    hw_metrics=hw_metrics,
                    step=global_step,
                    seen_examples=seen_examples,
                    step_tokens=step_tokens,
                )

                self._log_step(metrics, step=global_step)

                step_tokens = 0
                accumulated_loss.zero_()
                self.hw_monitor.reset()

    def _collect_train_metrics(
        self,
        loss: torch.Tensor,
        hw_metrics: dict[str, float],
        step: int,
        seen_examples: int,
        step_tokens: int,
    ) -> dict[str, Any]:

        metrics: dict[str, Any] = {
            "train_loss": round(loss.item(), 6),
            "seen_examples": seen_examples,
            "step_tokens": step_tokens,
            **hw_metrics,
        }

        # collect unique LR values
        unique_lrs = sorted(
            list(set(group["lr"] for group in self.optimizer.param_groups)),
            reverse=True,
        )

        if len(unique_lrs) == 1:
            metrics["learning_rate"] = unique_lrs[0]
        else:
            for i, lr in enumerate(unique_lrs):
                metrics[f"learning_rate_{i}"] = lr

        if hw_metrics.get("step_time_sec", 0) > 0:
            step_time = hw_metrics["step_time_sec"]
            metrics["throughput_steps_per_sec"] = round(1.0 / step_time, 2)
            metrics["throughput_tokens_per_sec"] = round(step_tokens / step_time)
        return metrics

    def _log_step(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """Log metrics"""
        self.metrics_logger.log(metrics, step=step)

        lr_keys = sorted([k for k in metrics.keys() if k.startswith("learning_rate")])
        lr_str = " | ".join(
            f"{k.replace('learning_rate', 'lr')}={metrics[k]:.2e}" for k in lr_keys
        )

        logger.info(
            "step=%d | samples=%d | loss=%.4f | %s | step_time=%.3fs | "
            "peak_mem=%.0fMB | gpu_util=%.1f%%",
            step,
            metrics.get("seen_examples", 0),
            metrics.get("train_loss", float("nan")),
            lr_str,
            metrics.get("step_time_sec", 0.0),
            metrics.get("gpu_peak_memory_mb", 0.0),
            metrics.get("gpu_utilization_pct", 0.0),
        )
