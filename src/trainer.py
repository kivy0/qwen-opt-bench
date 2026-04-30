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
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.metrics_logger = metrics_logger
        self.num_steps = num_steps
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_backward = use_backward

        self.device = device
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
                )

                self._log_step(metrics, step=global_step)

                accumulated_loss.zero_()
                self.hw_monitor.reset()

    def _collect_train_metrics(
        self,
        loss: torch.Tensor,
        hw_metrics: dict[str, float],
        step: int,
        seen_examples: int,
    ) -> dict[str, Any]:
        lr = self.scheduler.get_last_lr()[0]

        metrics: dict[str, Any] = {
            "train_loss": round(loss.item(), 6),
            "learning_rate": lr,
            "seen_examples": seen_examples,
            **hw_metrics,
        }

        if hw_metrics.get("step_time_sec", 0) > 0:
            metrics["throughput_steps_per_sec"] = round(
                1.0 / hw_metrics["step_time_sec"], 2
            )

        return metrics

    def _log_step(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        """Log metrics"""
        self.metrics_logger.log(metrics, step=step)

        logger.info(
            "step=%d | loss=%.4f | lr=%.2e | step_time=%.3fs | "
            "peak_mem=%.0fMB | gpu_util=%.1f%%",
            step,
            metrics.get("train_loss", float("nan")),
            metrics.get("learning_rate", float("nan")),
            metrics.get("step_time_sec", 0.0),
            metrics.get("gpu_peak_memory_mb", 0.0),
            metrics.get("gpu_utilization_pct", 0.0),
        )
