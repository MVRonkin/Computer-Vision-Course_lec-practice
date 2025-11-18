# setup.py или в начале скрипта
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def setup_experiment(
    seed: int = 42,
    deterministic: bool = False,   # benchmark=True + deterministic=False = быстрее
    allow_tf32: bool = True,
    device_preference: str = "auto"
) -> torch.device:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = deterministic
        cudnn.benchmark = not deterministic
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_preference)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")

    print(f"[✓] Device: {device} | Seed: {seed} | TF32: {allow_tf32}")
    return device


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

from tqdm.auto import tqdm

def train_epoch(model, dataloader, optimizer, criterion, metrics, device, *,
                use_amp=False, grad_clip=None, ema=None, accumulation_steps=1):
    model.train()
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == 'cuda') else None

    epoch_loss = 0.0
    epoch_metrics = {k: 0.0 for k in metrics}
    n_batches = len(dataloader)

    optimizer.zero_grad()

    for i, (x, y) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(x)
            loss = criterion(y_pred, y) / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Шаг оптимизатора
        if (i + 1) % accumulation_steps == 0 or (i + 1) == n_batches:
            if grad_clip:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if ema:
                ema.update()
            optimizer.zero_grad()

        # Метрики
        with torch.no_grad():
            for name, fn in metrics.items():
                epoch_metrics[name] += fn(y_pred, y).item()
        epoch_loss += loss.item() * accumulation_steps

    return epoch_loss / n_batches, {k: v / n_batches for k, v in epoch_metrics.items()}


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, metrics, device, *, ema=None):
    model.eval()
    if ema:
        ema.apply_shadow()

    epoch_loss = 0.0
    epoch_metrics = {k: 0.0 for k in metrics}
    n_batches = len(dataloader)

    for x, y in tqdm(dataloader, desc="Val", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        epoch_loss += loss.item()
        for name, fn in metrics.items():
            epoch_metrics[name] += fn(y_pred, y).item()

    if ema:
        ema.restore()

    return epoch_loss / n_batches, {k: v / n_batches for k, v in epoch_metrics.items()}

import pandas as pd
import time
from pathlib import Path

def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    metrics,
    *,
    epochs: int = 10,
    scheduler=None,
    device: torch.device,
    checkpoint_path: str = "best.pt",
    monitor_metric: str = "acc",
    mode: str = "max",
    patience: int = 10,
    min_delta: float = 1e-4,
    grad_clip: float = None,
    use_amp: bool = False,
    ema_decay: float = None,
    accumulation_steps: int = 1,
    verbose: bool = True,
):
    assert mode in ("max", "min")
    assert monitor_metric in list(metrics.keys()) + ["loss"], "Invalid monitor_metric"

    # Инициализация
    ema = EMA(model, decay=ema_decay) if ema_decay else None
    best_score = -float('inf') if mode == "max" else float('inf')
    patience_counter = 0
    best_epoch = 0

    history = {
        "train_loss": [], "val_loss": [], "lr": [], "epoch_time": []
    }
    for name in metrics:
        history[f"train_{name}"] = []
        history[f"val_{name}"] = []

    for epoch in range(epochs):
        start = time.time()

        # --- Train ---
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, metrics, device,
            use_amp=use_amp, grad_clip=grad_clip, ema=ema,
            accumulation_steps=accumulation_steps
        )

        # --- Validate ---
        val_loss, val_metrics = evaluate_epoch(
            model, val_loader, criterion, metrics, device, ema=ema
        )

        # --- Scheduler ---
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                score = val_loss if monitor_metric == "loss" else val_metrics[monitor_metric]
                scheduler.step(score)
            else:
                scheduler.step()

        # --- Logging ---
        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        history["epoch_time"].append(elapsed)
        for name in metrics:
            history[f"train_{name}"].append(train_metrics[name])
            history[f"val_{name}"].append(val_metrics[name])

        # --- Early stopping & checkpoint ---
        current_score = val_loss if monitor_metric == "loss" else val_metrics[monitor_metric]
        improved = (current_score > best_score + min_delta) if mode == "max" else (current_score < best_score - min_delta)

        if improved:
            best_score = current_score
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_score": best_score,
                "ema_shadow": ema.shadow if ema else None,
                "history": history,
            }, checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # --- Print ---
        if verbose:
            print(f"Epoch {epoch+1:02d} | "
                  f"Time: {elapsed:.1f}s | LR: {lr:.2e} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val {monitor_metric}: {current_score:.4f} {'★' if improved else ''}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best: {best_score:.6f} at epoch {best_epoch+1}")
            break

    df = pd.DataFrame(history)
    df.attrs.update(best_epoch=best_epoch, best_score=best_score, monitor_metric=monitor_metric)
    return df


# def fit(
#     model,
#     train_loader,
#     val_loader,
#     optimizer,
#     criterion,
#     metrics,
#     *,
#     epochs: int = 10,
#     scheduler=None,
#     device: torch.device,
#     checkpoint_path: str = "best.pt",
#     monitor_metric: str = "acc",
#     mode: str = "max",
#     patience: int = 10,
#     min_delta: float = 1e-4,
#     grad_clip: float = None,
#     use_amp: bool = False,
#     ema_decay: float = None,
#     accumulation_steps: int = 1,
#     verbose: bool = True,
#     profile: bool = False,          # ← новый аргумент
#     profile_dir: str = "./log",     # ← куда сохранять трейсы
# ):
#     assert mode in ("max", "min")
#     assert monitor_metric in list(metrics.keys()) + ["loss"], "Invalid monitor_metric"

#     # Инициализация
#     ema = EMA(model, decay=ema_decay) if ema_decay else None
#     best_score = -float('inf') if mode == "max" else float('inf')
#     patience_counter = 0
#     best_epoch = 0

#     history = {
#         "train_loss": [], "val_loss": [], "lr": [], "epoch_time": []
#     }
#     for name in metrics:
#         history[f"train_{name}"] = []
#         history[f"val_{name}"] = []

#     # --- Инициализация профайлера ---
#     if profile:
#         profiler = torch.profiler.profile(
#             activities=[
#                 torch.profiler.ProfilerActivity.CPU,
#                 torch.profiler.ProfilerActivity.CUDA
#             ],
#             schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#             on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
#             record_shapes=True,
#             profile_memory=True,
#             with_stack=True,  # для анализа вызовов
#         )
#         profiler.start()
#         print(f"[Profiler] Запись трассировки в {profile_dir}")
#     else:
#         profiler = None

#     try:
#         for epoch in range(epochs):
#             start = time.time()

#             # --- Train ---
#             train_loss, train_metrics = train_epoch(
#                 model, train_loader, optimizer, criterion, metrics, device,
#                 use_amp=use_amp, grad_clip=grad_clip, ema=ema,
#                 accumulation_steps=accumulation_steps
#             )

#             # --- Validate ---
#             val_loss, val_metrics = evaluate_epoch(
#                 model, val_loader, criterion, metrics, device, ema=ema
#             )

#             # --- Шаг профайлера ---
#             if profiler is not None:
#                 profiler.step()

#             # --- Scheduler ---
#             if scheduler is not None:
#                 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     score = val_loss if monitor_metric == "loss" else val_metrics[monitor_metric]
#                     scheduler.step(score)
#                 else:
#                     scheduler.step()

#             # --- Logging ---
#             lr = optimizer.param_groups[0]["lr"]
#             elapsed = time.time() - start
#             history["train_loss"].append(train_loss)
#             history["val_loss"].append(val_loss)
#             history["lr"].append(lr)
#             history["epoch_time"].append(elapsed)
#             for name in metrics:
#                 history[f"train_{name}"].append(train_metrics[name])
#                 history[f"val_{name}"].append(val_metrics[name])

#             # --- Early stopping & checkpoint ---
#             current_score = val_loss if monitor_metric == "loss" else val_metrics[monitor_metric]
#             improved = (current_score > best_score + min_delta) if mode == "max" else (current_score < best_score - min_delta)

#             if improved:
#                 best_score = current_score
#                 best_epoch = epoch
#                 torch.save({
#                     "epoch": epoch,
#                     "model_state_dict": model.state_dict(),
#                     "optimizer_state_dict": optimizer.state_dict(),
#                     "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
#                     "best_score": best_score,
#                     "ema_shadow": ema.shadow if ema else None,
#                     "history": history,
#                 }, checkpoint_path)
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             # --- Print ---
#             if verbose:
#                 print(f"Epoch {epoch+1:02d} | "
#                       f"Time: {elapsed:.1f}s | LR: {lr:.2e} | "
#                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
#                       f"Val {monitor_metric}: {current_score:.4f} {'★' if improved else ''}")

#             if patience_counter >= patience:
#                 if verbose:
#                     print(f"Early stopping at epoch {epoch+1}. Best: {best_score:.6f} at epoch {best_epoch+1}")
#                 break

#     finally:
#         # --- Завершение профайлера ---
#         if profiler is not None:
#             profiler.stop()
#             print(f"[Profiler] Трассировка сохранена. Запустите: tensorboard --logdir={profile_dir}")

#     df = pd.DataFrame(history)
#     df.attrs.update(best_epoch=best_epoch, best_score=best_score, monitor_metric=monitor_metric)
#     return df