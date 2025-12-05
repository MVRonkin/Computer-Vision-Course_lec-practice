# setup.py или в начале скрипта
import os
import random
import numpy as np
import torch
from PIL import Image
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

    # История для текущей эпохи
    history = {
        'batch_losses': [],
        'batch_metrics': {k: [] for k in metrics}
    }

    for i, (x, y) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            y_pred = model(x)
            loss = criterion(y_pred, y) / accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Метрики для батча
        with torch.no_grad():
            for name, fn in metrics.items():
                batch_metric = fn(y_pred, y).item()
                history['batch_metrics'][name].append(batch_metric)
        
        batch_loss = loss.item() * accumulation_steps
        history['batch_losses'].append(batch_loss)

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

        epoch_loss += batch_loss

        # Метрики
        with torch.no_grad():
            for name, fn in metrics.items():
                epoch_metrics[name] += fn(y_pred, y).item()

    return epoch_loss / n_batches, {k: v / n_batches for k, v in epoch_metrics.items()}, history


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, metrics, device, *, ema=None):
    model.eval()
    if ema:
        ema.apply_shadow()

    epoch_loss = 0.0
    epoch_metrics = {k: 0.0 for k in metrics}
    n_batches = len(dataloader)

    # История для текущей эпохи
    history = {
        'batch_losses': [],
        'batch_metrics': {k: [] for k in metrics}
    }

    for x, y in tqdm(dataloader, desc="Val", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)

        batch_loss = loss.item()
        history['batch_losses'].append(batch_loss)
        
        for name, fn in metrics.items():
            batch_metric = fn(y_pred, y).item()
            history['batch_metrics'][name].append(batch_metric)

        epoch_loss += batch_loss
        for name, fn in metrics.items():
            epoch_metrics[name] += fn(y_pred, y).item()

    if ema:
        ema.restore()

    return epoch_loss / n_batches, {k: v / n_batches for k, v in epoch_metrics.items()}, history

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
    return_batch_history: bool = False,  # <-- НОВЫЙ ПАРАМЕТР
):
    """
    Обучает модель с поддержкой различных опций и логирования.

    Parameters:
    -----------
    model : torch.nn.Module
        Модель для обучения
    train_loader : DataLoader
        Загрузчик обучающих данных
    val_loader : DataLoader
        Загрузчик валидационных данных
    optimizer : torch.optim.Optimizer
        Оптимизатор
    criterion : callable
        Функция потерь
    metrics : dict
        Словарь метрик {'name': function}
    epochs : int, default=10
        Количество эпох
    scheduler : torch.optim.lr_scheduler, optional
        Планировщик скорости обучения
    device : torch.device
        Устройство для обучения
    checkpoint_path : str, default="best.pt"
        Путь для сохранения лучшей модели
    monitor_metric : str, default="acc"
        Метрика для мониторинга (для сохранения лучшей модели)
    mode : str, default="max"
        Режим мониторинга ("max" или "min")
    patience : int, default=10
        Количество эпох до остановки при отсутствии улучшений
    min_delta : float, default=1e-4
        Минимальное изменение для учета улучшения
    grad_clip : float, optional
        Градиентный клиппинг
    use_amp : bool, default=False
        Использовать автоматическое масштабирование точности (AMP)
    ema_decay : float, optional
        Параметр экспоненциального скользящего среднего
    accumulation_steps : int, default=1
        Количество шагов для накопления градиентов
    verbose : bool, default=True
        Печатать прогресс
    return_batch_history : bool, default=False
        Если True, возвращает историю по батчам в дополнение к истории по эпохам
    """
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

    # Если нужна история по батчам
    if return_batch_history:
        batch_history = {
            "train": [],
            "val": []
        }

    for epoch in range(epochs):
        start = time.time()

        # --- Train ---
        train_loss, train_metrics, train_batch_history = train_epoch(
            model, train_loader, optimizer, criterion, metrics, device,
            use_amp=use_amp, grad_clip=grad_clip, ema=ema,
            accumulation_steps=accumulation_steps
        )

        # --- Validate ---
        val_loss, val_metrics, val_batch_history = evaluate_epoch(
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

        # Сохраняем историю по батчам, если нужно
        if return_batch_history:
            batch_history["train"].append(train_batch_history)
            batch_history["val"].append(val_batch_history)

        # --- Early stopping & checkpoint ---
        current_score = val_loss if monitor_metric == "loss" else val_metrics[monitor_metric]
        improved = (current_score > best_score + min_delta) if mode == "max" else (current_score < best_score - min_delta)

        if improved:
            best_score = current_score
            best_epoch = epoch
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_score": best_score,
                "ema_shadow": ema.shadow if ema else None,
                "history": history,
            }
            if return_batch_history:
                checkpoint_data["batch_history"] = batch_history
            torch.save(checkpoint_data, checkpoint_path)
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
    
    if return_batch_history:
        return df, batch_history
    else:
        return df
    # if return_batch_history:
    #     # Объединяем историю по батчам в один DataFrame
    #     all_batch_data = []
    #     for epoch_idx, (train_bh, val_bh) in enumerate(zip(batch_history["train"], batch_history["val"])):
    #         # Для каждого эпоха объединяем train и val данные
    #         epoch_data = {}
    #         # Добавляем данные тренировки
    #         for metric_name, values in train_bh['batch_metrics'].items():
    #             epoch_data[f'train_batch_{metric_name}'] = values
    #         epoch_data['train_batch_losses'] = train_bh['batch_losses']
            
    #         # Добавляем данные валидации
    #         for metric_name, values in val_bh['batch_metrics'].items():
    #             epoch_data[f'val_batch_{metric_name}'] = values
    #         epoch_data['val_batch_losses'] = val_bh['batch_losses']
            
    #         # Добавляем индекс эпохи
    #         epoch_data['epoch'] = epoch_idx
    #         all_batch_data.append(epoch_data)
        
    #     batch_df = pd.DataFrame(all_batch_data)
    #     batch_df.attrs.update(best_epoch=best_epoch, best_score=best_score, monitor_metric=monitor_metric)
    #     return batch_df
    # else:
    #     df = pd.DataFrame(history)
    #     df.attrs.update(best_epoch=best_epoch, best_score=best_score, monitor_metric=monitor_metric)
    #     return df



def plot_batch_history(batch_history, metric_name='acc', window=10):
    """
    Строит графики для истории по батчам.
    
    Parameters:
    -----------
    batch_history : dict
        История по батчам в формате {'train': [...], 'val': [...]}
    metric_name : str, default='acc'
        Название метрики для отображения
    window : int, default=10
        Размер окна для скользящего среднего
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Подготовка данных с масштабированием по эпохам
    all_train_losses = []
    all_val_losses = []
    all_train_metrics = []
    all_val_metrics = []
    train_batch_positions = []  # Позиции батчей в масштабе эпох
    val_batch_positions = []    # Позиции батчей в масштабе эпох
    
    # Для тренировки
    current_pos = 0
    for epoch_idx, epoch_data in enumerate(batch_history['train']):
        n_train_batches = len(epoch_data['batch_losses'])
        batch_step = 1.0 / n_train_batches  # Шаг для масштабирования
        
        for i, loss in enumerate(epoch_data['batch_losses']):
            all_train_losses.append(loss)
            train_batch_positions.append(epoch_idx + i * batch_step)
        
        if metric_name in epoch_data['batch_metrics']:
            for i, metric_val in enumerate(epoch_data['batch_metrics'][metric_name]):
                all_train_metrics.append(metric_val)
    
    # Для валидации
    current_pos = 0
    for epoch_idx, epoch_data in enumerate(batch_history['val']):
        n_val_batches = len(epoch_data['batch_losses'])
        batch_step = 1.0 / n_val_batches  # Шаг для масштабирования
        
        for i, loss in enumerate(epoch_data['batch_losses']):
            all_val_losses.append(loss)
            val_batch_positions.append(epoch_idx + i * batch_step)
        
        if metric_name in epoch_data['batch_metrics']:
            for i, metric_val in enumerate(epoch_data['batch_metrics'][metric_name]):
                all_val_metrics.append(metric_val)
    
    # Создаем 2 графика
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # График 1: Loss
    ax1.plot(train_batch_positions, all_train_losses, label='Train Loss', alpha=0.6, color='blue', marker='o', markersize=2)
    ax1.plot(val_batch_positions, all_val_losses, label='Val Loss', alpha=0.6, color='red', marker='o', markersize=2)
    
    # Скользящее среднее для loss
    if len(all_train_losses) >= window:
        train_loss_ma = np.convolve(all_train_losses, np.ones(window)/window, mode='valid')
        ax1.plot(train_batch_positions[window-1:len(train_loss_ma)+window-1], train_loss_ma, 
                label=f'Train Loss MA ({window})', color='darkblue', linewidth=2)
    
    if len(all_val_losses) >= window:
        val_loss_ma = np.convolve(all_val_losses, np.ones(window)/window, mode='valid')
        ax1.plot(val_batch_positions[window-1:len(val_loss_ma)+window-1], val_loss_ma, 
                label=f'Val Loss MA ({window})', color='darkred', linewidth=2)
    
    ax1.set_title('Batch Loss')
    ax1.set_xlabel('Epoch (scaled by batch count)')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Метрика
    ax2.plot(train_batch_positions[:len(all_train_metrics)], all_train_metrics, 
             label=f'Train {metric_name}', alpha=0.6, color='blue', marker='o', markersize=2)
    ax2.plot(val_batch_positions[:len(all_val_metrics)], all_val_metrics, 
             label=f'Val {metric_name}', alpha=0.6, color='red', marker='o', markersize=2)
    
    # Скользящее среднее для метрики
    if len(all_train_metrics) >= window:
        train_metric_ma = np.convolve(all_train_metrics, np.ones(window)/window, mode='valid')
        ax2.plot(train_batch_positions[window-1:len(train_metric_ma)+window-1], train_metric_ma, 
                label=f'Train {metric_name} MA ({window})', color='darkblue', linewidth=2)
    
    if len(all_val_metrics) >= window:
        val_metric_ma = np.convolve(all_val_metrics, np.ones(window)/window, mode='valid')
        ax2.plot(val_batch_positions[window-1:len(val_metric_ma)+window-1], val_metric_ma, 
                label=f'Val {metric_name} MA ({window})', color='darkred', linewidth=2)
    
    ax2.set_title(f'Batch {metric_name}')
    ax2.set_xlabel('Epoch (scaled by batch count)')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
        
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