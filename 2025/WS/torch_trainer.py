# setup.py или в начале скрипта
import os
import random
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import pandas as pd
import time
from pathlib import Path
import timm
import torch.nn as nn
from tqdm.auto import tqdm

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



def init_classifier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        # Инициализация весов нормальным распределением
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            # Инициализация смещения нулями
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        # Инициализация gamma (weight) = 1 и beta (bias) = 0
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
class EMA:
    """
    Класс для реализации Exponential Moving Average (EMA) для параметров модели PyTorch.

    EMA помогает улучшить устойчивость и качество модели во время обучения,
    усредняя параметры модели с использованием экспоненциального скользящего среднего.
    Это особенно полезно в задачах, где важно избежать переобучения или улучшить сходимость.
    """

    def __init__(self, model, decay=0.999):
        """
        Инициализирует EMA для заданной модели.

        Args:
            model (torch.nn.Module): PyTorch-модель, для которой будет вычисляться EMA.
            decay (float, optional): Коэффициент затухания для EMA. По умолчанию 0.999.
                                     Чем ближе к 1, тем сильнее усреднение (меньше влияние новых параметров).
        """
        self.model = model
        self.decay = decay
        self.shadow = {}  # Словарь для хранения EMA значений параметров
        self.backup = {}  # Словарь для временного хранения оригинальных значений (например, при валидации)
        self.register()

    def register(self):
        """
        Инициализирует словарь self.shadow, копируя текущие значения обучаемых параметров модели.
        Вызывается один раз при создании экземпляра EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        Обновляет EMA значения параметров модели.
        Вызывается после каждого шага оптимизации (обычно после optimizer.step()).
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Обновляем EMA: новое значение = (1 - коэффициент) * текущее + коэффициент * старое_среднее
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        """
        Применяет EMA значения параметров к модели.
        Полезно перед оценкой модели (валидацией или тестом), чтобы использовать усреднённые параметры.
        Оригинальные параметры сохраняются в self.backup.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()  # Сохраняем оригинальные значения
                param.data = self.shadow[name]  # Применяем EMA

    def restore(self):
        """
        Восстанавливает оригинальные значения параметров модели из self.backup.
        Обычно вызывается после apply_shadow(), чтобы вернуть модель к её текущему обученному состоянию.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]  # Восстанавливаем оригинальные значения



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
    start_from_checkpoint: str = None,  # <-- НОВЫЙ ПАРАМЕТР
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
        Количество эпох (считая от start_epoch)
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
    start_from_checkpoint : str, optional
        Путь к checkpoint файлу для продолжения обучения
    """
    assert mode in ("max", "min")
    assert monitor_metric in list(metrics.keys()) + ["loss"], "Invalid monitor_metric"

    # Инициализация
    ema = EMA(model, decay=ema_decay) if ema_decay else None
    best_score = -float('inf') if mode == "max" else float('inf')
    patience_counter = 0
    best_epoch = 0
    start_epoch = 0  # Начинаем с 0

    # Загружаем checkpoint если указан
    if start_from_checkpoint is not None:
        if os.path.exists(start_from_checkpoint):
            checkpoint = torch.load(start_from_checkpoint, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'ema_shadow' in checkpoint and ema is not None:
                ema.shadow = checkpoint['ema_shadow']
                ema.collected_params = checkpoint.get('ema_params', [])
            
            # Восстанавливаем историю
            if 'history' in checkpoint:
                history = checkpoint['history']
                # Обновляем best_score и best_epoch из checkpoint
                best_score = checkpoint.get('best_score', best_score)
                best_epoch = checkpoint.get('best_epoch', best_epoch)
                # Обновляем start_epoch
                start_epoch = checkpoint.get('epoch', 0) + 1  # Продолжаем с следующей эпохи
            else:
                # Если в checkpoint нет истории, инициализируем заново
                history = {
                    "train_loss": [], "val_loss": [], "lr": [], "epoch_time": []
                }
                for name in metrics:
                    history[f"train_{name}"] = []
                    history[f"val_{name}"] = []
            
            if return_batch_history and 'batch_history' in checkpoint:
                batch_history = checkpoint['batch_history']
            elif return_batch_history:
                batch_history = {"train": [], "val": []}
            
            if verbose:
                print(f"Loaded checkpoint '{start_from_checkpoint}' | Starting from epoch {start_epoch} | Best score: {best_score:.6f}")
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {start_from_checkpoint}")
    else:
        # Инициализация истории
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

    for epoch in range(start_epoch, start_epoch + epochs):
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
                # scheduler.step()
                if isinstance(scheduler, timm.scheduler.scheduler.Scheduler):
                    # timm-овский планировщик, требует номер эпохи c 1
                    scheduler.step(epoch + 1)
                else:
                    # Стандартный PyTorch-планировщик, вызывается без аргументов
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
                "best_epoch": best_epoch,
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

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Загружает сохраненную модель и состояние тренировки из checkpoint.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Модель для загрузки весов
    optimizer : torch.optim.Optimizer
        Оптимизатор для загрузки состояния
    scheduler : torch.optim.lr_scheduler
        Планировщик для загрузки состояния (может быть None)
    checkpoint_path : str
        Путь к файлу checkpoint
    device : torch.device
        Устройство для загрузки
    
    Returns:
    --------
    dict : Словарь с информацией из checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Загружаем состояние модели
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Загружаем состояние оптимизатора
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Загружаем состояние планировщика (если есть)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Возвращаем дополнительную информацию
    info = {
        'epoch': checkpoint['epoch'],
        'best_score': checkpoint['best_score'],
        'best_epoch': checkpoint['best_epoch'], 
        'history': checkpoint.get('history', {}),
        'batch_history': checkpoint.get('batch_history', None),  # Если был сохранен
        'ema_shadow': checkpoint.get('ema_shadow', None)
    }
    
    return info
    
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
    
    # Скользящее среднее и дисперсия для loss
    if len(all_train_losses) >= window:
        train_loss_ma = np.convolve(all_train_losses, np.ones(window)/window, mode='valid')
        # Скользящая дисперсия
        train_loss_var = np.array([np.var(all_train_losses[i:i+window]) for i in range(len(all_train_losses)-window+1)])
        ax1.plot(train_batch_positions[window-1:len(train_loss_ma)+window-1], train_loss_ma, 
                label=f'Train Loss MA ({window})', color='darkblue', linewidth=2)
        ax1.fill_between(train_batch_positions[window-1:len(train_loss_ma)+window-1], 
                        train_loss_ma - np.sqrt(train_loss_var), 
                        train_loss_ma + np.sqrt(train_loss_var), 
                        color='darkblue', alpha=0.2, label=f'Train Loss ±σ')
    
    if len(all_val_losses) >= window:
        val_loss_ma = np.convolve(all_val_losses, np.ones(window)/window, mode='valid')
        # Скользящая дисперсия
        val_loss_var = np.array([np.var(all_val_losses[i:i+window]) for i in range(len(all_val_losses)-window+1)])
        ax1.plot(val_batch_positions[window-1:len(val_loss_ma)+window-1], val_loss_ma, 
                label=f'Val Loss MA ({window})', color='darkred', linewidth=2)
        ax1.fill_between(val_batch_positions[window-1:len(val_loss_ma)+window-1], 
                        val_loss_ma - np.sqrt(val_loss_var), 
                        val_loss_ma + np.sqrt(val_loss_var), 
                        color='darkred', alpha=0.2, label=f'Val Loss ±σ')
    
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
    
    # Скользящее среднее и дисперсия для метрики
    if len(all_train_metrics) >= window:
        train_metric_ma = np.convolve(all_train_metrics, np.ones(window)/window, mode='valid')
        # Скользящая дисперсия
        train_metric_var = np.array([np.var(all_train_metrics[i:i+window]) for i in range(len(all_train_metrics)-window+1)])
        ax2.plot(train_batch_positions[window-1:len(train_metric_ma)+window-1], train_metric_ma, 
                label=f'Train {metric_name} MA ({window})', color='darkblue', linewidth=2)
        ax2.fill_between(train_batch_positions[window-1:len(train_metric_ma)+window-1], 
                        train_metric_ma - np.sqrt(train_metric_var), 
                        train_metric_ma + np.sqrt(train_metric_var), 
                        color='darkblue', alpha=0.2, label=f'Train {metric_name} ±σ')
    
    if len(all_val_metrics) >= window:
        val_metric_ma = np.convolve(all_val_metrics, np.ones(window)/window, mode='valid')
        # Скользящая дисперсия
        val_metric_var = np.array([np.var(all_val_metrics[i:i+window]) for i in range(len(all_val_metrics)-window+1)])
        ax2.plot(val_batch_positions[window-1:len(val_metric_ma)+window-1], val_metric_ma, 
                label=f'Val {metric_name} MA ({window})', color='darkred', linewidth=2)
        ax2.fill_between(val_batch_positions[window-1:len(val_metric_ma)+window-1], 
                        val_metric_ma - np.sqrt(val_metric_var), 
                        val_metric_ma + np.sqrt(val_metric_var), 
                        color='darkred', alpha=0.2, label=f'Val {metric_name} ±σ')
    
    ax2.set_title(f'Batch {metric_name}')
    ax2.set_xlabel('Epoch (scaled by batch count)')
    ax2.set_ylabel(metric_name)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    

@torch.inference_mode()
def evaluate(model, dataloader, metrics, criterion=None, device=None, return_batch_metrics=False):
    """
    Полная валидация/оценка модели на всех батчах из dataloader.
    Использует torch.inference_mode() для максимальной эффективности инференса.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Модель для оценки
    dataloader : DataLoader
        Загрузчик данных для оценки
    metrics : dict
        Словарь метрик {'name': function}
    criterion : callable, optional
        Функция потерь
    device : torch.device
        Устройство для вычислений
    return_batch_metrics : bool, default=False
        Возвращать ли все значения метрик по батчам
    
    Returns:
    --------
    dict : Словарь с итоговыми метриками
        {
            'metrics': {название_метрики: среднее_значение},
            'all_metrics': {название_метрики: список всех batch-значений} (если return_batch_metrics)
        }
    """
    model.eval()

    total_loss = 0.0
    total_metrics = {k: 0.0 for k in metrics}
    n_batches = len(dataloader)

    # Сбор всех значений по батчам (если нужно)
    all_losses = [] if criterion and return_batch_metrics else None
    all_metrics = {k: [] for k in metrics} if return_batch_metrics else None

    for x, y in tqdm(dataloader, desc="Evaluate", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        y_pred = model(x)

        if criterion:
            loss = criterion(y_pred, y)
            batch_loss = loss.item()
            if all_losses is not None:
                all_losses.append(batch_loss)
            total_loss += batch_loss
        
        for name, fn in metrics.items():
            batch_metric_val = fn(y_pred, y).item()
            if all_metrics is not None:
                all_metrics[name].append(batch_metric_val)
            total_metrics[name] += batch_metric_val

    # Подсчет итоговых метрик
    avg_loss = total_loss / n_batches if criterion else None
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    result = {'metrics': avg_metrics}
    
    if avg_loss is not None:
        result['loss'] = avg_loss
    
    if return_batch_metrics:
        if all_losses is not None:
            result['all_losses'] = all_losses
        if all_metrics is not None:
            result['all_metrics'] = all_metrics

    return result

@torch.inference_mode()
def predict(model, dataloader, device, return_predictions=True, return_targets=False):
    """
    Выполняет предсказания модели на всех батчах из dataloader.
    Использует torch.inference_mode() для максимальной эффективности инференса.
    
    Parameters:
    -----------
    model : torch.nn.Module
        Модель для предсказания
    dataloader : DataLoader
        Загрузчик данных для предсказания
    device : torch.device
        Устройство для вычислений
    return_predictions : bool, default=True
        Возвращать ли предсказания
    return_targets : bool, default=False
        Возвращать ли целевые значения (если доступны в dataloader)
    
    Returns:
    --------
    dict : Словарь с результатами
        {
            'predictions': тензор всех предсказаний (если return_predictions=True),
            'targets': тензор всех целевых значений (если return_targets=True)
        }
    """
    model.eval()

    predictions = [] if return_predictions else None
    targets = [] if return_targets else None

    for batch in tqdm(dataloader, desc="Predict", leave=False):
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        else:
            x = batch[0]
            x = x.to(device, non_blocking=True)
            y = None

        y_pred = model(x)
        
        if return_predictions:
            predictions.append(y_pred.cpu())  # Перемещаем на CPU для сохранения памяти
        
        if return_targets and y is not None:
            targets.append(y.cpu())

    result = {}
    if predictions is not None:
        result['predictions'] = torch.cat(predictions, dim=0)
    if targets is not None:
        result['targets'] = torch.cat(targets, dim=0)

    return result        
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