import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import gc
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def lr_finder(model, train_loader, optimizer, criterion, start_lr=1e-7, end_lr=10, num_iter=100,
              step_mode='exp', smooth_f=0.05, diverge_th=5, device='cuda', 
              accumulation_steps=1, use_amp=False, verbose=True, plot=True):
    """
    Поиск оптимальной скорости обучения для больших моделей с оптимизациями памяти.

    Parameters:
    - model: torch.nn.Module
    - train_loader: torch.utils.data.DataLoader
    - optimizer: torch.optim.Optimizer
    - criterion: loss function
    - start_lr: начальная скорость обучения
    - end_lr: конечная скорость обучения
    - num_iter: количество итераций
    - step_mode: 'exp' (экспоненциальный) или 'linear' (линейный)
    - smooth_f: коэффициент сглаживания для скользящего среднего
    - diverge_th: порог расхождения (в разах от минимального loss)
    - device: устройство ('cuda' или 'cpu')
    - accumulation_steps: количество шагов накопления градиентов
    - use_amp: использовать автоматическое масштабирование точности (AMP)
    - verbose: выводить прогресс
    - plot: строить график

    Returns:
    - lrs: список скоростей обучения
    - losses: список значений loss
    - best_lr: оптимальная скорость обучения

    

    Улучшения для функции lr_finder:
    
    1. **Оптимизации памяти для больших моделей:**
       - Использование градиентного накопления (--accumulation_steps)
       - AMP (Automatic Mixed Precision) для уменьшения использования памяти
       - Очистка кэша CUDA и сборка мусора после завершения
    
    2. **Улучшенная логика определения оптимального LR:**
       - Использование сглаженного loss для более стабильных результатов
       - Поиск точки перед резким ростом loss, а не просто минимального значения
       - Возможность настройки порога расхождения
    
    3. **Гибкость:**
       - Поддержка экспоненциального и линейного изменения LR
       - Настраиваемый сглаживающий коэффициент
       - Возможность остановки при расхождении
    
    4. **Улучшенная визуализация:**
       - Логарифмическая шкала для оси X
       - Отображение оптимального LR на графике
       - Настройка стиля сетки
    
    5. **Дополнительные рекомендации:**
       - Использовать меньший batch_size при поиске LR для экономии памяти
       - Протестировать на подмножестве данных
       - Использовать warm restarts после определения оптимального LR
       - Рассмотреть использование циклического LR после нахождения оптимального значения

    """
    
    # Сохраняем исходное состояние оптимизатора
    original_state = optimizer.state_dict()
    original_lr = optimizer.param_groups[0]['lr']
    
    # Инициализация AMP scaler
    scaler = GradScaler() if use_amp else None
    
    # Настройка scheduler для изменения LR
    if step_mode == 'exp':
        lr_multiplier = (end_lr / start_lr) ** (1.0 / num_iter)
    elif step_mode == 'linear':
        lr_multiplier = (end_lr - start_lr) / num_iter
    else:
        raise ValueError("step_mode должен быть 'exp' или 'linear'")
    
    # Подготовка модели
    model.train()
    model.to(device)
    
    # Списки для хранения результатов
    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0.0
    iteration = 0
    
    # Загрузчик итератор
    train_iter = iter(train_loader)
    
    # Основной цикл
    while iteration < num_iter:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Обновление LR
        if step_mode == 'exp':
            current_lr = start_lr * (lr_multiplier ** iteration)
        else:  # linear
            current_lr = start_lr + lr_multiplier * iteration
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        lrs.append(current_lr)
        
        # Прямой проход с AMP
        context = autocast() if use_amp else nullcontext()
        with context:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Нормализация loss при градиентном накоплении
            loss = loss / accumulation_steps
        
        # Обратный проход
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Обновление параметров каждые accumulation_steps
        if (iteration + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Обработка loss
        if use_amp:
            current_loss = loss.item() * accumulation_steps  # Восстановление масштабированного loss
        else:
            current_loss = loss.item()
        
        # Сглаживание loss
        if iteration == 0:
            avg_loss = current_loss
        else:
            avg_loss = smooth_f * current_loss + (1 - smooth_f) * avg_loss
        
        losses.append(avg_loss)
        
        # Проверка на расхождение
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if avg_loss > diverge_th * best_loss:
            if verbose:
                print(f"Loss diverged at iteration {iteration}, stopping...")
            break
        
        # Вывод прогресса
        if verbose and iteration % max(1, num_iter // 10) == 0:
            print(f"Iteration {iteration}/{num_iter}, LR: {current_lr:.2e}, Loss: {avg_loss:.4f}")
        
        iteration += 1
    
    # Определение оптимального LR (точка с минимальным сглаженным loss перед резким ростом)
    losses = np.array(losses)
    lrs = np.array(lrs)
    
    # Поиск точки с минимальным loss до начала резкого роста
    # Ищем точку с минимальным loss в первой половине кривой (до значительного роста)
    min_loss_idx = np.argmin(losses[:len(losses)//2])
    min_loss_val = losses[min_loss_idx]
    
    # Альтернативный метод: найти точку перед резким ростом (где производная максимальна)
    gradients = np.gradient(losses)
    # Ищем точку с минимальным loss перед резким увеличением градиента
    grad_threshold = np.percentile(gradients, 80)  # Берем 80-й перцентиль как порог резкого роста
    steep_idx = np.where(gradients > grad_threshold)[0]
    if len(steep_idx) > 0:
        # Используем точку перед резким ростом
        cutoff_idx = max(0, steep_idx[0] - 10)  # Отступаем на 10 шагов
        if cutoff_idx > min_loss_idx:  # Используем минимальный loss в допустимом диапазоне
            min_loss_idx = np.argmin(losses[:cutoff_idx+1])
    
    best_lr = lrs[min_loss_idx]
    
    # Восстановление исходного состояния
    optimizer.load_state_dict(original_state)
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr
    model.train()  # Возврат в режим обучения
    
    # Очистка памяти
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Построение графика
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses, label='Smoothed Loss')
        plt.axvline(x=best_lr, color='red', linestyle='--', 
                   label=f'Best LR: {best_lr:.2e}')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.title('Learning Rate Finder')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    
    return lrs, losses, best_lr


def warmup_finder(model, train_loader, optimizer, criterion, 
                  base_lr, warmup_epochs=5, num_batches=None,
                  warmup_method='linear', device='cuda',
                  accumulation_steps=1, use_amp=False, 
                  verbose=True, plot=True):
    """
    Поиск оптимальной стратегии warmup для обучения модели.

    Parameters:
    - model: torch.nn.Module
    - train_loader: torch.utils.data.DataLoader
    - optimizer: torch.optim.Optimizer
    - criterion: loss function
    - base_lr: базовая скорость обучения (после warmup)
    - warmup_epochs: количество эпох warmup
    - num_batches: количество батчей для анализа (None для всех в warmup эпохах)
    - warmup_method: 'linear', 'exp', 'cosine' - метод увеличения LR
    - device: устройство ('cuda' или 'cpu')
    - accumulation_steps: количество шагов накопления градиентов
    - use_amp: использовать автоматическое масштабирование точности (AMP)
    - verbose: выводить прогресс
    - plot: строить график

    Returns:
    - epochs: список эпох
    - losses: список значений loss
    - lrs: список скоростей обучения
    - optimal_warmup_epochs: рекомендуемое количество эпох warmup

    Улучшения для функции warmup_finder:
    
    1. **Оптимизации памяти:**
       - Градиентное накопление (--accumulation_steps)
       - AMP для уменьшения использования памяти
       - Очистка кэша CUDA после завершения
    
    2. **Гибкие методы warmup:**
       - Линейное, экспоненциальное, косинусное увеличение LR
       - Возможность анализа на подмножестве данных
    
    3. **Быстрый анализ:**
       - Фокус на первые эпохи обучения
       - Определение точки стабилизации loss
       - Минимальные вычисления
    
    4. **Улучшенная визуализация:**
       - Отображение фазы warmup
       - Настройка стиля сетки
       - Логарифмическая шкала для LR
    """
    
    # Проверяем доступность AMP для соответствующего устройства
    if use_amp and device != 'cuda':
        print(f"AMP доступен только для CUDA, но указано устройство: {device}. Устанавливаем use_amp=False")
        use_amp = False
    
    if use_amp:
        from torch.amp import GradScaler, autocast
    else:
        from contextlib import nullcontext
        autocast = lambda: nullcontext()
    
    import gc
    
    # Сохраняем исходное состояние оптимизатора
    original_state = optimizer.state_dict()
    original_lr = optimizer.param_groups[0]['lr']
    
    # Инициализация AMP scaler для CUDA
    scaler = GradScaler(device=device) if use_amp else None
    
    # Подготовка модели
    model.train()
    model.to(device)
    
    # Списки для хранения результатов
    epochs = []
    losses = []
    lrs = []
    batch_count = 0
    max_batches = num_batches if num_batches else float('inf')
    
    # Вычисление общего количества батчей для warmup
    total_batches = len(train_loader) * warmup_epochs
    if num_batches:
        total_batches = min(total_batches, num_batches)
    
    # Основной цикл
    for epoch in range(warmup_epochs):
        if batch_count >= max_batches:
            break
            
        for inputs, targets in train_loader:
            if batch_count >= max_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Вычисление текущего LR в зависимости от метода
            progress = batch_count / total_batches
            if warmup_method == 'linear':
                current_lr = base_lr * progress
            elif warmup_method == 'exp':
                current_lr = base_lr * (progress ** 2) if progress > 0 else 1e-8
            elif warmup_method == 'cosine':
                current_lr = base_lr * (1 - np.cos(progress * np.pi)) / 2
            else:
                raise ValueError("warmup_method должен быть 'linear', 'exp' или 'cosine'")
            
            # Убедимся, что LR не слишком мал
            current_lr = max(current_lr, 1e-8)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Прямой проход с AMP
            if use_amp:
                with autocast(device_type=device):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Нормализация loss при градиентном накоплении
                    loss = loss / accumulation_steps
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Нормализация loss при градиентном накоплении
                loss = loss / accumulation_steps
            
            # Обратный проход
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Обновление параметров каждые accumulation_steps
            if (batch_count + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            # Обработка loss
            if use_amp:
                current_loss = loss.item() * accumulation_steps
            else:
                current_loss = loss.item()
            
            epochs.append(epoch + batch_count / len(train_loader))
            losses.append(current_loss)
            lrs.append(current_lr)
            
            # Вывод прогресса
            if verbose and batch_count % max(1, total_batches // 10) == 0:
                print(f"Batch {batch_count}/{total_batches}, LR: {current_lr:.2e}, Loss: {current_loss:.4f}")
            
            batch_count += 1
    
    # Определение оптимального количества эпох warmup
    # Ищем точку, после которой loss стабилизируется
    losses_array = np.array(losses)
    
    # Простой метод: найти точку с минимальным loss
    min_loss_idx = np.argmin(losses_array)
    optimal_epoch = epochs[min_loss_idx]
    
    # Более умный метод: найти точку стабилизации
    window_size = max(1, len(losses_array) // 10)
    if len(losses_array) > window_size:
        # Вычисляем скользящее среднее
        from scipy.ndimage import uniform_filter1d
        smoothed_losses = uniform_filter1d(losses_array, size=window_size, mode='nearest')
        
        # Находим точку с минимальным smoothed loss
        min_smooth_idx = np.argmin(smoothed_losses)
        optimal_epoch = epochs[min_smooth_idx]
    else:
        optimal_epoch = optimal_epoch
    
    # Преобразуем в количество эпох (округляем вверх)
    optimal_warmup_epochs = int(np.ceil(optimal_epoch))
    optimal_warmup_epochs = max(1, min(optimal_warmup_epochs, warmup_epochs))
    
    # Восстановление исходного состояния
    optimizer.load_state_dict(original_state)
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr
    model.train()
    
    # Очистка памяти
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Построение графика
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # График loss
        ax1.plot(epochs, losses, label='Training Loss', color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Warmup Analysis - Loss')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Добавляем вертикальную линию в оптимальную точку
        ax1.axvline(x=optimal_epoch, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_epoch:.1f} epochs')
        ax1.legend()
        
        # График LR
        ax2.plot(epochs, lrs, label='Learning Rate', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.set_title('Warmup Schedule')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.axvline(x=optimal_epoch, color='red', linestyle='--', 
                   label=f'Optimal: {optimal_epoch:.1f} epochs')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return epochs, losses, lrs, optimal_warmup_epochs


# Пример вызова функции
"""
# Пример использования warmup_finder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Создание модели и данных (пример)
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Синтетические данные
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Запуск поиска оптимального warmup
base_lr = 1e-3  # Базовая скорость обучения
epochs, losses, lrs, optimal_warmup = warmup_finder(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    base_lr=base_lr,
    warmup_epochs=5,
    num_batches=200,  # Анализировать только первые 200 батчей
    warmup_method='linear',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    accumulation_steps=1,
    use_amp=torch.cuda.is_available(),
    verbose=True,
    plot=True
)

print(f"Рекомендуемое количество эпох warmup: {optimal_warmup}")
"""


def weight_decay_finder(model, train_loader, optimizer, criterion, 
                       base_lr, weight_decays=None, num_epochs=3, 
                       device='cuda', accumulation_steps=1, use_amp=False, 
                       verbose=True, plot=True):
    """
    Поиск оптимального параметра weight decay для модели.

    Parameters:
    - model: torch.nn.Module
    - train_loader: torch.utils.data.DataLoader
    - optimizer: torch.optim.Optimizer (должен поддерживать weight_decay)
    - criterion: loss function
    - base_lr: базовая скорость обучения
    - weight_decays: список значений weight decay для тестирования (если None, используется стандартный диапазон)
    - num_epochs: количество эпох для тестирования каждого weight decay
    - device: устройство ('cuda' или 'cpu')
    - accumulation_steps: количество шагов накопления градиентов
    - use_amp: использовать автоматическое масштабирование точности (AMP)
    - verbose: выводить прогресс
    - plot: строить график

    Returns:
    - results: словарь с результатами {weight_decay: [losses]}
    - optimal_wd: оптимальное значение weight decay
    - final_losses: финальные значения loss для каждого weight decay

    Улучшения для функции weight_decay_finder:
    
    1. **Оптимизации памяти:**
       - Градиентное накопление (--accumulation_steps)
       - AMP для уменьшения использования памяти
       - Очистка кэша CUDA после завершения
    
    2. **Гибкие настройки:**
       - Возможность тестирования различных значений weight decay
       - Настраиваемое количество эпох
       - Поддержка различных оптимизаторов
    
    3. **Быстрый анализ:**
       - Сравнение нескольких значений weight decay за короткое время
       - Определение оптимального баланса между регуляризацией и производительностью
    
    4. **Улучшенная визуализация:**
       - Сравнение кривых обучения для разных weight decay
       - Отображение оптимального значения
       - Настройка стиля сетки
    """
    
    # Проверяем доступность AMP для соответствующего устройства
    if use_amp and device != 'cuda':
        print(f"AMP доступен только для CUDA, но указано устройство: {device}. Устанавливаем use_amp=False")
        use_amp = False
    
    if use_amp:
        from torch.amp import GradScaler, autocast
    else:
        from contextlib import nullcontext
        autocast = lambda: nullcontext()
    
    import gc
    
    # Стандартный диапазон weight decay, если не указан
    if weight_decays is None:
        weight_decays = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    # Сохраняем исходное состояние оптимизатора
    original_state = optimizer.state_dict()
    original_wd = optimizer.param_groups[0].get('weight_decay', 0)
    
    # Подготовка модели
    model.train()
    model.to(device)
    
    # Результаты для каждого weight decay
    results = {}
    final_losses = {}
    
    # Инициализация AMP scaler для CUDA
    scaler = GradScaler(device=device) if use_amp else None
    
    for wd in weight_decays:
        if verbose:
            print(f"\nТестирование weight_decay: {wd}")
        
        # Обновляем weight decay в оптимизаторе
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = wd
        
        # Списки для хранения результатов текущего weight decay
        epoch_losses = []
        
        # Тренировка на заданное количество эпох
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Устанавливаем base_lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr
                
                # Прямой проход с AMP
                if use_amp:
                    with autocast(device_type=device):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        # Нормализация loss при градиентном накоплении
                        loss = loss / accumulation_steps
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Нормализация loss при градиентном накоплении
                    loss = loss / accumulation_steps
                
                # Обратный проход
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Обновление параметров каждые accumulation_steps
                if (num_batches + 1) % accumulation_steps == 0:
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                
                # Обработка loss
                if use_amp:
                    current_loss = loss.item() * accumulation_steps
                else:
                    current_loss = loss.item()
                
                epoch_loss += current_loss
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_losses.append(avg_epoch_loss)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        
        results[wd] = epoch_losses
        final_losses[wd] = epoch_losses[-1]  # Последнее значение loss
    
    # Определение оптимального weight decay
    optimal_wd = min(final_losses, key=final_losses.get)
    
    # Восстановление исходного состояния
    optimizer.load_state_dict(original_state)
    for param_group in optimizer.param_groups:
        param_group['weight_decay'] = original_wd
    model.train()
    
    # Очистка памяти
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    
    # Построение графика
    if plot:
        plt.figure(figsize=(12, 8))
        
        for wd, losses in results.items():
            epochs_x = list(range(1, len(losses)+1))
            plt.plot(epochs_x, losses, label=f'WD: {wd:.0e}', marker='o')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Weight Decay Finder - Training Loss Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Добавляем вертикальную линию с оптимальным weight decay
        plt.axhline(y=final_losses[optimal_wd], color='red', linestyle='--', 
                   label=f'Optimal WD: {optimal_wd:.2e}')
        
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print(f"\nОптимальный weight decay: {optimal_wd}")
        print(f"Финальный loss при оптимальном WD: {final_losses[optimal_wd]:.4f}")
    
    return results, optimal_wd, final_losses


# Пример вызова функции
"""
# Пример использования weight_decay_finder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Создание модели и данных (пример)
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Синтетические данные
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)  # Изначально без weight decay
criterion = nn.MSELoss()

# Запуск поиска оптимального weight decay
results, optimal_wd, final_losses = weight_decay_finder(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    base_lr=1e-3,
    weight_decays=[0, 1e-5, 1e-4, 1e-3, 1e-2],  # Специфический диапазон
    num_epochs=3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    accumulation_steps=1,
    use_amp=torch.cuda.is_available(),
    verbose=True,
    plot=True
)

print(f"Оптимальный weight decay: {optimal_wd}")
print("Финальные значения loss для каждого weight decay:")
for wd, loss in final_losses.items():
    print(f"  WD={wd:.0e}: {loss:.4f}")
"""
    
 
