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
    
def batch_size_finder(model, train_loader, optimizer_class, criterion, initial_batch_size=32,
                      max_batch_size=512, growth_factor=1.5, num_steps_per_batch=10,
                      device='cuda', use_amp=False, verbose=True, plot=True, 
                      target_metric='throughput'):
    """
    Определение оптимального размера батча для больших моделей.

    Parameters:
    - model: torch.nn.Module
    - train_loader: torch.utils.data.DataLoader (с малым batch_size)
    - optimizer_class: класс оптимизатора (например, torch.optim.Adam)
    - criterion: loss function
    - initial_batch_size: начальный размер батча для тестирования
    - max_batch_size: максимальный размер батча для тестирования
    - growth_factor: множитель увеличения batch_size
    - num_steps_per_batch: количество шагов для оценки каждого batch_size
    - device: устройство ('cuda' или 'cpu')
    - use_amp: использовать автоматическое масштабирование точности (AMP)
    - verbose: выводить прогресс
    - plot: строить график
    - target_metric: целевая метрика ('throughput' или 'stability')

    Returns:
    - batch_sizes: список протестированных размеров батча
    - metrics: список значений метрики (throughput или loss stability)
    - optimal_batch_size: оптимальный размер батча

        
    Улучшения для функции batch_size_finder:
    
    1. **Проверка на OOM:** Предварительная проверка возможности использования batch_size
    2. **Многокритериальная оптимизация:** Учет throughput, стабильности и потребления памяти
    3. **Поддержка AMP:** Учет автоматического масштабирования точности
    4. **Гибкая целевая метрика:** Возможность выбора между производительностью и стабильностью
    5. **Комплексная визуализация:** Графики для всех ключевых метрик
    6. **Оптимизация памяти:** Очистка временных моделей и данных после каждой итерации
    
    Дополнительные рекомендации:
    - Для очень больших моделей можно использовать gradient accumulation вместо большого batch_size
    - Рассмотреть использование progressive resizing (постепенное увеличение размера изображений)
    - При использовании large batch_size может потребоваться корректировка learning rate (Linear Scaling Rule)
    - Для распределенного обучения оптимальный глобальный batch_size может отличаться

    """
    
    # Проверка на CUDA OOM ошибки
    def test_batch_size(batch_size, num_steps=5):
        """Тестирование возможности использования заданного batch_size."""
        try:
            # Создание временной модели и оптимизатора
            temp_model = type(model)(**model.__init__.__code__.co_consts[0]) \
                        if hasattr(model, '__init__') else model
            temp_model = nn.Sequential(*list(model.children())[:2]).to(device) if len(list(model.children())) > 1 else model.to(device)
            temp_model.eval()
            
            # Создание фиктивных данных
            sample_input = torch.randn(batch_size, *next(iter(train_loader))[0].shape[1:]).to(device)
            sample_target = torch.randint(0, model.fc.out_features if hasattr(model, 'fc') else 10, 
                                        (batch_size,)).to(device)
            
            # Тестовый проход
            with autocast() if use_amp else nullcontext():
                with torch.no_grad():
                    output = temp_model(sample_input)
                    loss = criterion(output, sample_target)
            
            del sample_input, sample_target, output, loss
            if hasattr(temp_model, 'module'):
                del temp_model.module
            del temp_model
            torch.cuda.empty_cache()
            return True
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                return False
            else:
                raise e
    
    # Оценка производительности для заданного batch_size
    def evaluate_batch_size(batch_size, num_steps=num_steps_per_batch):
        """Оценка метрик для заданного размера батча."""
        # Создание модели и оптимизатора
        temp_model = type(model)(**{k: v for k, v in model.__dict__.items() 
                                  if not k.startswith('_')}).to(device) \
                    if hasattr(model, '__dict__') else model.to(device)
        temp_optimizer = optimizer_class(temp_model.parameters(), lr=1e-3)
        temp_model.train()
        
        scaler = GradScaler() if use_amp else None
        
        times = []
        losses = []
        memory_usages = []
        
        # Итератор данных
        data_iter = iter(train_loader)
        
        for step in range(num_steps):
            try:
                # Сборка батча нужного размера
                inputs_batch = []
                targets_batch = []
                
                count = 0
                while count < batch_size:
                    try:
                        inputs, targets = next(data_iter)
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Добавляем части в батч
                        needed = batch_size - count
                        if len(inputs) >= needed:
                            inputs_batch.append(inputs[:needed])
                            targets_batch.append(targets[:needed])
                            count += needed
                        else:
                            inputs_batch.append(inputs)
                            targets_batch.append(targets)
                            count += len(inputs)
                    except StopIteration:
                        data_iter = iter(train_loader)
                        continue
                
                # Конкатенация в один батч
                inputs = torch.cat(inputs_batch, dim=0)[:batch_size]
                targets = torch.cat(targets_batch, dim=0)[:batch_size]
                
                if len(inputs) < batch_size:
                    # Дополнение до нужного размера (если нужно)
                    pad_size = batch_size - len(inputs)
                    inputs = torch.cat([inputs, inputs[:pad_size]], dim=0)
                    targets = torch.cat([targets, targets[:pad_size]], dim=0)
                
            except Exception:
                # Если данные закончились, используем последний батч
                continue
            
            # Измерение времени
            start_time = time.time()
            
            # Прямой проход
            context = autocast() if use_amp else nullcontext()
            with context:
                outputs = temp_model(inputs)
                loss = criterion(outputs, targets)
            
            # Обратный проход
            temp_optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(temp_optimizer)
                scaler.update()
            else:
                loss.backward()
                temp_optimizer.step()
            
            # Измерение времени и памяти
            end_time = time.time()
            times.append(end_time - start_time)
            losses.append(loss.item())
            
            if device == 'cuda':
                memory_usages.append(torch.cuda.memory_allocated(device))
            
        # Очистка
        del temp_model, temp_optimizer
        if scaler:
            del scaler
        torch.cuda.empty_cache()
        
        # Вычисление метрик
        avg_time = np.mean(times)
        throughput = batch_size / avg_time if avg_time > 0 else 0
        loss_std = np.std(losses)
        avg_memory = np.mean(memory_usages) if memory_usages else 0
        
        return {
            'throughput': throughput,
            'loss_std': loss_std,
            'time_per_batch': avg_time,
            'avg_loss': np.mean(losses),
            'memory_usage': avg_memory
        }
    
    # Генерация последовательности размеров батча
    batch_sizes = []
    current_batch_size = initial_batch_size
    
    while current_batch_size <= max_batch_size:
        if test_batch_size(int(current_batch_size)):
            batch_sizes.append(int(current_batch_size))
        current_batch_size *= growth_factor
    
    if verbose:
        print(f"Testing batch sizes: {batch_sizes}")
    
    # Оценка метрик для каждого размера батча
    results = []
    metrics_values = {'throughput': [], 'stability': [], 'memory': []}
    
    for i, batch_size in enumerate(batch_sizes):
        if verbose:
            print(f"Testing batch_size: {batch_size}")
        
        try:
            metrics = evaluate_batch_size(batch_size)
            results.append({
                'batch_size': batch_size,
                'metrics': metrics
            })
            
            metrics_values['throughput'].append(metrics['throughput'])
            metrics_values['stability'].append(1.0 / (metrics['loss_std'] + 1e-8))  # Обратное значение для стабильности
            metrics_values['memory'].append(metrics['memory_usage'] / 1024**3)  # в GB
            
            if verbose:
                print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
                print(f"  Loss std: {metrics['loss_std']:.4f}")
                print(f"  Memory usage: {metrics['memory_usage'] / 1024**3:.2f} GB")
                
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if verbose:
                    print(f"  Batch size {batch_size} caused OOM, stopping...")
                break
            else:
                raise e
    
    if not results:
        raise RuntimeError("No batch size could be tested successfully - all caused OOM")
    
    # Определение оптимального размера батча
    batch_sizes_tested = [r['batch_size'] for r in results]
    throughputs = [r['metrics']['throughput'] for r in results]
    stabilities = [1.0 / (r['metrics']['loss_std'] + 1e-8) for r in results]
    
    if target_metric == 'throughput':
        # Максимизация throughput с учетом ограничений на стабильность
        # Ищем баланс между высокой производительностью и приемлемой стабильностью
        normalized_throughput = (np.array(throughputs) - np.min(throughputs)) / (np.max(throughputs) - np.min(throughputs) + 1e-8)
        normalized_stability = (np.array(stabilities) - np.min(stabilities)) / (np.max(stabilities) - np.min(stabilities) + 1e-8)
        
        # Взвешенная комбинация: 70% throughput, 30% stability
        combined_score = 0.7 * normalized_throughput + 0.3 * normalized_stability
        optimal_idx = np.argmax(combined_score)
        
    elif target_metric == 'stability':
        # Максимизация стабильности с минимальным падением throughput
        optimal_idx = np.argmax(stabilities)
        # Но проверяем, что throughput не слишком низкий
        min_acceptable_throughput = 0.5 * max(throughputs)  # Не ниже 50% максимума
        valid_indices = [i for i in range(len(throughputs)) if throughputs[i] >= min_acceptable_throughput]
        if valid_indices:
            valid_stabilities = [stabilities[i] for i in valid_indices]
            optimal_idx = valid_indices[np.argmax(valid_stabilities)]
    
    optimal_batch_size = batch_sizes_tested[optimal_idx]
    
    # Построение графиков
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Throughput vs Batch Size
        axes[0, 0].plot(batch_sizes_tested, throughputs, 'b-o')
        axes[0, 0].axvline(x=optimal_batch_size, color='red', linestyle='--', 
                          label=f'Optimal: {optimal_batch_size}')
        axes[0, 0].set_xlabel('Batch Size')
        axes[0, 0].set_ylabel('Throughput (samples/sec)')
        axes[0, 0].set_title('Throughput vs Batch Size')
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)
        axes[0, 0].legend()
        
        # Loss Stability vs Batch Size
        axes[0, 1].plot(batch_sizes_tested, [1/x if x > 0 else float('inf') for x in [r['metrics']['loss_std'] for r in results]], 'g-o')
        axes[0, 1].axvline(x=optimal_batch_size, color='red', linestyle='--',
                          label=f'Optimal: {optimal_batch_size}')
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Stability (1/loss_std)')
        axes[0, 1].set_title('Loss Stability vs Batch Size')
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)
        axes[0, 1].legend()
        
        # Memory Usage vs Batch Size
        memory_gb = [r['metrics']['memory_usage'] / 1024**3 for r in results]
        axes[1, 0].plot(batch_sizes_tested, memory_gb, 'r-o')
        axes[1, 0].axvline(x=optimal_batch_size, color='red', linestyle='--',
                          label=f'Optimal: {optimal_batch_size}')
        axes[1, 0].set_xlabel('Batch Size')
        axes[1, 0].set_ylabel('GPU Memory (GB)')
        axes[1, 0].set_title('Memory Usage vs Batch Size')
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
        axes[1, 0].legend()
        
        # Combined Score
        axes[1, 1].plot(batch_sizes_tested, combined_score if target_metric == 'throughput' else normalized_stability, 'm-o')
        axes[1, 1].axvline(x=optimal_batch_size, color='red', linestyle='--',
                          label=f'Optimal: {optimal_batch_size}')
        axes[1, 1].set_xlabel('Batch Size')
        axes[1, 1].set_ylabel('Combined Score')
        axes[1, 1].set_title(f'Optimization Target: {target_metric}')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.suptitle('Batch Size Finder Results', fontsize=16, y=1.02)
        plt.show()
    
    if verbose:
        print(f"\nOptimal batch size: {optimal_batch_size}")
        print(f"Target metric: {target_metric}")
        print(f"Final metrics for optimal batch size:")
        optimal_metrics = results[optimal_idx]['metrics']
        print(f"  Throughput: {optimal_metrics['throughput']:.2f} samples/sec")
        print(f"  Loss std: {optimal_metrics['loss_std']:.4f}")
        print(f"  Memory usage: {optimal_metrics['memory_usage'] / 1024**3:.2f} GB")
    
    return batch_sizes_tested, throughputs, optimal_batch_size

 

