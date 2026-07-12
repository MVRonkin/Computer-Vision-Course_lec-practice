Материалы курса представлены в репозитории https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/tree/main/2025 

#  Содержание курса Mind Map

<img src="https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/%D0%B1%D1%83%D0%BC%D0%B0%D0%B3%D0%B8/Course%20structure.png?raw=true" alt="Description" style="width:700px;height:800px;">


#  Основные материалы курса

| код | Раздел  | Содержание      | ссылка на лекцию     | ссылка на практику    | Контрольное мероприятие    |
| --- | ------- | --------------- | -------------------- | --------------------- | -------------------------- |
| 1.  | Введение в предмет компьютерное зрение   | <details>Задачи компьютерного зрения, понятие и особенности цифровых изображений, понятие признаки в компьютерном зрении, понятие свертка в цифровой обработке изображений.<br>Обзор алгоритмов предобработки изображений:   построение фильтров размытия, резкости и  выделения границ,  бинаризация изображений, нелинейные фильтры<br>Построение последовательностей предобработки изображений в OpenCV<br>Сравнение библиотек работы с изображениями<br></details>  | [GitHub 1](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/01%20%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D0%BE%D0%B5%20%D0%B7%D1%80%D0%B5%D0%BD%D0%B8%D0%B5.pptx), [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/02%20%D0%9F%D0%BE%D0%BD%D1%8F%D1%82%D0%B8%D0%B5%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F.pptx) | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS1_Classic_into_PyTorch.ipynb) [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS2_Augment_CNNLayer.ipynb) |  -|
| 2.  | Обзор классических подходов к решению задач компьютерного зрения  | <details>Классические подходы к решению задач компьютерного зрения: поиск изображений по признакам SIFT, и по эталону, методы обнаружения объектов (Виолы-Джонса), Сегментация Watershed, Понятие Оптический поток<br>Решение задачи компьютерного зрения классическими методами<br></details>     | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/03%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%B7%D1%80%D0%B5%D0%BD%D0%B8%D1%8F.pptx)     | [GitHub](https://github.com/MVRonkin/Classical-computer-vision)      |  -   |
| 3.  |Современное состояние сверточных нейронных сетей в задачах компьютерного зрения. | <details>Внутренняя структура и особенности обучения сверточной нейронной сети ResNet<br>Сверточные сети семейства ResNet: ResNeXt, Xception, BIT, DenseNet, ResNet-D, NFNet, Step By Step network<br>Построение и дообучение сверточных нейронных сетей в PyTorch + torchvision с применением backbone сетей семейства ResNet: training pipeline, transfer learning, функции потерь и метрики обучения, анализ результатов обучения.<br>Решение задачи компьютерного зрения с использованием дообученных моделей<br></details>     | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/04%20%D0%9F%D1%80%D0%B8%D0%BD%D1%86%D0%B8%D0%BF%D1%8B%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D1%85%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85%20%D1%81%D0%B5%D1%82%D0%B5%D0%B9.pptx)   | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS5_PyTorch_transfer.ipynb)                                                                                                                              | [Invite](https://www.kaggle.com/t/bdd0a0c4cd4f489bb3d2db81a36e9d33) |
| 4.  | Оптимизация сверточных сетей   | <details>Другие ахритектуры СНС и Архитектуры для низкопроизводительных устройств: MoblieNet block, SE-блок, NAS: EfficientNet, RepVGG/MobileOne, <br>Приемы разработки и оптимизации специализированных архитектур для работы с изображениями:  разработка решений на основе сложных конфигураций сверточных нейронных сетей в timm, приемы обучения СНС.<br>Решение задачи компьютерного зрения с использованием кастомизации архитектуры под задачу<br> </details>    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/05%20%D0%A1%D0%BE%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B5%20%D1%81%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5%20%D0%A1%D0%9D%D0%A1.pptx)   | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS6_Tricks_In_Train.ipynb)  | [Invite](https://www.kaggle.com/t/37d51df9321249ac90299ab055237838) |
| 5.  | Архитектуры трансфромеры для построения продуктов с компьютерным зрением   | <details>Архитектура ViT и ее модификации: SwiGLU, MOE, RMSNorm,  DeiT и другие<br>Другие архитектуры типа трансфромеры: SWIN, MobileViT, архитекутры-миксеры, ConvNeXt<br>Исследование и имплементация SOTA-моделей в задачах анализа изображений на примере семейства моделей ViT в экосистеме HuggingFace<br>Построение пайплайна. Комбинирует несколько моделей в комплексные решения<br> </details>    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/06%20%D0%A1%D0%B5%D1%82%D0%B8%20%D0%BD%D0%B0%20%D0%BE%D1%81%D0%BD%D0%BE%D0%B2%D0%B5%20%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80%D0%BE%D0%B2.pptx)    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS9%20-%20Hugging%20Face.ipynb)    |    -    |
| 6.  | Задачи сегментации изображений      | <details> Переход от задач классификации к задачам Dense Prediction на примере сегментации изображений.<br>Архитекутры Энкодер-Декодер на примере U-Net,<br>Архитектуры головной части на примере DeepLab V3/расширенная свертка и нелинейное рецептивное поле.<br>Архитектуры трансформеры для сегменатции изображений: SegFormer  и другие архитектуры.<br>Применение алгоритмов и библиотек компьютерного зрения для прикладных задач сегментации изображений: SMP, albumentation,<br>Решение задачи семантической сегментации с использованием дообученных моделей<br></details> | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/07%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D1%81%D0%B5%D0%BC%D0%B0%D0%BD%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9%20%D1%81%D0%B5%D0%B3%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D0%B8.pptx)    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS7_Semantic%20Segmentation.ipynb)    | [Invite](https://www.kaggle.com/t/4f7b3ae36cc14385bb68b20d7eaef4dd) |
| 7.  | Задачи обнаружения объектов    | <details>Многоэтапные архитектуры обнаружения объектов на примере Faster-R-CNN/Mask-R-CNN.<br>Быстрые одноэтапные архитектуры на примере семейства архитектур YOLO<br>Архитектуры трансформеры для обнаружения объектов на примере сеймества DETER + RT-DETER<br>Применение алгоритмов и библиотек компьютерного зрения для прикладных задач обнаружения объектов на изображениях на примере YOLO<br>Решение задачи семантической обнаружение объектов с использованием дообученных моделей<br></details>            | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/08%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D0%BE%D0%B1%D0%BD%D0%B0%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%BE%D0%B2.pptx)  | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS8_YOLOv8%20API.ipynb)  | [Invite](https://www.kaggle.com/t/5026ff51255641c68c92d4d04a285c3d) |
| 8.  | Подходы к построению базисных моделей компьютерного зрения   | <details>Принципы самообучения (MAE, I-JEPA), Контрастное обучение (DINO), мультимодальное обучение (CLIP)<br>Базисные архитектуры решения отдельных задач (SAM, Ground DINO)<br>Обзор принципов построения VLM<br>Разработка стратегий применения базисных моделей CV (компьютерного зрения): zero-shot learning, linerar/non-linear probe и других,<br>Решение задач распознавания текста на изображении<br> </details>    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/09%20%D0%91%D0%B0%D0%B7%D0%B8%D1%81%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.pptx)   |   -  |  [Invite](https://www.kaggle.com/t/4f7b3ae36cc14385bb68b20d7eaef4dd)  |
| 9.  | Особенности построения систем с компьютерным зрением  | <details>Обзор вопросов подготовки данных <br> Вопросы интерпретируемости результатов работы моделей Grad-Cam <br> Вопросы оценки надежности решения задач компьютерного зрения TTA, <br> специальные методы повышения качества, OOD тестирование <br>Решение задач обнаружения аномалий/дрейфа данных <br></details>     | -  |   -  |  -   |
| 10. | Предпосылки соврменного компьютерного зрения: Переход от LeNet к ResNet      | <details>Повтор понятий сверточный блок, рецептивное поле, регуляризация, нормализация, обучение СНС и других понятий),<br>Построение кастомных сверточных нейронных сетей в PyTorch + torchvision: training pipeline, аугментация, работа с набором данных..<br>Решение задачи классификации методами компьютерного зрения<br></details>    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/04%20%D0%9F%D1%80%D0%B8%D0%BD%D1%86%D0%B8%D0%BF%D1%8B%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D1%85%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85%20%D1%81%D0%B5%D1%82%D0%B5%D0%B9.pptx)  | [GitHub 1](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS3_LeNet.ipynb), [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS4_Make_CNN_Better.ipynb)      | [Invite](https://www.kaggle.com/t/bdd0a0c4cd4f489bb3d2db81a36e9d33) |
<!-- | 8.  | Решение прикладных задач компьютерного зрения для работы с видеопотоком   | Обзор особенностей видеоданных в задачах компьютерного зрения. Примеры задач для видеоданных.<br>Использование CNN и архитекутр трансформеров в обработке видео данных<br>Построение пайплайн обработки видеопотока при помощи нейронных сетей.<br>Решение задач object tracking<br>   |   - | - |  -  | -->

## Курс в логике КРМ


### Cхема и место курса

<code>Пререквизиты: MF (математика) →  ML (классическое МО) → DL-1 (основы DL)
                                                        ↓
                                         DL-3 (Компьютерное зрение) 
                                                        ↓
Постреквизиты: DL-2/4/5 → LLM → MLOps → FC (опережающие)
                    ↓
                Отраслевые применения (ОПД)
</code>

---

### Таблица соответствия модулей курса и индикаторов DL-3


| № | Модуль курса | Индикатор DL-3 | Уровень | Что конкретно покрывается из КРМ |
|---|--------------|----------------|---------|----------------------------------|
| 1. | Введение в CV. OpenCV, фильтры, бинаризация | **DL-3.1** | **Б** | Использует готовые модели из библиотеки OpenCV и популярных фреймворков (TensorFlow Hub, TorchVision). Умеет применять стандартные архитектуры (ResNet, YOLO) для базовых задач (классификация, детекция объектов). Запускает инференс на изображениях и простых видеопотоках. Использует базовые функции библиотек OpenCV и PIL для обработки. Умеет сохранять и загружать модели. |
| | | **DL-3.2** | **Б** | Понимает принципы представления изображений и кодирования цвета; применяет фильтрацию изображений, включая частотную; владеет аппаратом математической морфологии. Понимает базовые задачи анализа изображений — классификация, детекция, сегментация. Применяет известные архитектуры нейронных сетей (CNN) в решении простых задач распознавания изображений. |
| 2. | Классические подходы к решению задач CV (SIFT, Viola-Jones, Watershed, оптический поток) | **DL-3.2** | **С**| Разрабатывает алгоритмы сегментации изображений (разделение-слияние регионов, нормализованный разрез графа, сдвиг среднего значения), включая семантическую сегментацию; применяет преобразование Хафа и RANSAC; применяет алгоритмы детекции характеристических точек (детектор Харриса, детектор Фестнера, SUSAN, блобы, DoG); применяет дескрипторы изображений, например, SIFT. Строит нейросетевые архитектуры для анализа изображений (VGG, Inception, ResNet, EfficientNet и т.д.) с учетом особенностей обучения и дообучения. |
| 3. | Современные сверточные нейронные сети (ResNet, ResNeXt, DenseNet, transfer learning) | **DL-3.1** | **С** | Сравнивает разные предобученные модели под конкретную задачу. Проводит перенос обучения на своих данных. Оптимизирует гиперпараметры для улучшения качества. Создает сложные пайплайны аугментации (с использованием библиотеки Albumentations). Умеет работать с видео: извлечение кадров, обработка временных последовательностей путём применения CNN+RNN, 3D CNN. |
| | | **DL-3.2** | **С** | Строит нейросетевые архитектуры для анализа изображений (VGG, Inception, ResNet, EfficientNet и т.д.) с учетом особенностей обучения и дообучения. Строит архитектуры FCN и Unet в задачах сегментации, функции потерь для задачи сегментации. Строит одностадийные (SSD, YOLO) и двухстадийные (FASTER R-CNN, Mask R-CNN) детекторы в задачах детекции, функции потерь в задаче детекции. |
| 4. | Оптимизация сверточных сетей (MobileNet, SE-блок, NAS, EfficientNet, RepVGG) | **DL-3.2** | **С** | Строит решения на основе сложных конфигураций CNN (EfficientNet, RetinaNet). Разрабатывает backbone-сети. |
| | | **DL-3.3** | **Б** | Умеет применять стандартные архитектуры CNN (ResNet, EfficientNet, YOLO) для базовых задач компьютерного зрения (классификация, детекция, сегментация). Использует готовые реализации из библиотек (OpenCV, TensorFlow/PyTorch, Hugging Face). Проводит базовую аугментацию и нормализацию изображений. Умеет оценивать качество по стандартным метрикам (точность, mAP, IoU). |
| 5. | Архитектуры трансформеров для CV (ViT, SWIN, MobileViT, ConvNeXt, HuggingFace) | **DL-3.1** | **П** | Разрабатывает стратегии применения моделей компьютерного зрения под бизнес-задачи. Комбинирует несколько моделей в комплексные решения. Строит сквозные пайплайны обработки видеопотоков. |
| | | **DL-3.2** | **П** | Применяет марковские случайные поля и условные случайные поля. Исследует и имплементирует перспективные модели в задачах анализа изображений. Обладает глубоким пониманием подходов и методов дообучения ViT. Применяет метрическое обучение для задач поиска и распознавания на изображениях, выбирает и использует функции потерь (контрастивная, триплетная, на основе углового расстояния). Реализует распознавание текста на изображении, используя адекватные техники (CRNN, OCR на основе внимания и трансформерное OCR). |
| 6. | Задачи сегментации изображений (U-Net, DeepLab V3, SegFormer, SMP) | **DL-3.1** | **С** | Проводит перенос обучения на своих данных. Создает сложные пайплайны аугментации (Albumentations). |
| | | **DL-3.2** |**С** | Строит архитектуры FCN и Unet в задачах сегментации, функции потерь для задачи сегментации. |
| 7. | Задачи обнаружения объектов (Faster R-CNN, Mask R-CNN, YOLO, DETR) | **DL-3.1** | **С** | Сравнивает разные предобученные модели. Проводит перенос обучения. Оптимизирует гиперпараметры. Оценивает точность по mAP и IoU. |
| | | **DL-3.2** | **С** | Строит одностадийные (SSD, YOLO) и двухстадийные (Faster R-CNN, Mask R-CNN) детекторы в задачах детекции, функции потерь в задаче детекции. |
| 8. | Foundation models CV (MAE, DINO, CLIP, SAM, Grounding DINO, VLM, zero-shot) | **DL-3.1** | **П** | Разрабатывает стратегии применения моделей CV под бизнес-задачи. Комбинирует несколько моделей в комплексные решения. Строит сквозные пайплайны обработки видеопотоков. Применяет zero-shot learning. |
| | | **DL-3.2** | **П**| Применяет метрическое обучение для задач поиска и распознавания на изображениях, выбирает и использует функции потерь (контрастивная, триплетная, на основе углового расстояния). Реализует распознавание текста на изображении, используя адекватные техники (CRNN, OCR на основе внимания и трансформерное OCR). |
| 9. | Особенности построения систем CV (Grad-CAM, TTA, OOD, обнаружение аномалий) | **DL-3.3** | **С** | Кастомизирует архитектуры под задачу (изменение слоев, замена опорной сети). Применяет методы ускорения инференса (квантование, прунинг, TensorRT). Строит сложные стратегии аугментации (albumentations, кастомные трансформеры). Настраивает распределённое обучение (DDP, Horovod). Создает пайплайны CI/CD для моделей компьютерного зрения. |

---

## Итогового задания 
https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/README.md 

# Литература
## Курс сопровождается учбено-методическимими материалами 

* Ронкин М. В. Долганов А.Ю. Глубокое обучение систем компьютерного зрения : учебное пособие : https://elar.urfu.ru/handle/10995/143721</li>
* Ронкин М.В. Он-лайн курс «Компьютерное зрение» (24 лекции и 10 практик) https://courses.openedu.urfu.ru/course-v1:UrFU+COMPVISION+original. </li>


## доп.литература
* [Глубокое обучение Китов В.](https://deepmachinelearning.ru/docs/Neural-networks/book-title)
* [Учебник по машинному обучению. Разделы по DeepLearning](https://education.yandex.ru/handbook/)
* [Нейронные сети и их применение в научных исследованиях. Курс МГУ разделы про CV](https://msu.ai/programme) + [GitHub](https://github.com/EPC-MSU/EduNet-lectures)
* [d2l Dive into Deep Learning. 7,8,14 Chapters](https://d2l.ai/)
* [Deep_learning_school МФТИ](https://github.com/deouron/Deep_learning_school/tree/main)
* [deepschool blog CV](https://blog.deepschool.ru/category/cv/)
* [MIT Foundation of CV](https://visionbook.mit.edu/)
* [stanford cs231n 2025 CV](https://cs231n.stanford.edu/slides/2025/)
* [hyper.ai (paperwithcode)](https://hyper.ai/en/sota)
* [F.Chole Deep learning with python](https://deeplearningwithpython.io/chapters/) and [github](https://github.com/fchollet/deep-learning-with-python-notebooks)
* [Zero-to-Mastery learn pytorch](https://www.learnpytorch.io/)
* [HF compunity CV course](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)
* [autonomous-vision course](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous-vision/lectures/computer-vision/)

## еще более доп. лит.
* [курс по дифузионкам](https://arxiv.org/pdf/2406.08929)
* [pytorch-gradcam-book](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)
* [VLM 2025 blogpost HF](https://huggingface.co/blog/vlms-2025)
* [roboflow notebooks collection](https://github.com/roboflow/notebooks/tree/main/notebooks)
* [yandex blog VLM](https://habr.com/ru/companies/yandex/articles/904584/) и [тут](https://habr.com/ru/companies/yandex/articles/847706/) и [от МТС](https://habr.com/ru/companies/ru_mts/articles/944942/) и [тут](https://habr.com/ru/companies/yandex/articles/886466/)
* [Transformers Tutorials](https://github.com/NielsRogge/Transformers-Tutorials)
* [DLSchool](https://github.com/gracikk-ds/DeepLearningSchool)
* [CVRocket]https://github.com/gracikk-ds/cv-rocket)
* [CVEpam](https://github.com/gracikk-ds/cv-epam-course)
* [Vit project](https://github.com/sovit-123/vision_transformers)
* [Vae blog](https://rohitbandaru.github.io/blog/VAEs/)
* [NVidea DeepLearning Examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch)
* [DL for Vision 2020](https://sif-dlv.github.io/)
* [DL for CV book](https://download.e-bookshelf.de/download/0014/4806/95/L-G-0014480695-0052209288.pdf)
* [prgoram history cv](https://programminghistorian.org/en/lessons/computer-vision-deep-learning-pt1)
* [deep_learning_tutorial_iciap](https://www.ee.cuhk.edu.hk/~xgwang/deep_learning_tutorial_iciap.pdf)
* [mit cv dl](https://introtodeeplearning.com/2019/materials/2019_6S191_L3.pdf)
* [understanding dl book](https://udlbook.github.io/udlbook/)
* [uav dl course](https://uvadlc-notebooks.readthedocs.io/en/latest/index.html)
* [full stack dl 2022](https://fullstackdeeplearning.com/course/2022/)
* [Famous Deep Learning Papers](https://papers.baulab.info/)
* [Annotated Research Paper Implementations/Deep Learning Paper Implementations](https://nn.labml.ai/)
* [blog ssl](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
* [blog pytorch internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
* [tuning_playbook](https://github.com/google-research/tuning_playbook)
  
## всякие интересные репозитории
* [Deep-Learning-From-Scratch](https://github.com/lakshyaag/Deep-Learning-From-Scratch)
* [Deep Learning Fundamentals: Code Materials and Exercises](https://github.com/Lightning-AI/dl-fundamentals)
* [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
* [the incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch)

## инструменты
* [AlbumentationsX  - новая версия библиотеки Albumentation](https://github.com/albumentations-team/AlbumentationsX)
* [Vit=pytorch - библа о ViT](https://github.com/lucidrains/vit-pytorch)
* [lightly - библиотека для SSL](https://docs.lightly.ai/self-supervised-learning/index.html)
* [skorch - high api  в стиле sklearn для торч](https://github.com/skorch-dev/skorch)
* [fast.ai - very fast torch compatible framework](https://www.fast.ai/)
