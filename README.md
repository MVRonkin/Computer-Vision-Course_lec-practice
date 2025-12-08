Материалы курса представлены в репозитории https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/tree/main/2025 

Курс сопровождается учбено-методическимими материалами 
<ul> 
<li> 	Ронкин М. В. Долганов А.Ю. Глубокое обучение систем компьютерного зрения : учебное пособие : https://elar.urfu.ru/handle/10995/143721</li>
<li> 	Ронкин М.В. Он-лайн курс «Компьютерное зрение» (24 лекции и 10 практик) https://courses.openedu.urfu.ru/course-v1:UrFU+COMPVISION+original. </li>
</ul>

| код | Раздел  | Содержание      | ссылка на лекцию     | ссылка на практику    | Контрольное мероприятие    |
| --- | ------- | --------------- | -------------------- | --------------------- | -------------------------- |
| 1.  | Введение в предмет компьютерное зрение   | Задачи компьютерного зрения, понятие и особенности цифровых изображений, понятие признаки в компьютерном зрении, понятие свертка в цифровой обработке изображений.<br>Обзор алгоритмов предобработки изображений: построение фильтров размытия, резкости и  выделения границ,  бинаризация изображений, нелинейные фильтры<br>Построение последовательностей предобработки изображений в OpenCV<br>Сравнение библиотек работы с изображениями<br>  | [GitHub 1](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/01%20%D0%92%D0%B2%D0%B5%D0%B4%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%B2%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D0%BE%D0%B5%20%D0%B7%D1%80%D0%B5%D0%BD%D0%B8%D0%B5.pptx), [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/02%20%D0%9F%D0%BE%D0%BD%D1%8F%D1%82%D0%B8%D0%B5%20%D0%BF%D1%80%D0%B8%D0%B7%D0%BD%D0%B0%D0%BA%20%D0%B8%D0%B7%D0%BE%D0%B1%D1%80%D0%B0%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F.pptx) | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS1_Classic_into_PyTorch.ipynb) [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS2_Augment_CNNLayer.ipynb) |  -|
| 2.  | Современное состояние сверточных нейронных сетей в задачах компьютерного зрения. | Внутренняя структура и особенности обучения сверточной нейронной сети ResNet<br>Сверточные сети семейства ResNet: ResNeXt, Xception, BIT, DenseNet, ResNet-D, NFNet, Step By Step network<br>Построение и дообучение сверточных нейронных сетей в PyTorch + torchvision с применением backbone сетей семейства ResNet: training pipeline, transfer learning, функции потерь и метрики обучения, анализ результатов обучения.<br>Решение задачи компьютерного зрения с использованием дообученных моделей<br>     | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/04%20%D0%9F%D1%80%D0%B8%D0%BD%D1%86%D0%B8%D0%BF%D1%8B%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D1%85%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85%20%D1%81%D0%B5%D1%82%D0%B5%D0%B9.pptx)   | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS5_PyTorch_transfer.ipynb)                                                                                                                              | [Invite](https://www.kaggle.com/t/bdd0a0c4cd4f489bb3d2db81a36e9d33) |
| 3.  | Оптимизация сверточных сетей   | Другие ахритектуры СНС и Архитектуры для низкопроизводительных устройств: MoblieNet block, SE-блок, NAS: EfficientNet, RepVGG/MobileOne, <br>Приемы разработки и оптимизации специализированных архитектур для работы с изображениями:  разработка решений на основе сложных конфигураций сверточных нейронных сетей в timm, приемы обучения СНС.<br>Решение задачи компьютерного зрения с использованием кастомизации архитектуры под задачу<br>     | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/05%20%D0%A1%D0%BE%D0%B2%D1%80%D0%B5%D0%BC%D0%B5%D0%BD%D0%BD%D0%BE%D0%B5%20%D1%81%D0%BE%D1%81%D1%82%D0%BE%D1%8F%D0%BD%D0%B8%D0%B5%20%D0%A1%D0%9D%D0%A1.pptx)   | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS6_Tricks_In_Train.ipynb)  | [Invite](https://www.kaggle.com/t/37d51df9321249ac90299ab055237838) |
| 4.  | Архитектуры трансфромеры для построения продуктов с компьютерным зрением   | Архитектура ViT и ее модификации: SwiGLU, MOE, RMSNorm,  DeiT и другие<br>Другие архитектуры типа трансфромеры: SWIN, MobileViT, архитекутры-миксеры, ConvNeXt<br>Исследование и имплементация SOTA-моделей в задачах анализа изображений на примере семейства моделей ViT в экосистеме HuggingFace<br>Построение пайплайна. Комбинирует несколько моделей в комплексные решения<br>     | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/06%20%D0%A1%D0%B5%D1%82%D0%B8%20%D0%BD%D0%B0%20%D0%BE%D1%81%D0%BD%D0%BE%D0%B2%D0%B5%20%D0%B0%D1%80%D1%85%D0%B8%D1%82%D0%B5%D0%BA%D1%82%D1%83%D1%80%20%D1%82%D1%80%D0%B0%D0%BD%D1%81%D1%84%D0%BE%D1%80%D0%BC%D0%B5%D1%80%D0%BE%D0%B2.pptx)    | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS9%20-%20Hugging%20Face.ipynb)    |    -    |
| 5.  | Задачи сегментации изображений                                                   | Переход от задач классификации к задачам Dense Prediction на примере сегментации изображений.<br>Архитекутры Энкодер-Декодер на примере U-Net,<br>Архитектуры головной части на примере DeepLab V3/расширенная свертка и нелинейное рецептивное поле.<br>Архитектуры трансформеры для сегменатции изображений: SegFormer  и другие архитектуры.<br>Применение алгоритмов и библиотек компьютерного зрения для прикладных задач сегментации изображений: SMP, albumentation,<br>Решение задачи семантической сегментации с использованием дообученных моделей<br> | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/07%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D1%81%D0%B5%D0%BC%D0%B0%D0%BD%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%BE%D0%B9%20%D1%81%D0%B5%D0%B3%D0%BC%D0%B5%D0%BD%D1%82%D0%B0%D1%86%D0%B8%D0%B8.pptx)                                                                                                                                                                                                                                                           | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS7_Semantic%20Segmentation.ipynb)                                                                                                                       | [Invite](https://www.kaggle.com/t/4f7b3ae36cc14385bb68b20d7eaef4dd) |
| 6.  | Задачи обнаружения объектов                                                      | Многоэтапные архитектуры обнаружения объектов на примере Faster-R-CNN/Mask-R-CNN.<br>Быстрые одноэтапные архитектуры на примере семейства архитектур YOLO<br>Архитектуры трансформеры для обнаружения объектов на примере сеймества DETER + RT-DETER<br>Применение алгоритмов и библиотек компьютерного зрения для прикладных задач обнаружения объектов на изображениях на примере YOLO<br>Решение задачи семантической обнаружение объектов с использованием дообученных моделей<br>                                                                           | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/08%20%D0%97%D0%B0%D0%B4%D0%B0%D1%87%D0%B0%20%D0%BE%D0%B1%D0%BD%D0%B0%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%BE%D0%B1%D1%8A%D0%B5%D0%BA%D1%82%D0%BE%D0%B2.pptx)                                                                                                                                                                                                                                                                                         | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS8_YOLOv8%20API.ipynb)                                                                                                                                  | [Invite](https://www.kaggle.com/t/5026ff51255641c68c92d4d04a285c3d) |
| 7.  | Подходы к построению базисных моделей компьютерного зрения                       | Принципы самообучения (MAE, I-JEPA), Контрастное обучение (DINO), мультимодальное обучение (CLIP)<br>Базисные архитектуры решения отдельных задач (SAM, Ground DINO)<br>Обзор принципов построения VLM<br>Разработка стратегий применения базисных моделей CV (компьютерного зрения): zero-shot learning, linerar/non-linear probe и других,<br>Решение задач распознавания текста на изображении<br>                                                                                                                                                            | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/09%20%D0%91%D0%B0%D0%B7%D0%B8%D1%81%D0%BD%D1%8B%D0%B5%20%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8.pptx)                                                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                     |                                                                     |
| 8.  | Решение прикладных задач компьютерного зрения для работы с видеопотоком          | Обзор особенностей видеоданных в задачах компьютерного зрения. Примеры задач для видеоданных.<br>Использование CNN и архитекутр трансформеров в обработке видео данных<br>Построение пайплайн обработки видеопотока при помощи нейронных сетей.<br>Решение задач object tracking<br>                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                                                                                     |                                                                     |
| 9.  | Обзор классических подходов к решению задач компьютерного зрения                 | Классические подходы к решению задач компьютерного зрения: поиск изображений по признакам SIFT, и по эталону, методы обнаружения объектов (Виолы-Джонса), Сегментация Watershed, Понятие Оптический поток<br>Решение задачи компьютерного зрения классическими методами<br>                                                                                                                                                                                                                                                                                      | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/03%20%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5%20%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B%20%D1%80%D0%B5%D1%88%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%20%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D0%BE%D0%B3%D0%BE%20%D0%B7%D1%80%D0%B5%D0%BD%D0%B8%D1%8F.pptx)                                                                                                                                | [GitHub](https://github.com/MVRonkin/Classical-computer-vision)                                                                                                                                                                                     |                                                                     |
| 10. | Предпосылки соврменного компьютерного зрения: Переход от LeNet к ResNet          | Повтор понятий сверточный блок, рецептивное поле, регуляризация, нормализация, обучение СНС и других понятий),<br>Построение кастомных сверточных нейронных сетей в PyTorch + torchvision: training pipeline, аугментация, работа с набором данных..<br>Решение задачи классификации методами компьютерного зрения<br>                                                                                                                                                                                                                                           | [GitHub](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/LEC/04%20%D0%9F%D1%80%D0%B8%D0%BD%D1%86%D0%B8%D0%BF%D1%8B%20%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%8B%20%D1%81%D0%B2%D0%B5%D1%80%D1%82%D0%BE%D1%87%D0%BD%D1%8B%D1%85%20%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D1%8B%D1%85%20%D1%81%D0%B5%D1%82%D0%B5%D0%B9.pptx)                                                                                                                                                                                                     | [GitHub 1](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS3_LeNet.ipynb), [GitHub 2](https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/WS/WS4_Make_CNN_Better.ipynb)              | [Invite](https://www.kaggle.com/t/bdd0a0c4cd4f489bb3d2db81a36e9d33) |


Материалы состоят из 

1.	лекционных презентаций по следующим темам<ul>
<li>1.1.	Введение в предмет компьютерное зрение</li>
<li>1.2.	Понятия о признаках изображений и методы работы с цифровыми изображениями.</li>
<li>1.3.	Классические методы решения задач компьютерного зрения.</li>
<li>1.4.	Принципы и базовые архитектуры сверточных нейронных сетей</li>
<li>1.5.	Современное состояние архитектур нейронных сетей и моделей решения зада с их помощью</li>
<li>1.6.	Использование архитектур-трансформеров в компьютерном зрении</li>
<li>1.7.	Решение задач семантической сегментации</li>
<li>1.8.	Решение задач обнаружения объектов и связанных задач</li>
<li>1.9.	Базисные и другие современные модели для компьютерного зрения.</li>
</ul>
<br>

2.	материалов практик в формате ipynb по следующим темам<ul>
<li>2.1.	Использование методов цифровой обработки изображений в решении задач компьютерного зрения.</li>
<li>2.2.	Решение задач при помощи классических методов с ручным выделением признаков.</li>
<li>2.3.	Исследование и построение базовых архитектур сверточных нейронных сетей и настройка процесса обучения</li>
<li>2.4.	Перенос обучения и современные архитектуры для решения задач компьютерного зрения</li>
<li>2.5.	Решения задач семантической сегментации </li>
<li>2.6.	Решение задач обнаружения объектов и связанных задач</li>
<li>2.7.	Использование генеративных подходов при решении задач компьютерного зрения.</li>
<li>2.8.	Построение и оптимизация работы пайплайнов решения задач компьютерного зрения для изображений и видеопотоков.</li>
<li>2.9.	Решение задач поиска аномалий во ВР</li>
<li>2.10.	Решение задач классификации ВР (доп. Раздел).</li>
</ul>

3.	итогового задания 
https://github.com/MVRonkin/Computer-Vision-Course_lec-practice/blob/main/2025/README.md 

доп.литература
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

еще более доп. лит.
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
  
всякие интересные репозитории
* [Deep-Learning-From-Scratch](https://github.com/lakshyaag/Deep-Learning-From-Scratch)
* [Deep Learning Fundamentals: Code Materials and Exercises](https://github.com/Lightning-AI/dl-fundamentals)
* [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
* [the incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch)

инструменты
* [AlbumentationsX  - новая версия библиотеки Albumentation](https://github.com/albumentations-team/AlbumentationsX)
* [Vit=pytorch - библа о ViT](https://github.com/lucidrains/vit-pytorch)
* [lightly - библиотека для SSL](https://docs.lightly.ai/self-supervised-learning/index.html)
* [skorch - high api  в стиле sklearn для торч](https://github.com/skorch-dev/skorch)
* [fast.ai - very fast torch compatible framework](https://www.fast.ai/)
