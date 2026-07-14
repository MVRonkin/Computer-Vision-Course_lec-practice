![](09Foundation_393.jpg)

![](09Foundation_394.png)

__Базисные и генеративные модели __

_Тренды и другие задачи компьютерного зрения\. _

_Курс: Компьютерное зрение\. _

Ронкин Михаил Владимирович

к\.т\.н\. доцент ИРИТ\-РТФ\, УРФУ

https://rohitbandaru\.github\.io/blog/Vision\-Language\-Models/

https://github\.com/lightly\-ai/lightly

https://arxiv\.org/pdf/2304\.12210

# Ограничения supervised models

Image Processing

\- Ручная формализация признаков и принятие решений

Очень ограниченные задачи\, до 100 изображений

\- Ручная формализация признаков и автомат\. принятие решений

Ограниченные задачи\, до 1000 изображений

\- Автоматическая формализация признаков и принятия решений

Ограниченное предобучение

Интерпретируемость\, Формализмазция

Ограниченные задачи\, до 1 млн изображений

\- Автоматическое решение на основе специального предобучения

Несколько задач\, до 100 млн изображений и не только

Foundation model

\- SSL предобучение\, zero\-shot

Несколько задач\, до 100 млн изображений и не только

Сложность модели/абстрактность задач и их широта и количество\, Объем данных

__Архитектуры и данные__

Сложность задачи

AGI \(Гипотетический этап\)

Система обучения архитектур

Мультимодальные и мультиагентные НС

Система из архитектур

Базисные модели \(LLM\, дифуз\. Сети\, \.\.\)

«Широкая» архитектура

Универсальная архитектуры кодировщика признаков

архитектуры кодировщика признаков для модальности

Архитектуры под модальности

Машинное обучение

Описание признаков

Модельная статистика

__Архитектуры и данные__

Объем данных растет – растет разнообразие данных\, растет объем моделей

Растет время на разметку данных – растет число ошибок

Данные любого датасета – нужно считать конечными – требуется OOD обобщение

![](09Foundation_395.png)

![](09Foundation_396.png)

Разметка Imagenet \(14 млн семплов\, 21 841 классов\) заняла 3 года у 49 000 человек и содержит 6% ошибок

https://uni\-tuebingen\.de/fakultaeten/mathematisch\-naturwissenschaftliche\-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous\-vision/lectures/computer\-vision/

https://arxiv\.org/abs/2103\.14749

https://zshn25\.github\.io/self\-supervision\-for\-foundation\-models/

# Базисная модель (Foundation Model, FM)

Базисная модель \(Foundation Model\, FM\):

Модель предобученная на огромных наборах данных \(large X model\, LxM\)\.

В силу указанных прежде  причин \- как правило предобучение self\-supervision \(SSL\) at scale\.

![](09Foundation_397.png)

https://en\.wikipedia\.org/wiki/Foundation\_model

__См\. коллекцию базисных моделей для комп\. Зрения __  __https://__  __huggingface\.co__  __/collections/__  __merve__  __/foundation\-models\-for\-vision__

# self-supervision learning (SSL) at scale



* Веса модели\, предобученной на ImageNet’e\, оказываются субоптимальными\.
  * У каждого датасета есть свое смещение признаков\,
  * Классы ImageNet – не всегда подходящие
  * Известно что разметка ImageNet содержит ошибки
* SSL предобучение позволяет достичь более высоких показателей на известных датасетах без их непосредственной разметки


![](09Foundation_398.png)

https://uni\-tuebingen\.de/fakultaeten/mathematisch\-naturwissenschaftliche\-fakultaet/fachbereiche/informatik/lehrstuehle/autonomous\-vision/lectures/computer\-vision/

https://alexanderdyakonov\.wordpress\.com/2020/06/03/самообучение\-self\-supervision/

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/

https://readmedium\.com/zero\-shot\-learning\-understanding\-machines\-that\-learn\-like\-humans\-e670f83186b8

# Базисная модель (foundation model, FM)

В основе идеи FM переход от парадигмы supervised learning  к парадигме самобучения – Self\-supervised learning

SSL позволяет FM освоить общие закономерности данных

– обобщить не только задач в train\, но между задачами\.

__ __ Идея: SSL at scale – приводит к надежному выделению признаков в данных

![](09Foundation_399.png)

https://en\.wikipedia\.org/wiki/Foundation\_model

# Cвойства FM



* FM можно применять в широком круге задач с небольшими затратами на дообучение или без дообучения \(zero\-shot learning\)\.
    * Хорошие Out\-of\-distribution обобщение
* Модели могут быть мультимодальными  \(например image\-text\)
* Появление эмерджентных свойств \(свойств\, которых нет в supervised моделях\)
  * Известно\, что при тех же показателях целевой метрики SSL требует меньше данных чем supervised


![](09Foundation_400.png)

https://en\.wikipedia\.org/wiki/Foundation\_model

# self-supervision learning (SSL)

__Self\-Supervised__  __ __  __Learning__  \(или unsupervised\) \- режим обучения моделей\, при котором задача обучения формируется исходя из внутренней структуры самих объектов\, либо из базовых знаний об объектах\.

Метка в SSL – искаженное входное изображение\.

Цель SSL – извлечение признакового описания изображений\.

![](09Foundation_401.png)

https://habr\.com/ru/articles/704710/

https://encord\.com/blog/self\-supervised\-learning/

https://insights\.willogy\.io/self\-supervised\-learning\-part\-1\-simple\-and\-intuitive\-introduction\-for\-beginners/

# Виды SSL at scale



  * Context Prediction/Generative/ Masked Image Modeling
  * Автоэнкодер: восстановление маскированных или искаженных изображений\,
  * Псевдометка: Предсказание класса искажений \(или след\. кадра в видео\)
  * World Models/Knowledge Matching: Выучивание представлений разных частей изображений
  * image\-Joint embedding architectures\, V\-JEPA \(эмбединги кадров и тд\)\.
  * self\-distillation
  * Контрастивное обучение \(Contrastive Learning\)
  * Контрастные:  Отличия схожих и разных изображений


![](09Foundation_402.jpg)

![](09Foundation_403.jpg)

![](09Foundation_404.jpg)

https://habr\.com/ru/articles/704714/

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/

https://insights\.willogy\.io/self\-supervised\-learning\-part\-1\-simple\-and\-intuitive\-introduction\-for\-beginners/

https://arxiv\.org/abs/2304\.12210

https://alexanderdyakonov\.wordpress\.com/2020/06/03/самообучение\-self\-supervision/

# Подходы к SSL. Context Prediction



* Masked Image Modeling with autoencoder / Generative SSL\.
* Цель – обучить модель восстанавливать фичи
* Достоинства: не требуется даже слабая разметка\, возможно обучение нескольких моделей \(самодисциляция\)


![](09Foundation_405.jpg)

![](09Foundation_406.png)

![](09Foundation_407.jpg)

![](09Foundation_408.jpg)

__Augmentation __  __Prediction__

__Masked Position Prediction__

https://github\.com/lucidrains/vit\-pytorch

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/

# Виды SSL. Сontrastive Learning



* Цель – контрастного обучения \- максимально разнести похожие и разные примеры в батче
* Достоинство: может быть мультимодальным \(вместо пар картинок\, пара картинка\-изображение\, но требуется слабая разметка\)
* Подход с примерами разных и похожих классов более устойчив \(седловидная точка\)


![](09Foundation_409.png)

![](09Foundation_410.png)

![](09Foundation_411.png)

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/



  * Идея: фичи должны
  * отображать\, как изображения соотносятся друг с другом
  * должны быть инвариантны к внешним факторам \(местоположение\, освещение\, цвет\)
  * То есть нужно не только восстонавливать фичи\, но и отличать их от других


![](09Foundation_412.png)

![](09Foundation_413.png)

https://drive\.google\.com/file/d/1CFn\_zTzdx\-xg6x\-\_NdPAqI4gIfWrftKm/view

# SSL  Pretext tasks vs Downstream task



* предобучение  \- использование на столько большого количества данных – как это возможно\.
  * Тут не имеет значение как данные связаны с задачей\,
  * ЦЕЛЬ – выучить наиболее робастные фичи между данными\.
* После предобучения FM дообучают для решения так называемых “downstream” задач \(специальных задач CV — классификации\, сегментации и др\.\)\.


![](09Foundation_414.png)

# Стратегии использования FM. Promt-based

Promt\-based  использование FM

https://arxiv\.org/html/2407\.12210v2

Zero\-shot learning:

Задав вход мы ищем к какому кластеру признаков мы ближе всего

Few\-shot learning:

Сравнение неизвестного изображения с представителями разных известных

Оne\-shot learning:

Сравнение известного и неизвестного входа

![](09Foundation_415.jpg)

![](09Foundation_416.jpg)

![](09Foundation_417.png)

![](09Foundation_418.png)

https://blog\.deepschool\.ru/cv/few\-shot\-learning/

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/

https://readmedium\.com/zero\-shot\-learning\-understanding\-machines\-that\-learn\-like\-humans\-e670f83186b8

https://alexanderdyakonov\.wordpress\.com/2020/06/03/самообучение\-self\-supervision/

https://encord\.com/blog/one\-shot\-learning\-guide/

# Стратегии использования FM. Fine Tune

Fine\-tuning\-based  использование FM

![](09Foundation_419.jpg)

Дистилляция знаний

![](09Foundation_420.png)

Non\-linear probling

![](09Foundation_421.png)

Supervised Fine\-Tuning\, SFT

Parameter efficient SFT \(PEFT\): LORA

![](09Foundation_422.jpg)

![](09Foundation_423.png)

https://arxiv\.org/html/2407\.12210v2

https://arxiv\.org/pdf/2401\.10222

https://arxiv\.org/html/2401\.10222v2

https://medium\.com/@sayedebad\.777/mastering\-lora\-low\-rank\-adaptation\-of\-llms\-be4ed2293ed2

https://cleverx\.com/blog/advanced\-supervised\-fine\-tuning\-sft\-trends\-pitfalls\-and\-whats\-next\-in\-2025

https://www\.ibm\.com/think/topics/lora\#523796992

# Стратегии использования FM.

FM как часть архитектуры OMNI моделей

Visual\-language models \(VLM\)

Open\-vocabulary адаптеры

![](09Foundation_424.png)

![](09Foundation_425.png)

![](09Foundation_426.png)

Visual\-Language\-Action models

![](09Foundation_427.png)

![](09Foundation_428.png)

https://arxiv\.org/pdf/2401\.10222

https://arxiv\.org/html/2401\.10222v2

https://cleverx\.com/blog/advanced\-supervised\-fine\-tuning\-sft\-trends\-pitfalls\-and\-whats\-next\-in\-2025

https://lilianweng\.github\.io/posts/2022\-06\-09\-vlm/

# DINO (self-DIstillation with NO labels)



* __Идея__  __ DINO __ \- самообучение по средствам подстраивайся двух сетей
* Самодистилляция знаний – цель выдавать одинаковые эмединги для нескольких разных патчей одного изображения \(в отличии от обычного contrastive learnin не требуется явно негативных примеров\)\.
  * Multi\-Crop Augmentation: несколько кропов с разной аугментаций
* Student: Обучается стандартным методом backprop
* Teacher: Веса обновляются скользящим средним весов студента \(momentum encoder\)
  * Учитель \- это более стабильной версией студента


![](09Foundation_429.png)

Local and global crops

![](09Foundation_430.png)

![](09Foundation_431.png)

![](09Foundation_432.png)

![](09Foundation_433.png)

![](09Foundation_434.png)

![](09Foundation_435.png)

![](09Foundation_436.png)

Предотвр\. колапса

![](09Foundation_437.png)

https://blog\.deepschool\.ru/cv/dino\-self\-distilation\-with\-no\-labels/?ysclid=micx9x33wy491219292

https://theaisummer\.com/self\-supervised\-representation\-learning\-computer\-vision/

https://readmedium\.com/review\-dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-cfddbb4d3549

https://towardsdatascience\.com/dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-summary\-ab91df82cc3c/

# DINO

![](09Foundation_438.png)

F\_s  — модель\-студент\, F\_t — модель\-учитель

В течение эпохи веса учителя F\_t заморожены\.

Берем картинку из датасета\, делаем 2 глобальных и несколько локальных кропа \(напр\. 128х128 и 96х96\)\,

подаем на вход F\_t один глобальный патчей\, а на вход F\_s подаем остальные патчи

Обе модели выдают на выход эмбеддинги одинакового размера\.

Производится softmax с температурой

результат сравниваем с помощью лосса и бэкпропагейтим градиенты в модель\-студент F\_s

Для F\_t веса aF\_s \+\(1\-a\)F\_t \(EMA\)

![](09Foundation_439.png)

https://readmedium\.com/review\-dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-cfddbb4d3549

https://teletype\.in/@atmyre/ooMFzB7YADA?ysclid=mibynnbgsp272715591

![](09Foundation_440.png)

Цель EMA не позволить свести к колапсу \(когда модель дает один и тот же вектор для всех изображений\)

F\_s вынужден подстраиваться под F\_t

Что бы не сводить колапс к равномерному распр\. Используется центрирование \(смещение \+ загрубление входа сети\) – по сути аналог нормализации

В DINO V2 дополнительно используется расчет энтропии между патчами и маскирование\.

![](09Foundation_441.png)

![](09Foundation_442.png)

https://readmedium\.com/review\-dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-cfddbb4d3549

https://teletype\.in/@atmyre/ooMFzB7YADA?ysclid=mibynnbgsp272715591

# DINO (self-DIstillation with NO labels)

Полезное \(эмерджентное\) свойство DINO для моделей ViT \.

Патчи моделей ViT при DINO\-самообучении содержат явную информацию о семантической сегментации объекта \(его границах\)

![](09Foundation_443.png)

Разные головы “подсвечивают” различные объекты или части объектов на изображении

![](09Foundation_444.png)

DINO достигает 78\.3 % точности на ImageNet без дополнительного дообучения

Self\-distillation вообще не требует разметки\!

https://arxiv\.org/pdf/2304\.07193

https://readmedium\.com/review\-dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-cfddbb4d3549

https://blog\.deepschool\.ru/cv/dino\-self\-distilation\-with\-no\-labels/?ysclid=micx9x33wy491219292

Если при помощи DINO визуализировать категории объектов из  ImageNet\, можно увидеть\, что похожие категории располагаются рядом друг с другом\.

Это говорит о том\, что модели удаётся связать категории на основе визуальных свойств\, подобно людям\.

Например\, виды животных чётко разделены по структуре\, напоминающей биологическую таксономию\.

«ноги» собаки\, «ноги» лошади и «ноги» стола имеют похожие ембединги

![](09Foundation_445.png)

https://neurohive\.io/ru/frameworki/dino/

https://readmedium\.com/review\-dino\-emerging\-properties\-in\-self\-supervised\-vision\-transformers\-cfddbb4d3549

https://blog\.deepschool\.ru/cv/dino\-self\-distilation\-with\-no\-labels/?ysclid=micx9x33wy491219292

![](09Foundation_446.png)

примеры признаков\, выученных SD DINO

https://sd\-complements\-dino\.github\.io/

# CLIP – мультимодальное контрастное обучение

__Идея: контрастное обучение на парах__  __ image – text__  __ позволяет:__

__Автоматическая разметка __ \(zero\-shot\) без участия людей\.

__Поиск похожих изображений/дубликатов __ \.

__Получение робастных __  __ембеддингов__

__Связь текста и изображений __ в генеративных задачах

![](09Foundation_447.png)

Задача предварительного обучения \(предобучения\)\, заключается в том\, чтобы для каждого изображения из пары изображение\-текст угадать\, какое текстовое описание \( __promt__ \) соответствует именно ему\.

https://blog\.deepschool\.ru/cv/clip/

[https://diophontine\.github\.io/csc412/slides/w10/tut10\.pdf](https://diophontine.github.io/csc412/slides/w10/tut10.pdf) https://habr\.com/ru/articles/539312/ [https://habr\.com/ru/articles/546586/](https://habr.com/ru/articles/546586/)\. https://habr\.com/ru/articles/908168/

CLIP\, BLIP\, SIGLIP и др\.

# CLIP



* Image: ResNet или ViT encoder \+ слой кодирования изображения в 512\-мерное пр\-во
* Text: transformer  \+слой  кодирования текст в вектор 512\-мерное пр\-во\.
* Сеть обучалась на массиве 400 млн пар изображений\-описаний\, собранных из сети интернет без указания классов
* При обучении картинка с подходящей подписью должны быть близки\, а с неподходящей \- максимально далеки в пространстве embedding\-ов\.
  * Прогоняем изображения через image encoder и тексты через text encoder\.
  * Получаем N векторов для изображений и N векторов для текстов\.
  * Сравниваем их по косинусной мере или MSE\.
  * Получаем матрицу сравнений NxN\, по которой вычисляется функция потерь \(сross\-entropy loss\)\.


![](09Foundation_448.png)

[https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/) [https://habr\.com/ru/articles/546586/](https://habr.com/ru/articles/546586/) [https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/) [https://habr\.com/ru/articles/539312/](https://habr.com/ru/articles/539312/)

https://habr\.com/ru/articles/540312/

# self-supervision at scale CLIP

__Результат __  __Zero\-shot inference __ : вместо обучения на конкретных классах вы передаёте модели текстовые описания \(«кошка»\, «собака»\, «машина»\)\, и она выбирает наиболее подходящее\.

![](09Foundation_449.png)

Запрос формулируется очень специфически:  _а _  _photo of a \_\_\_\__  или  _a centered satellite photo of \_\_\_\_\. _

Така\] формулировка позволяет более узко и специфично производить настройку или адаптацию под конкретный датасет

Zero\-shot CLIP оказывается более устойчивым к сдвигу распределений\, чем модель обученная на ImageNet\.

CLIP – хорошо смотрит на самые яркие фичи\, но плохо работает с деталями

[https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/) [https://habr\.com/ru/articles/546586/](https://habr.com/ru/articles/546586/) [https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/)

https://amaarora\.github\.io/posts/2023\-03\-06\_Understanding\_CLIP\.html

# CLIP-like модели

Помимо оригинального CLIP также есть OpenCLIP

Модификации под бекбоны\, например BCLIP

Модификции под домены: MedCLIP\, StreetCLIP  и тд

Модели для различных задач\, напр\. CLIPseg\, DiffusionCLIP\, GroundingLIP \(GLIP\) – obj\.det

Модель GroundDINO позволяет хорошо детектить объекты

Модель CoCa показывает наивысшую точность на ImageNet

Модели для распределенных вычислений SigLIP\,

Улучшение работы с мелкими деталями SigLIPv2 \(MaskAE\, Self\-Distil\, фича в каждом патче\)

__Семейство моделей пополняется\.__

![](09Foundation_450.png)

https://github\.com/yzhuoning/Awesome\-CLIP

https://sh\-tsang\.medium\.com/brief\-review\-openclip\-reproducible\-scaling\-laws\-for\-contrastive\-language\-image\-learning\-fe5ed4ea7161 https://sh\-tsang\.medium\.com/glip\-grounded\-language\-image\-pre\-training\-2be2483295b3

[https://roboflow\.com/model\-alternatives/clip](https://roboflow.com/model-alternatives/clip) [https://encord\.com/blog/open\-ai\-clip\-alternatives/](https://encord.com/blog/open-ai-clip-alternatives/) [https://github\.com/LAION\-AI/CLIP\_benchmark](https://github.com/LAION-AI/CLIP_benchmark)

# Segment Anything Models

__Image __  __encoder__  __ – __ ViT \+ MAE \(Masked SSL\)

__Энкодер дополнительной инфы \(__  __кондишена__ \)\, на основе которой нужно сегментировать картинку\.

Prompt энкодер – кодирование промпта в дополнение к фичам

__mask decoder\. __ Принимает на вход эмбеддинг картинки из image encoder и эмбеддинг допинфы\. Архитектура — декодер трансформера

mask decoder выдает три карты сегментации \(для нечетких случаев\) \+ confidence

Модель MAE дообучена на датасете с разметкой \(но на 99% автоматической\)

![](09Foundation_451.png)

<span style="color:#0563c1">[https://blog\.deepschool\.ru/cv/segment\-anything\-sam](https://blog.deepschool.ru/cv/segment-anything-sam/)</span> [/](https://blog.deepschool.ru/cv/segment-anything-sam/)  <span style="color:#0563c1">[https://teletype\.in/@atmyre/](https://teletype.in/@atmyre/hrnsBpXZMll)</span> [hrnsBpXZMll](https://teletype.in/@atmyre/hrnsBpXZMll)



* __Энкодер дополнительной инфы \(__  __кондишена__ \)\, на основе которой нужно сегментировать картинку\.
  * __Mask__  __ __ — через image encoder\. эмбеддинг потом суммируется с эмбеддингом из image encoder;
  * __Points__  — набор точек\, которые относятся к объекту\. Переводятся в эмбеддинг с помощью positional encoding’а\.
  * __Box__  — заданный юзером bounding box объекта\, который нужно сегментировать\. Box также переводится в эмбеддинг с помощью positional encoding\.
* Текстовый энкодер через обучение CLIP
* случаев\) \+ confidence


![](09Foundation_452.png)

![](09Foundation_453.png)

<span style="color:#0563c1">[https://blog\.deepschool\.ru/cv/segment\-anything\-sam](https://blog.deepschool.ru/cv/segment-anything-sam/)</span> [/](https://blog.deepschool.ru/cv/segment-anything-sam/)  <span style="color:#0563c1">[https://teletype\.in/@atmyre/](https://teletype.in/@atmyre/hrnsBpXZMll)</span> [hrnsBpXZMll](https://teletype.in/@atmyre/hrnsBpXZMll) [https://blog\.deepschool\.ru/cv/segment\-anything\-sam/](https://blog.deepschool.ru/cv/segment-anything-sam/)

# Segment Anything Models V2/V3



* __SAM 2__  — обобщение архитектуры SAM\, которое должно хорошо работать на изображениях и видео\. При этом новой модели необходимо получать промпты из точек\, масок и боксов на отдельных фреймах и выделять этот объект на протяжении всего видео\.
  * Использован иерархический трансформер HIERA
* __SAM __  __3__ — комбайн с расширенными функциями\. Свободный промпт\,  __detection\, segmentation\, and tracking__  __\, __  __детекция__  __ по примеру __  __изобржений__  __\.__


![](09Foundation_454.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">www\.ultralytics\.com</span>  <span style="color:#0563c1">/blog/exploring\-sam\-3\-meta\-ais\-new\-segment\-anything\-model</span>

https://blog\.deepschool\.ru/cv/segment\-anything\-model\-2/

# SAM-Like models

__SAM __  __подход является одним из наиболее популярных сегодня для полу\-автоматической разметки__

__SAM1/SAM2/SAM3 – __  __основное семейство__

__HQ\-SAM__  __ – для тонких линий и объектов__

__MEDSAM /SAM4MIS –__  __ для медицинских данных__

__SAM\-Video – __  __для трекинга__

__TRex__  __ –__  __ __  __детекция__  __ всех __  __объктов__  __ по выбранному__

Grounding Dino – детекция по промпту

FastSAM/Yolo\-World/YOLOE – SAM\-like архитектуры для YOLOv8

MobileSAM/ FASTER SAM – дистилированный SAM

DepthAnything V1/V2 \-  предсказание по глубине

https://huggingface\.co/collections/merve/segment\-anything\-model

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">blog\.deepschool\.ru</span>  <span style="color:#0563c1">/cv/segment\-anything\-model\-2/</span>

https://habr\.com/ru/companies/recognitor/articles/786646/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">www\.ultralytics\.com</span>  <span style="color:#0563c1">/blog/exploring\-sam\-3\-meta\-ais\-new\-segment\-anything\-model</span>

https://docs\.ultralytics\.com/ru/models/fast\-sam/

https://sh\-tsang\.medium\.com/brief\-review\-faster\-segment\-anything\-towards\-lightweight\-sam\-for\-mobile\-applications\-c226bd0a3a25

https://habr\.com/ru/companies/sberdevices/articles/757606/

# VLM



* Один из основных фокусов развития LLM сегодня — это попытка сделать их мультимодальными \(omni‑моделям\)\, причины:
  * __Больше данных для обучения__
  * __Больше задач – больше разнообразия учит модель \- __  __датасет__  __\-независимое обучение __
  * __Гипотеза о системе и составляющих: __  __OMNI\-__  __модель лучше в каждой из своих задач__
* __Visual\-Language Model \(VLM\)__  объединяет зрение и язык\, позволяя моделям не просто «видеть»\, но и понимать\, что они видят\.
  * VLM – дают текст на выходе\!


![](09Foundation_455.png)

![](09Foundation_456.png)

VLM – Text Alignment модели

https://huggingface\.co/blog/vlms\-2025

https://github\.com/huggingface/nanoVLM/tree/main

https://habr\.com/ru/companies/yandex/articles/847706/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">habr\.com</span>  <span style="color:#0563c1">/</span>  <span style="color:#0563c1">ru</span>  <span style="color:#0563c1">/companies/</span>  <span style="color:#0563c1">ru\_mts</span>  <span style="color:#0563c1">/articles/944942/</span>

# VLM состоит из



* Визуального энкодера \(например\, ViT\, CLIP или свёрточной сети\)
* Языковой модели \(LLM\)\.
* Компоненты связываются  __адаптером__ \.
  * __Deep Fusion \(cross\-attention\)__
  * __Early Fusion__


![](09Foundation_457.png)

Адаптер выравнивает визуальные и текстовые представления в общем пространстве эмбеддингов

Используя предобученные модели\, дообучаем VLM на мультимодальном картиночно‑текстовом домене\.

https://blog\.deepschool\.ru/llm/v\-llm/

https://habr\.com/ru/companies/yandex/articles/847706/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">habr\.com</span>  <span style="color:#0563c1">/</span>  <span style="color:#0563c1">ru</span>  <span style="color:#0563c1">/companies/</span>  <span style="color:#0563c1">ru\_mts</span>  <span style="color:#0563c1">/articles/944942/</span>

# VLM - Обучение



* LLM  и Visual Encoder  предобучены\.
* Как правило обучение VLM:
* Обучение адаптера \(pretraining\)
  * Адаптер учится связывать модальности
  * Новые домены за счет адаптера\, например OCR
  * Требуется 1e7 – 1e8 данных\, можно слабо\-размеченные \(напр\. Wiki\)
* Дообуение всей VLM \(alignment\): SFT\+ RL \(опционально\)


![](09Foundation_458.jpg)

__цель __ pretraining  — перевести представления с энкодера в пространство языковой модели\. Как правило\, для этого используются данные с описаниями\, а также большое количество OCR\-like данных\.

__цель __  __дообучения__  — научить модель работать с полученными представлениями\, правильно отвечать на вопросы и следовать инструкциям\, связанным с изображениями\.

<span style="color:#0563c1">[https://habr\.com/ru/companies/ru\_mts/articles/944942/](https://habr.com/ru/companies/ru_mts/articles/944942/)</span>  <span style="color:#0563c1"> </span>  <span style="color:#0563c1">https://habr\.com/ru/companies/yandex/articles/847706/</span>

https://aman\.ai/primers/ai/VLM/

# VLM - Предобучение

В последнее время активно развивается подход [interleaved](https://arxiv.org/pdf/2306.16527)\-претрейна\, суть которого заключается в использовании изображений внутри текста\. Это позволяет учить и языковую\, и визуальную часть одновременно\, но требует хорошо подготовленных данных и грамотной настройки\.

Другие стратегии: Image\-captition

table\-and\-charts understanding\, text crop

Web Code reconstruction

![](09Foundation_459.png)

![](09Foundation_460.png)

![](09Foundation_461.png)

https://arxiv\.org/pdf/2403\.05525

<span style="color:#0563c1">[https://habr\.com/ru/companies/ru\_mts/articles/944942/](https://habr.com/ru/companies/ru_mts/articles/944942/)</span>  <span style="color:#0563c1"> </span>  <span style="color:#0563c1">https://habr\.com/ru/companies/yandex/articles/847706/</span>

https://blog\.deepschool\.ru/llm/v\-llm/

# VLM - Обучение



* LLM  и Visual Encoder  предобучены\.
* Дообуение VLM \(alignment\): SFT\+ RL \(опционально\):
  * Высококачественные данные под конкретные сценарии использования\,
  * в т\.ч\. Text\-only\, OCR или другие
* Также в ряде случаев Alightment \(RLHF\)


![](09Foundation_462.png)

https://habr\.com/ru/companies/yandex/articles/847706/

LLM рефразер Яндекс

<span style="color:#0563c1">[https://habr\.com/ru/companies/ru\_mts/articles/944942/](https://habr.com/ru/companies/ru_mts/articles/944942/)</span>

# VLM – оценка качества

Оценка качества на бенчмарка

Оценка качества ассесером \(Side‑by‑Side\) по критериям \(напр\. Рассуждение\, читаемость и тд\)

High\-resolution – понимание мелких деталей

![](09Foundation_463.png)

![](09Foundation_464.png)

https://habr\.com/ru/companies/yandex/articles/847706/

[https://blog\.deepschool\.ru/llm/v\-llm//](https://habr.com/ru/companies/ru_mts/articles/944942/)

# VLM состоит из



* __Prompt\-based __  __адаптеры __ – репрезентация изображения в последовательность токенов\.
  * Качество лучше\, съедают контекст входной\, адаптер может быть простой MLP
  * Наиболее простой адаптер – MLP
  * Могут быть и более сложные варианты\, в т\.ч\. С сжатием многомасштабного патчнига в единый эмединг\.


![](09Foundation_465.png)

https://habr\.com/ru/companies/yandex/articles/847706/

https://blog\.deepschool\.ru/llm/v\-llm/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">habr\.com</span>  <span style="color:#0563c1">/</span>  <span style="color:#0563c1">ru</span>  <span style="color:#0563c1">/companies/</span>  <span style="color:#0563c1">ru\_mts</span>  <span style="color:#0563c1">/articles/944942/</span>

# Q-Fromer (BLIP2)

Основной модуль Querying Transformer or Q\-Former \- идея того\, что k\,v от изображения\, а q можно выучить для кодирования изображений\.

1 обучение только queries \(32 × 768\):

ITM \(Image\-Text Matching\): бинарная классификация совпадения пары\.

ITG \(Image\-grounded Text Generation\): генерация текста по изображению  \.

ITC \(Image\-Text Contrastive\): выравнивание эмбеддинга \(like a CLIP\)\.

2 дообучение в полной модели

![](09Foundation_466.png)

Через  _attention_  queries извлекают из image релевантную информацию\.

LLM рефразер Яндекс

https://thepythoncode\.com/article/visual\-question\-answering\-with\-transformers\-in\-python

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">readmedium\.com</span>  <span style="color:#0563c1">/q\-former\-1d83163975da</span>

# VLM состоит из



* cross‑attention‑based адаптеры \(Deep Fusion \)
* основная идея  \- выход энкодера – в K\, V   cross‑attention‑блок LLM\)
    * Адаптер сложный\, много параметров\.
    * Не потребляет контекст


![](09Foundation_467.png)

![](09Foundation_468.png)

https://habr\.com/ru/companies/yandex/articles/847706/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">rohitbandaru\.github\.io</span>  <span style="color:#0563c1">/blog/Vision\-Language\-Models/</span>

# QWEN-VL

Visual Encoder – SigLIP/SAM\-B

LLM – DeepSeek

Adapter – CNN\-based

1 pre\-training \- обучается только адаптер \(LLM и ViT frozen\)\.

2 multi\-task — адаптер \+ LLM\.

3 SFT  — instruction\-based fine\-tuning\.

Visual Encoder – VLIP \(Vit\-G\)

LLM – Qwen\-7B

Adapter – MLP

1 pre\-training \- обучается только адаптер \(LLM и ViT frozen\)\.

2 multi\-task — всё обучается совместно\.

3 SFT  — замораживается ViT\, дообучается адаптер \+ LLM\.

![](09Foundation_469.png)

![](09Foundation_470.png)

https://rohitbandaru\.github\.io/blog/Vision\-Language\-Models/

https://sh\-tsang\.medium\.com/brief\-review\-deepseek\-vl\-towards\-real\-world\-vision\-language\-understanding\-caf9838afd97

# VLA

__Vision\-Language\-Action \(VLA\)__  модели представляют собой новый этап развития робототехники и воплощённого искусственного интеллекта\. Они объединяют восприятие\, рассуждения и действия в единую систему\, позволяют роботам видеть окружающий мир\, интерпретировать команды\, сформулированные на естественном языке\, и выполнять их с учётом контекста\. VLA снижают зависимость от ручного программирования и заранее прописанных сценариев\, открывают путь к роботам\, способным работать в  __неструктурированных и динамичных средах__  — например\, на кухне\, в мастерской или складе\.

На рисунке 4 показана типичная архитектура VLA\, включающая четыре основных компонента:  __визуальный энкодер\,__   __языковую модель\,__   __энкодер состояния робота__ \,  __декодер действий__ \.

![](09Foundation_471.png)

![](09Foundation_472.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">blog\.deepschool\.ru</span>  <span style="color:#0563c1">/dl/vision\-language\-action\-vla\-kak\-ii\-uchitsya\-videt\-ponimat\-i\-dejstvovat\-v\-realnom\-mire/?</span>  <span style="color:#0563c1">utm\_source</span>  <span style="color:#0563c1">=</span>  <span style="color:#0563c1">tg&utm\_medium</span>  <span style="color:#0563c1">=</span>  <span style="color:#0563c1">post&utm\_campaign</span>  <span style="color:#0563c1">=</span>  <span style="color:#0563c1">blog\-article&utm\_content</span>  <span style="color:#0563c1">=</span>  <span style="color:#0563c1">deepschool&utm\_term</span>  <span style="color:#0563c1">=</span>  <span style="color:#0563c1">vla</span>

# Ground DINO Open Vocabulary obj det

__Ground DINO__  __ – идея добавить __  __cross\-modality attention __  __поверх __  __DETR__

__	__  __Grounding__  \- установление связи между текстовым описанием и 	локальной областью на изображении\.

Таким образом добавляется  __Open Vocabulary __  __object detection__

__	__  __	zero\-short object detection__

![](09Foundation_473.png)

![](09Foundation_474.png)

https://www\.digitalocean\.com/community/tutorials/grounding\-dino\-1\-5\-open\-set\-object\-detection

https://medium\.com/axinc\-ai/grounding\-dino\-detect\-any\-object\-from\-text\-29808580cb32

https://learnopencv\.com/fine\-tuning\-grounding\-dino/

__Главные элемент __  __Text \+ Image Neck part__  __ и __  __text2bbox decoder__

__Основная идея__  __ __  __Neck __  __: микширование текстовых признаков и изображения\.__

__Основная идея __  __t__  __ext2bbox decoder__  __ __  __: __  __отбор __  __bbox__  __ __  __ при помощи __  __text __  __k\,v__  __ __  __и __  __image queries\.__

![](09Foundation_475.png)

![](09Foundation_476.png)

https://pyimagesearch\.com/2025/12/08/grounding\-dino\-open\-vocabulary\-object\-detection\-on\-videos/

https://blog\.roboflow\.com/grounding\-dino\-zero\-shot\-object\-detection/

https://medium\.com/axinc\-ai/grounding\-dino\-detect\-any\-object\-from\-text\-29808580cb32

# YOLO-World

__YOLO\-World __  __– __  __Open Vocabulary YOLO \(__  __Open Vocabulary object __  __detection\)__

__	__  __Grounding__  \- установление связи между текстовым описанием и 	локальной областью на изображении\.

Не требуется NMS\, anchors – отбор ROI по query словоря\.

__		zero\-short object detection__

![](09Foundation_477.png)

https://medium\.com/axinc\-ai/grounding\-dino\-detect\-any\-object\-from\-text\-29808580cb32

https://habr\.com/ru/articles/791154/

__NECK \- Re\-__  __parameterizable__  __ __  __Vision\-Language __  __PAN:__

__I\-pooling – Image2text embedding__

__T\-CSP – __  __добавление текста к изображению\.__

__Box __  __Head__  __ – предсказание  рамок__

__Text Contrastive __  __Head__  __ \-  сопоставление рамок и текста__

__YOLO Detector – yolo 8__

__Text Encoder – CLIP__

![](09Foundation_478.png)

![](09Foundation_479.png)

https://medium\.com/axinc\-ai/grounding\-dino\-detect\-any\-object\-from\-text\-29808580cb32

https://habr\.com/ru/articles/791154/

# Open-Vocabulary VLM

__prompt\-then\-detect__  подход

__YOLO 8 Detector__

__CLIP Text Encoder__

![](09Foundation_480.png)

![](09Foundation_481.png)

https://blog\.roboflow\.com/what\-is\-yolo\-world/

![](09Foundation_482.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">blog\.deepschool\.ru</span>  <span style="color:#0563c1">/cv/</span>  <span style="color:#0563c1">vlm\-dlya\-detekczii\-obektov\-na\-izobrazhenii</span>  <span style="color:#0563c1">/</span>

https://habr\.com/ru/articles/791154/

Пожалуй\, основной и главный строительный блок всей архитектуры\. Состоит он из top\-down и bottom\-up частей\, внутри которых производится сопоставление ранее извлеченных текстовых эмбеддингов и multi\-scale фичей изображения\.

Блок внутри себя включает 2 основных компонента:  __CSPLayer__  и  __Image\-Pooling Attention__ \. Говоря обывательским языком\, первый пытается добавить языковую информацию в элементы изображения\, а второй наоборот\, заложить информацию с изображения в текстовые эмбеддинги:

После RepVL\-PAN следуют блоки  __Box Head __ и  __Text Contrastive Head__ \. Если первый\, очевидно\, предсказывает ограничительные рамки объектов\, то второй их эмбеддинги \(на основании близости объект\-текст\)\.

Таким образом\, в конце пайплайна мы имеем ограничительные рамки и эмбеддинги объектов на изображении\, а также векторы текстов классов\, которые хотим обнаружить\. С помощью своего рода мэтчинга\, сравнивая попарные близости полученных векторов в рамках боксов\, на выходе получим список найденных классов с соответствующими вероятностями \(при заданном пороге близости\)\.

В первом приближении это все про саму модель\! Для того\, чтобы не усложнять изложения обзора\, я не стал приводить формул и строгих математических выкладок\. Желающих разобраться во всех деталях и тонкостях я отправляю к прочтению оригинальной статьи\. Там же можно ознакомиться с тем\, как все это обучается с помощью  __Region\-Text Contrastive Loss\, __ а также найти описание множества экспериментов по дообучению под конкретные задачи на разных датасетах для сравнения с предыдущими решениями\.

![](09Foundation_483.png)

https://habr\.com/ru/articles/791154/

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">blog\.deepschool\.ru</span>  <span style="color:#0563c1">/cv/</span>  <span style="color:#0563c1">vlm\-dlya\-detekczii\-obektov\-na\-izobrazhenii</span>  <span style="color:#0563c1">/</span>

https://habr\.com/ru/articles/791154/

# VLA

https://github\.com/gokayfem/awesome\-vlm\-architectures

https://aman\.ai/primers/ai/VLM/

# QWEN-VL



* LLM  и Visual Encoder  предобучены\.
* Дообуение всей VLM \(alignment\): SFT\+ RL \(опционально\):
  * тест на высококачественных данных под конкретные сценарии использования\,
  * в т\.ч\. Text\-only\, OCR или другие
  * Оценка качества на бенчмарка
  * Оценка качества ассесером \(Side‑by‑Side\) по критериям \(напр\. Рассуждение\, читаемость и тд\)


![](09Foundation_484.png)

https://rohitbandaru\.github\.io/blog/Vision\-Language\-Models/

LLM рефразер Яндекс

# VLM

Qwen2\.5\-VL объединяет энкодер изображений и декодер языковой модели для обработки мультимодальных данных\. Энкодер работает с данными в их исходном разрешении\, преобразуя изображения разных размеров и видео с различным FPS в последовательности токенов разной длины\. Особенность MRoPE — выравнивание временных меток с абсолютным временем\, что помогает модели лучше понимать временную динамику событий и точно определять моменты\.

![](09Foundation_485.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">habr\.com</span>  <span style="color:#0563c1">/</span>  <span style="color:#0563c1">ru</span>  <span style="color:#0563c1">/news/884652/</span>

refixLM is a Natural Language Processing \(NLP\) technique used for training models\. It starts with part of a sentence \(a prefix\) and learns to predict the next word\. In Vision\-Language Models\, PrefixLM helps the model predict the next words based on an image and a given piece of text\. It uses a Vision Transformer \(ViT\)\, which breaks an image into small patches\, each representing a part of the image\, and processes them in sequence\.

![](09Foundation_486.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">www\.ultralytics\.com</span>  <span style="color:#0563c1">/blog/understanding\-vision\-language\-models\-and\-their\-applications</span>

them in sequence\.

![](09Foundation_487.jpg)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">www\.ultralytics\.com</span>  <span style="color:#0563c1">/blog/understanding\-vision\-language\-models\-and\-their\-applications</span>

Qwen2\.5\-VL объединяет энкодер изображений и декодер языковой модели для обработки мультимодальных данных\. Энкодер работает с данными в их исходном разрешении\, преобразуя изображения разных размеров и видео с различным FPS в последовательности токенов разной длины\. Особенность MRoPE — выравнивание временных меток с абсолютным временем\, что помогает модели лучше понимать временную динамику событий и точно определять моменты\.

![](09Foundation_488.png)

![](09Foundation_489.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">habr\.com</span>  <span style="color:#0563c1">/</span>  <span style="color:#0563c1">ru</span>  <span style="color:#0563c1">/companies/</span>  <span style="color:#0563c1">ru\_mts</span>  <span style="color:#0563c1">/articles/944942/</span>

# self-supervision at scale SigCLIP

Цель SigLIP — упростить распределённое обучение на очень больших батчах\, сохранив при этом качество эмбеддингов\. Для этого заменяется softmax \(multiclass\) на sigmoid \(мультилейюл\)

![](09Foundation_490.png)

[https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/) [https://habr\.com/ru/articles/546586/](https://habr.com/ru/articles/546586/) [https://habr\.com/ru/articles/908168/](https://habr.com/ru/articles/908168/)

# Порождающие сети

![](09Foundation_491.png)

https://lilianweng\.github\.io/posts/2021\-07\-11\-diffusion\-models/



* Среди порождающих подходов следует выделить два:
  * Вариационные автоэнкодеры \( _Variational_  _ _  _Autoencoders_  _\,_  _ _  _VAE\)_
  * _Генеративно_  _\-состязательные сети _  _\(Generative\-Adversarial Network\, GAN\)_
    * _Подходы могут комбинироваться_
    * _Сегодня есть также ряд альтернативных подходов\, например\, деформационный или основанных на _  _zero\-short learning_


# Вариационный автоэенкодер VAE



* _Вариационные _  _автоэнкодеры_  \( _Variational_  _ _  _Autoencoders_ \) — это автоэнкодеры\, которые учатся отображать объекты в непрерывное скрытое пространство и\, затем\, воссоздавать их из него\.
* Суть VAE в кодировании информации в виде нормального распределения с параметрами
  * мат\. ожидания и дисперсии\, отображающими распределение каждого входного класса\.
    * Но не классы как таковые\.
    * Такое распределение называется скрытым пространством
      * Скрытое пространство имеет меньшую размерность\, чем основное
    * Суть вариационного декодера в формировании из скрытого пространства выходного изображения при помощи обученных весов\.


# Порождающие состязательные сети (GAN)

__Модель обучающаяся порождать по средствам итеративных сравнения своих результатов в тренировочными примерами\.__

_генератор _  __\(__  __generator__  __ __  _G_  __\)__  __ __ \- порождает объекты в пространстве данных\,

_дискриминатор _  __\(__  __discriminator__  __;__  __ __  _D_  __\)\, __ учится отличать порожденные генератором объекты от настоящих примеров из обучающей выборки\.



* __цель дискриминатора __ — по заданному примеру\, выглядящему как элемент пространства данных\, решить\, был ли он «настоящим» или был порожден генератором;
* __цель генератора __ — «обмануть» дискриминатор\, сделать так\, что дискриминатор не сможет различить распределение данных и распределение\, которое порождает генератор
* Таким образом обучение GAN представляет собой минимаксную задачу:
  * Минимизация ошибки генератора при максимизации вероятности найти ошибку со стороны дискриминатора\.


# Некоторые типы GAN



* __Без учителя __ \(GAN\, Deep Convolution GAN \(DCGAN\)\, WGAN\)
* __Полуконтролируемое__  __ обучение __
  * Conditional GAN – обучение дискреминатора и генератора с меткой класса\.
  * InfoGAN – генератор получает метку класса\, должен выучить ее классификацию самостоятельно\.
    * Под меткой класса может быть не только традиционный класс объекта\, но и некоторые его семантические особенности\, например цвет глаз\, размер объетка\.
* __Перенос изображений __ \(Image 2 Image translation\)
  * CycleGan \(обучение на основе изображений двух категорий\)


![](09Foundation_497.png)



* __Без учителя __ \(GAN\, Deep Convolution GAN \(DCGAN\)\, WGAN\)
* __Полуконтролируемое__  __ обучение __
  * Conditional GAN – обучение дискреминатора и генератора с меткой класса\.
  * InfoGAN – генератор получает метку класса\, должен выучить ее классификацию самостоятельно\.
    * Под меткой класса может быть не только традиционный класс объекта\, но и некоторые его семантические особенности\, например цвет глаз\, размер объетка\.
* __Перенос изображений __ \(Image 2 Image translation\)
  * CycleGan \(обучение на основе изображений двух категорий\)


![](09Foundation_498.png)

[https://rohitbandaru\.github\.io/blog/Vision\-Language\-Models/](https://rohitbandaru.github.io/blog/Vision-Language-Models/)

[https://rohitbandaru\.github\.io/blog/Vision\-Language\-Models/](https://rohitbandaru.github.io/blog/Vision-Language-Models/)

# Порождающие сети

# Использование порождающих сети



* __Генерация новых изображений\.__
* __Исследование и интерпретирование нейронных сетей __
  * позволяют проверить\, насколько хорошо понято распределение данных;
* __semi\-supervised learning __  __ \- сети могут обучаться с недостатком данных и без разметки;__
  * Оценка начального распределяя p\(x\)
* __Обучение __  __мультимодальным__  __ выходам\, __ когда есть несколько правильных ответов;
  * например\, задача предсказания следующего кадра в видеоролике\, когда несколько действий имеют одинаковую вероятность\.
* __могут служить моделями окружающего мира в обучении с__  __ __  __подкреплением\.__
* __Могут использоваться для генерации данных как таковых\.__


# GAN. Обучение

![](09Foundation_499.jpg)

# Соревновательный автоэенкодер

Обычный автоэенкодер \(сверху\) \+

результаты кодировки накладываются на нормальное распределение

Задача дискриминатора – различать примеры из исходного распределения  и примеры из распределения\, генерируемого энкодером \.

Таким образом кодировщик выучивает неявное априорное распределение\, или же говорят\, что задан implicit prior\, который как бы закодирован внутри сети\, но мы не можем получить его форму аналитически\.

![](09Foundation_500.jpg)

![](09Foundation_501.jpg)

# Порождающие состязательные сети (GAN). DCGAN

![](09Foundation_502.jpg)

![](09Foundation_503.jpg)

![](09Foundation_504.jpg)

# Порождающие состязательные сети. Conditional GAN

![](09Foundation_505.jpg)

_Проблема обычного _  _GAN _  _– модель может_  _ _  _выбрать только ряд мод и генерировать_  _ _  _только из них\._

_Напр\. выбрать только 3 и_  _ _  _7 из всех цифр и генерировать только из них\._

_Использование явных меток позволяет повысить независимость результатов\._

![](09Foundation_506.jpg)

![](09Foundation_507.jpg)

# Порождающие сети. InfoGAN



* __Info GAN __  __к задаче оптимизации__  __ GAN __  __добавляются  дополнительные теоретико\-информационные ограничения\. __
* _Задача обучить «распутанное» представление \(_  _disentangled_  _ _  _representation_  _\)\, в котором отдельные признаки имеют естественную интерпретацию\. _
  * Например\, хотелось бы\, чтобы признаки у GAN\, генерирующей человеческое лицо\, соответствовали цвету глаз\, форме
* Обычные GAN не накладывают ограничения на вектор скрытого представления\,
  * в теории\, генератор может начать использовать факторы очень нелинейно или не связано\, тем самым ни один из факторов не будет отвечать за какой либо семантический признак \.


![](09Foundation_508.jpg)

![](09Foundation_509.jpg)

![](09Foundation_510.jpg)

![](09Foundation_511.jpg)

![](09Foundation_512.jpg)

# Вариационный автоэенкодер VAE+GAN

Недостаток чистого VAE – размытость данных\, по этому к нему лучше добавить дискриминатор\, тогда получится VAE\+GAN модель

![](09Foundation_513.jpg)

![](09Foundation_514.jpg)

![](09Foundation_515.jpg)

# Cycle GAN

![](09Foundation_516.jpg)

Идея в том\, чтобы обучить

Два Генератора так\, чтобы каждый умел работать с двумя наборами данных\.

![](09Foundation_517.jpg)

__ПОДХОДЫ К РЕШЕНИЮ ЗАДАЧ НА ОСНОВЕ ДАННЫХ__

тут надо сказать что с ростом объемов данных ростет и сложность алгоритмов для принятия решний на них\, при этом снижается необходимость в формализации\, повышается абстрактность постановки задачи\, но и снижается интерпретируемость решний\. сегодня мы на этапе когда понимаем как отказаться от формальной постановки задачи\, но следующий этап это отказ и от формализации процесса обучения алгоритма\.

Четкие правила \(if\-else\)

Логический вывод и экспертные системы \(эвристики\)

Model based applied statistic

_Классическое стат\. решение задачи_

Data\-driven methods \(Model\-agnostic\, Machine Learning\):

_Замена формальной  модели на наборы_  _ _  _формализованных признаков и ответы для них\._

__Интерпретируемость\, Формализмазция__



  * Deep Learning \(глубокое обучение нейронных сетей\)


__Абстрактность постановки задачи __

_Замена формальной  модели на наборы сырых данных и ответы для них\. _

Foundation models \(Большие генеративные модели\)

_Широкий круг задач\, допустим набор ответов на одно воздействие_

Сильный ИИ \(AGI гипотетический этап\)

_Отказ от формализации обучения модели_

_Сложность задачи\,_  _ _  _Сложность модели\, Объем данных_

# Виды SSL at scale

![](09Foundation_518.jpg)



* Маскированный Автоэнкодер\. Цель – обучить модель восстанавливать фичи
  * Иногда вместо маски восстанавливают агентированные изображения
  * Или предсказывают величину поворота изображения\, патч изображения или его позицию
* Самодистилляция\. Цель
* Контрастное обучение\. Цель – максимально разнести похожие и разные примеры в батче


![](09Foundation_519.png)

![](09Foundation_520.png)

https://blog\.deepschool\.ru/dl/self\-supervised\-learning/

# self-supervision at scale

__Contrastive learning –__

__ __ Сближать векторные представления \(эмбеддинги\) «похожих» изображений \(например\, двух аугментированных версий одного объекта\)\,

Удалять от векторов «непохожих» изображений \(из других классов или контекстов\)\.

В классической формулировке триплет лосс \- выбирается 1 изображение как якорь\, ему ставят в соответствие 1 позитивный и несколько негативный примеров

Есть и другие постановки\, например knn

![](09Foundation_521.png)

Это ключевой подход в создании foundation models \(базовых моделей\) для компьютерного зрения — таких как CLIP\, DINO\, SimCLR\, MoCo и др\.

[https://uvadlc\-notebooks\.readthedocs\.io/en/latest/tutorial\_notebooks/tutorial17/SimCLR\.html](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html) [https://habr\.com/ru/articles/704716/](https://habr.com/ru/articles/704716/)

# FM – это Генеративные модели

![](09Foundation_522.png)

https://neerc\.ifmo\.ru/wiki/index\.php?title=Порождающие\_модели

# Segment Anything Models V3

__SAM __  __3__ — комбайн с расширенными функциями\. Свободный промпт\,  __detection\, segmentation\, and tracking__  __\, __  __детекция__  __ по примеру __  __изобржений__  __\.__

![](09Foundation_523.png)

<span style="color:#0563c1">https://</span>  <span style="color:#0563c1">www\.ultralytics\.com</span>  <span style="color:#0563c1">/blog/exploring\-sam\-3\-meta\-ais\-new\-segment\-anything\-model</span>

