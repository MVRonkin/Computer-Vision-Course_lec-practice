# Foundation and Generative Models

**Trends and Other Computer Vision Tasks**
*Course: Computer Vision*

Mikhail Vladimirovich Ronkin
PhD, Associate Professor, IRIT-RTF, UrFU

---

## Limitations of Supervised Models

**Image Processing**

| Approach | Data Scale | Characteristics |
|---|---|---|
| Manual feature engineering + manual decision-making | Up to 100 images | Very limited tasks |
| Manual feature engineering + automatic decision-making | Up to 1,000 images | Limited tasks |
| Automatic feature extraction + decision-making | Up to 1M images | Limited pre-training; interpretability required |
| Automatic solving with specialised pre-training | Up to 100M images | Multiple tasks |
| **Foundation Model** — SSL pre-training, zero-shot | 100M+ images (and beyond) | Many tasks simultaneously |

*Model complexity / task abstraction / data volume all increase together.*

**Architecture evolution ladder** (bottom → top by task complexity):

1. Statistical models / feature descriptions
2. Task-specific feature encoders
3. Universal feature encoders
4. "Wide" architectures
5. Foundation models (LLM, diffusion networks, …)
6. Multi-modal and multi-agent neural networks
7. Systems of architectures
8. *(Hypothetical)* AGI — systems that learn architectures

---

## Architectures and Data

- As data volume grows, so does data diversity and model size.
- Annotation time grows → annotation errors accumulate.
- Any dataset's data must be treated as **finite** → OOD generalisation is required.

> ImageNet (14M samples, 21,841 classes) took **3 years**, **49,000 people**, and still contains **~6% label errors**.

---

## Foundation Model (FM)

A **Foundation Model** is a model pre-trained on a massive dataset (*large X model, LxM*).

Due to the data-scale challenges above, pre-training is almost always done via **self-supervised learning (SSL) at scale**.

> See the collection of vision foundation models:
> https://huggingface.co/collections/merve/foundation-models-for-vision

---

## Self-Supervised Learning (SSL) at Scale

- Weights pre-trained on ImageNet are **sub-optimal** for many downstream datasets:
  - Each dataset has its own feature bias.
  - ImageNet classes do not always match the target domain.
  - ImageNet labels are known to contain errors.
- SSL pre-training can reach **higher performance** on known benchmarks **without directly labelling them**.

---

## Foundation Model — Core Idea

The FM paradigm shifts from **supervised learning** to **self-supervised learning**.

SSL lets an FM learn **general data patterns** — generalising not just within training tasks but **across tasks**.

> **Key insight:** SSL at scale → reliable, robust feature extraction from data.

---

## Properties of Foundation Models

- FMs can be applied to a **wide range of tasks** with minimal fine-tuning or even **zero-shot**.
  - Strong **out-of-distribution (OOD)** generalisation.
- Models can be **multi-modal** (e.g. image–text).
- **Emergent properties** appear that are absent in supervised-only models.
  - At the same target metric, SSL typically requires **fewer labelled examples** than supervised training.

---

## Self-Supervised Learning (SSL)

**Self-Supervised Learning** (also called unsupervised learning) is a training regime in which the learning task is derived from the **internal structure of the data itself** or from basic prior knowledge about it.

- The "label" in SSL is a **corrupted or masked version** of the input.
- The goal is to learn a **feature representation** of images.

---

## Types of SSL at Scale

| Category | Core idea |
|---|---|
| **Context Prediction / Generative / Masked Image Modeling** | Reconstruct masked or distorted images; predict augmentation class or next video frame |
| **World Models / Knowledge Matching** | Learn representations of different image parts (V-JEPA, self-distillation) |
| **Contrastive Learning** | Push similar images together and dissimilar images apart in embedding space |

---

## SSL Approach: Context Prediction

**Masked Image Modeling with Autoencoder / Generative SSL**

- Goal: train the model to **reconstruct features**.
- Variants:
  - **Augmentation Prediction** — predict what augmentation was applied.
  - **Masked Position Prediction** — predict the content of masked patches.
- Advantages: requires **no annotation at all**; enables training multiple models (self-distillation).

---

## SSL Approach: Contrastive Learning

- Goal: **maximise the distance** between dissimilar examples and **minimise the distance** between similar ones within a batch.
- Can be **multi-modal** (image–text pairs instead of image–image pairs), but requires weak supervision.
- Using both positive and negative examples is more stable (saddle-point optimisation).

> Feature requirements:
> - Capture **how images relate to each other**.
> - Be **invariant** to external factors (position, lighting, colour).
> - Not only reconstruct features, but also **distinguish** them from others.

---

## SSL: Pretext Tasks vs. Downstream Tasks

- **Pre-training** — use as much data as possible.
  - It does not matter how the data relates to the final task.
  - Goal: learn the **most robust cross-data features**.
- After pre-training, FMs are **fine-tuned** for specific *downstream tasks* (classification, segmentation, detection, etc.).

---

## Strategies for Using FMs

### Prompt-based

| Strategy | Description |
|---|---|
| **Zero-shot learning** | Given an input, find the nearest feature cluster — no training examples needed |
| **Few-shot learning** | Compare an unknown image against a small set of class representatives |
| **One-shot learning** | Compare a known example with an unknown input |

### Fine-tuning-based

| Strategy | Description |
|---|---|
| **Non-linear probing** | Freeze the backbone; train only a small head |
| **Supervised Fine-Tuning (SFT)** | Fine-tune the whole model on labelled data |
| **Parameter-Efficient SFT (PEFT / LoRA)** | Train only low-rank adapter matrices; keep most weights frozen |
| **Knowledge Distillation** | Transfer knowledge from a large teacher model to a smaller student |

### FM as Part of OMNI-model Architecture

- **Visual-Language Models (VLM)**
- **Open-vocabulary adapters**
- **Visual-Language-Action models (VLA)**

---

## DINO — Self-Distillation with NO Labels

**Idea:** self-supervised learning through the simultaneous training of two networks.

- **Self-distillation:** the goal is to produce **identical embeddings** for several different patches of the same image (unlike standard contrastive learning, **no explicit negative examples** are needed).
- **Multi-Crop Augmentation:** several crops at different scales and augmentations.

**Architecture:**

| Component | Role |
|---|---|
| **Student** (*F_s*) | Trained via standard backpropagation |
| **Teacher** (*F_t*) | Weights updated as an **exponential moving average (EMA)** of student weights |

The teacher is a more stable version of the student.

**Training loop (per epoch):**

1. Take an image; create 2 global crops and several local crops (e.g. 128×128 and 96×96).
2. Feed one global crop to *F_t*; feed the rest to *F_s*.
3. Both networks produce embeddings of the same size.
4. Apply softmax with temperature; compare outputs with a loss; backpropagate into *F_s* only.
5. Update *F_t* weights: `F_t ← α·F_s + (1−α)·F_t` (EMA).

**Collapse prevention:** centering (bias + input sharpening — analogous to normalisation).

> **DINOv2** additionally uses inter-patch entropy and masking.

### Emergent Property of DINO for ViT

ViT patch representations trained with DINO contain **explicit semantic segmentation** information:

- Different attention heads highlight different objects or object parts.
- DINO achieves **78.3% accuracy on ImageNet** without any additional fine-tuning.
- If ImageNet categories are visualised using DINO embeddings, visually similar categories cluster together — mimicking **biological taxonomy**.
  - For example: a dog's *leg*, a horse's *leg*, and a table's *leg* share similar embeddings.

---

## CLIP — Multi-Modal Contrastive Learning

**Idea:** contrastive learning on **image–text pairs** enables:

- **Automatic labelling** (zero-shot) without human annotators.
- **Similar image / duplicate search**.
- **Robust embeddings**.
- **Linking text and images** in generative tasks.

**Architecture:**

- **Image encoder:** ResNet or ViT → 512-dimensional embedding.
- **Text encoder:** Transformer → 512-dimensional embedding.
- Trained on **400 million image–caption pairs** crawled from the internet (no class labels).

**Training objective:**

1. Pass all images through the image encoder; all texts through the text encoder.
2. Obtain *N* image vectors and *N* text vectors.
3. Compare with cosine similarity → N×N comparison matrix.
4. Optimise with **cross-entropy loss**: matching pairs should be close; non-matching pairs should be far apart.

**Zero-shot inference:** instead of training on specific classes, pass **text descriptions** ("cat", "dog", "car") and the model picks the most similar one.

> Prompt format matters: `"a photo of a ___"` or `"a centered satellite photo of ___"` gives a narrower, domain-specific alignment.

- Zero-shot CLIP is **more robust to distribution shift** than a model trained on ImageNet.
- CLIP focuses on the **strongest features** but struggles with **fine details**.

---

## CLIP-Like Models

| Model | Description |
|---|---|
| **OpenCLIP** | Reproducible CLIP with open weights and scaling laws |
| **BCLIP** | CLIP adapted to specific backbones |
| **MedCLIP, StreetCLIP** | Domain-specific CLIP variants |
| **CLIPseg** | CLIP adapted for segmentation |
| **GroundingDINO (GLIP)** | CLIP + DINO for open-vocabulary object detection |
| **CoCa** | Achieves highest ImageNet accuracy |
| **SigLIP** | Replaces softmax with sigmoid for efficient distributed training |
| **SigLIPv2** | Adds MaskAE + self-distillation + per-patch features for better detail |

> The family is growing. See: https://github.com/yzhuoning/Awesome-CLIP

---

## SigLIP

**Goal:** simplify distributed training on very large batches while preserving embedding quality.

**Key change:** replace the **softmax (multi-class)** loss with a **sigmoid (multi-label)** loss — each image–text pair is treated as an independent binary classification, eliminating the need for a global normalisation denominator across the full batch.

---

## Segment Anything Models (SAM)

**Architecture components:**

| Component | Description |
|---|---|
| **Image Encoder** | ViT + MAE (masked SSL) |
| **Prompt Encoder** | Encodes the conditioning input (points, boxes, masks, text via CLIP) into embeddings |
| **Mask Decoder** | Transformer decoder; takes image embedding + prompt embedding → 3 segmentation masks + confidence |

**Prompt types:**
- **Mask** — processed through the image encoder; embedding is summed with the image embedding.
- **Points** — a set of points belonging to the target object → positional encoding → embedding.
- **Box** — a user-defined bounding box → positional encoding → embedding.
- **Text** — encoded via a CLIP text encoder.

> The model outputs **three** segmentation masks (for ambiguous cases) + confidence scores.

The base model is MAE-based; fine-tuned on a dataset with annotations (99% automatically generated).

---

## SAM 2 and SAM 3

- **SAM 2** — extends SAM to **images and video**. Accepts prompts on individual frames and tracks the object throughout the full video. Uses the **HIERA hierarchical transformer**.
- **SAM 3** — a multi-function model: free-form prompts, **detection + segmentation + tracking** + detection by example image.

---

## SAM-Like Models

| Model | Specialisation |
|---|---|
| **HQ-SAM** | Fine lines and small objects |
| **MedSAM / SAM4MIS** | Medical imaging |
| **SAM-Video** | Object tracking in video |
| **TRex** | Detect all objects matching a selected example |
| **Grounding DINO** | Text-prompt detection |
| **FastSAM / YOLO-World / YOLOE** | SAM-like architectures on top of YOLOv8 |
| **MobileSAM / Faster SAM** | Distilled lightweight SAM |
| **Depth Anything v1/v2** | Monocular depth prediction |

---

## Visual-Language Models (VLM)

One of the main directions of LLM development today is making them **multi-modal (omni-models)**. Reasons:

- **More training data**.
- **More tasks → more diversity → dataset-independent learning**.
- **Hypothesis:** an omni-model is better at each of its individual tasks.

**VLM** combines vision and language, allowing models to not just "see" but to **understand** what they see.

> VLMs produce **text output**.

---

## VLM Architecture

A VLM consists of three parts:

1. **Visual encoder** (e.g. ViT, CLIP, or a CNN).
2. **Language model (LLM)**.
3. **Adapter** that aligns visual and textual representations in a shared embedding space.

**Adapter types:**

| Type | Description | Trade-offs |
|---|---|---|
| **Prompt-based (Early Fusion)** | Image → sequence of tokens via MLP or more complex module | Better quality; consumes input context |
| **Cross-attention-based (Deep Fusion)** | Visual encoder output → K, V in cross-attention blocks of the LLM | Complex; many parameters; does not consume context |

---

## VLM Training

Both the LLM and the visual encoder are **pre-trained separately**.

Typical VLM training pipeline:

**Stage 1 — Adapter pre-training:**
- Only the adapter is trained; LLM and ViT are frozen.
- Goal: align modalities (image embeddings → LLM token space).
- Data: 10⁷–10⁸ weakly-labelled samples (e.g. Wikipedia image captions, OCR-like data).

**Stage 2 — Full VLM fine-tuning (alignment): SFT + optional RL:**
- High-quality data for specific use cases.
- Includes text-only, OCR, and other instruction-following data.
- Optional: RLHF (Reinforcement Learning from Human Feedback).

---

## VLM Pre-training Strategies

| Strategy | Description |
|---|---|
| **Image captioning** | Standard image–description pairs |
| **Interleaved pre-training** | Images embedded within text; trains both modalities simultaneously |
| **Table and chart understanding** | Structured visual data |
| **Text crop / web code reconstruction** | Screenshots of web pages, UI elements |

---

## VLM Quality Evaluation

- **Benchmark evaluation** — standardised test sets.
- **Human assessor evaluation (Side-by-Side)** — annotators score outputs by criteria (reasoning, readability, etc.).
- **High-resolution understanding** — ability to perceive fine visual details.

---

## Q-Former (BLIP-2)

The core module is the **Querying Transformer (Q-Former)**.

**Key idea:** K and V come from the image; Q vectors are **learnable** queries that extract relevant information from the image via cross-attention.

**Training in two stages:**

*Stage 1 — train only the 32 × 768 queries:*
- **ITC (Image-Text Contrastive)** — embedding alignment (like CLIP).
- **ITM (Image-Text Matching)** — binary classification of whether a pair matches.
- **ITG (Image-grounded Text Generation)** — generate text from an image.

*Stage 2 — fine-tune in the full VLM.*

---

## QWEN-VL Architecture

**DeepSeek-VL variant:**
- Visual Encoder: SigLIP / SAM-B
- LLM: DeepSeek
- Adapter: CNN-based
- Training: (1) adapter only → (2) adapter + LLM → (3) SFT

**QWEN-VL variant:**
- Visual Encoder: CLIP (ViT-G)
- LLM: Qwen-7B
- Adapter: MLP
- Training: (1) adapter only → (2) full joint training → (3) SFT (ViT frozen, adapter + LLM trained)

---

## Vision-Language-Action Models (VLA)

**VLA** models represent a new stage in robotics and embodied AI. They combine **perception, reasoning, and action** in a single system, enabling robots to:

- See the surrounding world.
- Interpret natural-language commands.
- Execute them in context.

VLA reduces dependence on manual programming and hard-coded scenarios, enabling robots to operate in **unstructured and dynamic environments** (kitchens, workshops, warehouses).

**Typical VLA architecture components:**

1. **Visual encoder**
2. **Language model**
3. **Robot state encoder**
4. **Action decoder**

---

## Grounding DINO — Open-Vocabulary Object Detection

**Idea:** add **cross-modality attention** on top of DETR.

**Grounding** — establishing a link between a textual description and a local image region.

This adds **open-vocabulary object detection** (zero-shot object detection).

**Key architectural elements:**

| Element | Role |
|---|---|
| **Text + Image Neck** | Mixes textual and visual features |
| **text2bbox Decoder** | Selects bounding boxes using text K, V and image queries |

---

## YOLO-World — Open-Vocabulary YOLO

**Goal:** open-vocabulary object detection without NMS or anchors — ROI selection is done via vocabulary query.

**Architecture:**

| Component | Description |
|---|---|
| **YOLO 8 Detector** | Base detector |
| **CLIP Text Encoder** | Encodes class name prompts |
| **RepVL-PAN Neck** | Bidirectional (top-down + bottom-up) feature fusion |
| **I-Pooling** | Image → text embedding |
| **T-CSP** | Adds text information to image features |
| **Box Head** | Predicts bounding boxes |
| **Text Contrastive Head** | Matches boxes to text queries |

**Pipeline:** bounding box embeddings + class text vectors → pairwise similarity → matched detections at a given similarity threshold.

---

## Generative Models — Overview

Among generative approaches, two are most prominent:

- **Variational Autoencoders (VAE)**
- **Generative Adversarial Networks (GAN)**
  - These approaches can be combined.
  - Additional alternatives exist (deformation-based, zero-shot-based, diffusion-based).

---

## Variational Autoencoder (VAE)

**VAEs** are autoencoders that learn to map objects into a **continuous latent space** and reconstruct them from it.

- The encoder maps input images to a **normal distribution** (mean + variance), representing the distribution of each class — not hard class labels.
- This distribution is called the **latent space** (lower-dimensional than the input).
- The decoder reconstructs output images from samples drawn from the latent space.

> **VAE drawback:** outputs tend to be blurry.
> **Solution:** add a discriminator → **VAE + GAN** hybrid.

---

## Generative Adversarial Network (GAN)

A model trained to **generate** data through iterative comparison with training examples.

| Component | Goal |
|---|---|
| **Generator** (*G*) | Produce samples that look like real data |
| **Discriminator** (*D*) | Distinguish generated samples from real ones |

**Minimax objective:**
- **Discriminator** maximises its ability to detect fakes.
- **Generator** minimises the discriminator's ability to detect them.

---

## GAN Variants

| Category | Examples |
|---|---|
| **Unsupervised** | GAN, DCGAN, WGAN |
| **Semi-supervised** | **Conditional GAN** — both G and D conditioned on class label; **InfoGAN** — G receives label and must learn to use it; labels can represent semantic attributes (eye colour, size, etc.) |
| **Image-to-Image Translation** | **CycleGAN** — unpaired image translation between two domains |

---

## DCGAN (Deep Convolutional GAN)

Replaces fully-connected layers with **transposed convolutions** in the generator and **strided convolutions** in the discriminator, enabling stable generation of higher-resolution images.

---

## Conditional GAN

**Problem with vanilla GAN:** the model can collapse to a few modes (e.g. generate only digits "3" and "7" from all 10 classes).

**Solution:** provide **explicit class labels** to both G and D, increasing output diversity and independence across classes.

---

## InfoGAN

**InfoGAN** adds **information-theoretic constraints** to the GAN objective.

- **Goal:** learn a *disentangled representation* where individual latent dimensions have natural, interpretable meaning.
  - e.g. separate dimensions for eye colour, face shape, object size.
- Standard GANs place no constraints on the latent vector → the generator may use factors non-linearly, with no dimension corresponding to any semantic attribute.

---

## Adversarial Autoencoder

**Standard autoencoder** + a constraint that encoder outputs **match a normal distribution**.

The discriminator's task is to distinguish:
- Samples from the **target prior distribution**.
- Samples from the **encoder's output distribution**.

The encoder learns an implicit prior — encoded inside the network rather than specified analytically.

---

## CycleGAN

Train **two generators** so that each can translate between two unpaired image domains:

- *G*: domain A → domain B
- *F*: domain B → domain A

**Cycle-consistency loss:** `F(G(x)) ≈ x` and `G(F(y)) ≈ y`

This enables unpaired image-to-image translation (e.g. horses ↔ zebras, summer ↔ winter).

---

## Uses of Generative Models

- **New image generation.**
- **Neural network research and interpretability** — verify how well the data distribution is understood.
- **Semi-supervised learning** — training with limited labelled data; estimate the prior p(x).
- **Multi-modal output modelling** — when multiple correct answers exist (e.g. predicting the next video frame).
- **World models for reinforcement learning.**
- **Synthetic data generation** for downstream training.

---

## Data-Driven Approaches — Evolution

| Approach | Characteristics |
|---|---|
| Hard rules (if-else) | Explicit logic; fully interpretable |
| Expert systems / heuristics | Logical inference; domain knowledge |
| Classical statistics | Model-based; formal feature specification |
| Machine Learning (data-driven, model-agnostic) | Replace formal models with labelled feature sets |
| **Deep Learning** | Replace formal models with raw data + labels; abstract task formulation |
| **Foundation Models (LLM, diffusion, …)** | Wide task coverage; multiple valid outputs per input |
| *(Hypothetical)* **AGI** | No formalisation of the training process required |

*As data volume grows → model complexity grows → task abstraction grows → interpretability decreases.*

---

## Self-Supervised Learning Summary

| Approach | Core Objective |
|---|---|
| **Masked Autoencoder** | Reconstruct masked or distorted image features; predict augmentation type, patch, or position |
| **Self-Distillation** | Student learns to match a slowly-updated teacher (EMA) |
| **Contrastive Learning** | Bring similar embeddings together; push dissimilar ones apart |

**Contrastive learning** is the key approach behind foundation models for computer vision: CLIP, DINO, SimCLR, MoCo, and others.

Classic formulation — **triplet loss:** one anchor, one positive, several negatives.

---

## FM and Generative Models

Foundation models are in large part **generative models** — they learn the underlying data distribution *p(x)*, enabling both representation learning and data generation.

---

*References and further reading available in the original lecture slides.*
