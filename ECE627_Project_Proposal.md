# ECE 627/527 Project Proposal
## Mission-Aware Generative Semantic Communications for UAV-to-Ground Image Classification

**Group Members:** _________________________ (Partner 1), _________________________ (Partner 2)  
**Submitted:** October 2026

---

## 1. Problem Formulation and Mission Context

### 1.1 Foundation & Extension
**Building on Grassucci et al. (2026):** Our project uses diffusion-based generative semantic communication as the primary technical foundation. We will reuse their core concepts (generative prior for semantic reconstruction) with proper citation, but our contribution is a **task-aware, mission-adaptive extension** specifically designed for UAV-to-ground image classification:

- **Grassucci et al. baseline:** Diffusion model for general-purpose semantic recovery
- **Our extension:** 
  1. Task-conditioned semantic encoder optimized for classification (not generic reconstruction)
  2. Adaptive bitrate/latent-dimension control based on channel SNR and mission priority
  3. Comparative evaluation on image classification with explicit fallback mechanisms for hallucination detection
  4. Analysis of when generation helps vs. when more transmitted bits are preferable

### 1.2 System Overview
We propose a **UAV-to-ground semantic communication system for real-time image classification**. A UAV observes a scene and must transmit compact task-conditioned representations to a ground station within strict bandwidth and latency constraints. The ground receiver performs inference to classify the observed objects/scenes and optionally regenerates a plausible reconstruction using a generative prior (building on diffusion models from Grassucci et al.).

### 1.2 Source and Dataset
- **Dataset:** CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
  - Manageable size, standard benchmark, realistic for embedded UAV processing
  - May subset to 5 classes if computational constraints arise
- **Transmitter:** Lightweight encoder running on UAV (e.g., ResNet-18 backbone, ~11M parameters)
- **Source Dimension:** 32×32 RGB images → task-conditioned latent vector (16–256 dims, adaptive)

### 1.3 Wireless Channel Model
- **Channel Type:** AWGN + optional Rayleigh fading to model atmospheric/terrain effects
- **SNR Range:** 0–20 dB (realistic for UAV-ground links)
- **Bandwidth Constraint:** Latent vector quantized to 64–512 bits per image
- **Latency Requirement:** <100 ms end-to-end (soft constraint for real-time awareness)

### 1.4 Mission Task and Utility
- **Primary Task:** Image classification (10 classes)
- **Mission Utility:** Maximize classification accuracy under bandwidth constraints
- **Task Utility Function:** Accuracy(SNR, transmission bits) — we want high classification accuracy even when channel is poor or budget is low
- **Adaptive Mechanism:** Reduce latent dimension or feature precision if SNR drops; increase transmission if mission-critical classes detected

---

## 2. Proposed Architecture

### 2.1 Semantic Encoder (UAV Transmitter)
- Lightweight task-aware encoder: ResNet-18 (pre-trained on ImageNet) → task-specific FC layers → latent vector
- Learns to extract **task-relevant features** for classification (not full reconstruction)
- Adaptive latent dimension: 32→256 dims based on SNR feedback or mission priority
- Quantization to integer bits before channel transmission

### 2.2 Differentiable Wireless Channel
- AWGN: `received = transmitted + noise, noise ~ N(0, σ²)` where `σ² = 1/(2·SNR)`
- Optional Rayleigh fading: `received = h·transmitted + noise, |h| ~ Rayleigh(1)`
- Trainable per sample or per batch; enables end-to-end optimization

### 2.3 Generative Receiver (Ground Station)
- **Generative Prior:** Lightweight VAE or conditional diffusion model trained on CIFAR-10
  - Alternative: Use pretrained prior from a public library (e.g., HuggingFace)
- **Decoder Path 1 (Classification):** Task decoder MLP → class logits → classification accuracy
- **Decoder Path 2 (Reconstruction):** Generative decoder uses transmitted latent + prior to reconstruct plausible image (optional, for analysis)
- **Confidence/Reliability:** Output prediction uncertainty; flag hallucination risk when reconstruction diverges from transmitted semantics

### 2.4 Adaptive Rate Controller
- Monitor SNR or received signal quality
- Adjust encoding bitrate, feature selection, or transmission priority:
  - High SNR: send full latent (512 bits) for high-fidelity features
  - Low SNR: compress to core task features only (64–128 bits)
- Optional: Mission-aware mode — prioritize certain classes (e.g., "always send enough bits to detect 'car'")

---

## 3. Comparison Baselines

### Baseline 1: Conventional Pipeline (Non-semantic)
- Standard JPEG compression (or fixed quantization) of 32×32 image to ~256–512 bytes
- Channel transmission (AWGN/fading)
- On-ground decoder: decompress → classify with standard CNN
- **Metric:** Classification accuracy vs. total bits transmitted

### Baseline 2: Non-Generative Learned JSCC (DeepJSCC-style)
- Autoencoder trained end-to-end: encoder → quantizer → AWGN/fading channel → decoder → task loss
- No generative prior; purely learned reconstruction
- Standard architecture from DeepJSCC literature or adaptive JSCC variants
- **Metric:** Serves as bridge between conventional and proposed generative method

### Baseline 3: Proposed Generative Semantic Method (Our Approach)
- Task-aware encoder + generative prior decoder as described above
- Adaptive rate control
- **Expected improvement:** Better accuracy at low bitrates and SNR due to generative prior

---

## 4. Performance Metrics

1. **Task Accuracy (%)** — Primary metric: classification accuracy on test set
   - Plot vs. SNR (0–20 dB)
   - Plot vs. transmitted bits (64–512 bits/image)

2. **Communication Cost** — Bits per image transmitted (normalized by bandwidth constraint)
   - Latency (if applicable): end-to-end inference time (ms)

3. **Semantic Fidelity** — Does reconstruction (if generated) preserve task-relevant information?
   - "Semantic PSNR": PSNR only on task-relevant regions or features, not entire image
   - Task-loss ratio: loss before/after generation

4. **Reliability/Confidence** — Confidence in prediction; hallucination detection
   - Prediction entropy or uncertainty estimate
   - Mismatch between transmitted feature and generated reconstruction

5. **Energy/Complexity** (if time permits)
   - Model size (UAV encoder parameters)
   - Inference time on embedded platform (e.g., Jetson Nano specs)

---

## 5. Experimental Plan

### Phase 1: Setup & Baselines (Weeks 1–2)
- Implement CIFAR-10 dataloading and standard CNN classifier
- Implement Baseline 1 (JPEG/quantization) and Baseline 2 (DeepJSCC)
- Establish channel simulation (AWGN + Rayleigh)

### Phase 2: Generative Architecture & Training (Weeks 2–3)
- Train lightweight VAE or load pretrained prior
- Implement task-aware semantic encoder
- Implement adaptive rate controller
- End-to-end training with task loss + reconstruction/generation loss

### Phase 3: Evaluation & Adaptation (Weeks 3–4)
- Benchmark all three methods across SNR and bitrate settings
- Analyze failure cases, hallucination risks
- Ablate generative prior (compare to non-generative variant)
- Test on different class distributions or channel types

### Phase 4: Report & Code (Week 5)
- Write final report with all required sections
- Clean up code, provide README with reproduction instructions
- Prepare final presentation slides

---

## 6. Key Research Questions & Extensions Beyond Grassucci et al.

1. **Task-Aware Encoding vs. Generic Generation:** Does a task-conditioned encoder (optimized for classification) outperform generic semantic encoding? How much does the generative prior need to "know" about the downstream task?

2. **Adaptive Rate Control:** How much does dynamic bitrate allocation (based on SNR/mission priority) improve over the fixed-rate generative baseline from Grassucci et al.? Can we achieve >85% accuracy on CIFAR-10 with <256 bits?

3. **Hallucination in Classification Context:** Unlike generic reconstruction quality, what happens when the generative model produces plausible images that lead to **wrong classification decisions**? Can we detect and mitigate these via confidence thresholding or fallback mechanisms?

4. **When Does Generation Fail (and When to Send More Bits):** Identify failure modes where increasing transmitted bits is preferable to relying solely on the generative prior. At what SNR/bitrate does the generative approach break down?

5. **UAV-Specific Constraints:** How do practical UAV factors (latency, energy, lightweight encoder constraints) affect the design choices compared to Grassucci et al.'s general setup?

---

## 7. Deliverables (by Dec 13, 2026)
- [ ] Final report (8–10 pages): problem formulation, method, results, analysis
- [ ] Python code (TensorFlow/PyTorch): encoder, channel, decoder, training loop, evaluation
- [ ] Plots: task accuracy vs. SNR, accuracy vs. bits, ablation results
- [ ] README: dataset download, model training, reproduction of main figures
- [ ] Citations: proper attribution to reference paper (Grassucci et al., 2026) and all external code/models

---

## 8. References & Resources

### Foundational Reference (Building Block for This Project)
- **[PRIMARY]** E. Grassucci, S. Barbarossa, and D. Comminiello, "Generative Semantic Communication: Diffusion Models Beyond Bit Recovery," *IEEE Transactions on Cognitive Communications and Networking*, vol. 12, pp. 8171–8185, 2026.
  - We will reuse/simplify their diffusion-based generative semantic framework with proper attribution.
  - Our extension: task-aware adaptive rate control for UAV-to-ground image classification.

### Supporting References
- **DeepJSCC:** Y. Yang et al., "Deep Joint Source-Channel Coding for Images," *IEEE Trans. Image Process.*, 2021.
- **Adaptive JSCC:** [Any adaptive rate-control papers the group identifies]
- **UAV Communications:** [Any UAV-specific channel modeling papers]

### Implementation Resources
- **Datasets:** CIFAR-10 (Keras/TensorFlow, direct import)
- **Generative Models:** Hugging Face diffusers, TensorFlow/Keras VAE examples
- **Channel Simulation:** NumPy/TensorFlow for AWGN and Rayleigh fading

---

**Next Steps:**
1. Confirm dataset, channel model, and baseline methods with advisor
2. Set up GitHub repository for version control
3. Begin Baseline 1 implementation (JPEG + CNN classifier)
4. Schedule weekly sync meetings with partner
