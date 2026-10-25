# WildTriage: Research-Validated Plan for an 8-Week Wildlife Monitoring Product

> **Update, 24 Oct 2026:** read [`AUDIT_2026-10-24.md`](AUDIT_2026-10-24.md) first. It supersedes this plan in several places:
> - the roadmap (§8 there);
> - the research framing (§3.3 there);
> - the priority of DepthSAM (now minimal P0) and the aerial module (now P2);
> - machine roles, now including the MacBook.
>
> Everything else here still stands.

## Context

**Course project.** The project is a computer vision course project (FAST NU, Semester 7). The team is **2 people** with **8 weeks**, starting Monday **19 Oct 2026** and ending **13 Dec 2026**. Shift the dates if your start date differs.

**Compute: three machines, used for different jobs.**

| Machine | GPU | CPU / RAM / free disk | Role in the project |
|---|---|---|---|
| **Lab PC**: Dell Precision 3680; Windows 11 Enterprise, domain-joined (university-managed) | **NVIDIA RTX 4000 Ada, 20 GB** (Ada; bf16) | i7-14700 (20 cores) / 32 GB / ~1.7 TB | **Heavy GPU work and the demo host.** Runs SAM 3 video, DepthSAM, BioCLIP 2 embedding extraction at scale, the SA-FARI evaluation and the final experiment runs. Stores all datasets. Worker queues: `gpu` + `gpu-heavy`. |
| **Teammate's PC** | **NVIDIA Quadro T1000, 4 GB** (Turing: fp16, **no bf16**) | Intel Core i7 (9th gen) / 32 GB / [A] | **Light GPU development.** Runs MegaDetector and BioCLIP 2 (fp16, small batches), adapter "golden" tests, data preparation (32 GB RAM) and Exp 1 runs. Worker queue: `gpu`. Not for SAM 3 or DepthSAM. |
| **MacBook Pro** (added 24 Oct): M1 Pro | Apple GPU via MPS (no CUDA); 32 GB unified memory | 32 GB / 512 GB disk | **Light models and development.** MegaDetector and BioCLIP 2 in fp32 on MPS; Exp 1 on cached embeddings. Worker queue: `gpu`. **Not for SAM 3 or DepthSAM** (CUDA only). |
| **Your laptop**: HP 15s-fq5xxx | Intel Iris Xe (no CUDA) | i5-1235U / 20 GB / ~180 GB | **Platform development.** Backend, frontend, Docker (installed) with CPU stub models, CI and docs. Exp 1 policy simulations on *cached embeddings* (NumPy on CPU). **No raw datasets.** |

**Design consequences** [R]:
- The design assumes **20 GB of VRAM at most**. It must not assume admin rights, Docker or WSL2 on the lab PC (see §0 and §12).
- **Job queues are routed by capability.** Light models run on any CUDA machine; heavy experts run on the lab PC only.
- **Each worker sets its own VRAM budget and dtype.** On the lab PC: 18 GB, bf16. On the T1000: 3.5 GB, fp16.
- **Only the lab PC handles raw media.** The other two machines receive cached embeddings and results (small files, shared via a team drive or git).

**Suggested ownership** (swap if you prefer):
- **Member A (ML & research)** works on the teammate's T1000 machine plus scheduled lab-PC sessions.
- **Member B (product & platform)** works on your laptop.

**Goal.** Turn three 2026 papers into one product that is coherent, evaluated, and defensible as research. It should not be three separate demos.

**How this plan was made.**
- The papers and their resources were checked on the web on **18 Oct 2026**: CVF Open Access, arXiv, GitHub, Hugging Face and LILA. Anything marked [V] links to the page it came from.
- The team decided to **swap Paper 3**, a CVPR *Workshop* paper, for an **ECCV 2026 main-conference paper** (see §2.4). This still needs instructor approval.
- A repository skeleton that follows §5, §6 and §11 lives next to this document (`wildtriage/`).

**Label legend** (used throughout):
- **[V]** verified from the linked source on 18 Oct 2026.
- **[R]** recommendation.
- **[A]** assumption that needs checking.
- **[U]** unresolved.

---

## 0. Before you start

1. **Move the repo out of OneDrive.** Syncing `node_modules`, virtual environments, datasets and `.git` through OneDrive causes file locks and slow syncs. Move `wildtriage/` to a folder OneDrive doesn't sync (e.g. `C:\dev\wildtriage` or the WSL2 ext4 filesystem), then run `git init`. Keep data under `D:\data` or WSL.
2. **Request gated access on day 1:** SAM 3 (`facebook/sam3`, `facebook/sam3.1`) and SA-FARI (`facebook/SA-FARI`) on Hugging Face.
3. **Get the swap of Paper 3 approved**, and confirm ECCV is on your approved venue list.
4. **Check the lab PC on day 1.** Ask IT before W2:
   - Can you install software? Specifically Docker Desktop, WSL2, a portable PostgreSQL and the CUDA-enabled PyTorch wheels.
   - Can jobs run overnight, and can you reach the PC remotely?
   - Is the disk persistent, or is it re-imaged or wiped at logout?
   - Is the GPU shared with other students?

   The skeleton runs **natively on Windows without admin rights** (uv + a portable PostgreSQL) and with Docker. Pick the path that matches the answers.
5. **Check the T1000 machine on day 1.**
   - Install the CUDA build of PyTorch.
   - Confirm `torch.cuda.get_arch_list()` includes `sm_75` (Turing).
   - Confirm `torch.cuda.is_bf16_supported()` is `False`. The code then falls back to fp16 automatically.
6. **Test the cloud fallback (molab) in W1.** Create a free [molab](https://molab.marimo.io) account and run one SAM 3 clip there. Record peak VRAM, seconds per clip, whether files and installed packages survive between sessions, how Hugging Face tokens are handled, and any quota you hit. Do this *before* you need it (see §12, risk 16).
7. **Read the README** in `wildtriage/` for how to run the skeleton.

---

## 1. Executive verdict

**Feasible in 8 weeks for 2 people, but only with a focused scope.** [R]

**What to build: a self-hosted camera-trap triage and verification platform** (working name *WildTriage*). It turns raw camera-trap images and videos into verified species records. Its main claim is measured: *how much human review it saves at a controlled error rate*.

| Category | Scope |
|---|---|
| **Fully supported, evaluated (P0)** | Camera-trap **images and videos**.<br>Detection (MegaDetector).<br>Species identification (BioCLIP 2, with per-site adaptation following Paper 3's recipe).<br>Video segmentation, tracking and per-clip counts (SAM 3, Paper 1's benchmark).<br>Calibrated confidence and a risk-controlled triage policy.<br>Human verification with an audit trail.<br>Activity summaries; CSV/COCO export. |
| **Experimental (P1)** | Camouflage refinement with DepthSAM (Paper 2).<br>Aerial *still-image* counting with MegaDetector-Overhead (OWL) or HerdNet on one public dataset.<br>An in-app site-adaptation loop.<br>Camtrap DP export and PDF/HTML reports. |
| **Out of scope** (future work) | Aerial video tracking.<br>Individual re-identification.<br>Statistically valid population **density/abundance** estimation: REM, CTDS and SCR need deployment geometry, the camera's detection zone and animal movement data, none of which we have.<br>Edge/embedded deployment.<br>Multi-tenant SaaS.<br>Training any large model from scratch. |

**Research contribution (one, testable).** *Review-budgeted, risk-controlled triage with site adaptation for camera-trap species recognition under temporal shift* (§3). It builds directly on Paper 3's open questions and Paper 1's video benchmark.

**What we will not claim:**
- state-of-the-art results;
- publication readiness;
- production-grade reliability;
- abundance estimates.

---

## 2. Paper validation

### 2.1 Comparison table

| | **P1: SA-FARI** | **P2: DepthSAM ("Beyond Appearance")** | **P3 (new): StreamTrap ("Lessons and Open Questions…")** | *P3 (dropped): Real-Wild-VLM* |
|---|---|---|---|---|
| **Venue** [V] | CVPR 2026 **main**, oral ([CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Wasmuht_The_SA-FARI_Dataset_Segment_Anything_in_Footage_of_Animals_for_CVPR_2026_paper.html), [oral listing](https://cvpr.thecvf.com/virtual/2026/oral/40322)) | CVPR 2026 **main**, poster, pp. 25830–25840 ([CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Han_Beyond_Appearance_Camouflaged_Object_Detection_via_Geometric_Structure_CVPR_2026_paper.html)) | **ECCV 2026 main** (arXiv comment and [project page](https://jeonso0907.github.io/stream-trap/); [arXiv 2603.20509](https://arxiv.org/abs/2603.20509)) | CVPR 2026 **Workshops**: 5th DataCV Workshop, 6 pp. ([CVF](https://openaccess.thecvf.com/content/CVPR2026W/DataCV/html/Deng_Real-Wild-VLM_Prompting_Large_Vision-Language_Models_for_Wildlife_Recognition_in_Camera-Trap_CVPRW_2026_paper.html)) |
| **Authors** | Wasmuht, Brookes, … Saenko, Surís (Conservation X Labs, Meta, Bristol et al.) | Han, Wu, Sun, Tang | Jeon, Tian, Wang, … Chao | Deng, Song, Bao, Ge |
| **Task** | Multi-animal detection, segmentation, tracking and species ID in camera-trap **video** (promptable concept segmentation) | Camouflaged object detection (class-agnostic binary segmentation, single image) | Camera-trap **species classification over time**, per site, under temporal shift | Zero-shot large-VLM recognition, including empty-clip rejection, on infrared camera-trap video |
| **Core contribution** [V] | A dataset: 11,609 videos, 99 species, 741 locations, 16,224 masklets, 942,702 boxes/masks.<br>SAM 3 benchmarks. | DepthSAM: a monocular-depth model adapted to camouflage via a sparse mixture-of-experts adapter (SMEA) and a geometric-semantic fusion module (GSFM).<br>Reported SOTA on COD10K (+3.0% S-α, +4.3% weighted-F over the prior best). | A benchmark of 546 LILA camera traps evaluated chronologically.<br>Shows BioCLIP 2 zero-shot fails at many sites and naive updates can hurt.<br>Recipe: LoRA + Balanced Softmax, plus logit calibration and WiSE weight interpolation. | Benchmark plus prompt engineering: explicit empty-rejection rules and morphology-guided instructions. |
| **Data** [V] | Train: 10,776 videos, 650 locations.<br>Test: 833 videos, 91 locations; cameras disjoint from train.<br>Annotated at 6 fps; RLE masks in a YTVIS/SA-Co format. | Standard COD benchmarks (COD10K, CAMO, NC4K, CHAMELEON). | 17 LILA datasets, about 3.2M images. | Private infrared videos from NE China, per the CamAgent paper by the same authors ([arXiv 2609.39112](https://arxiv.org/html/2609.39112)). |
| **Code / weights** [V] | Data: [HF `facebook/SA-FARI`](https://huggingface.co/datasets/facebook/SA-FARI) (gated; frames and videos in GCS bucket `cxl-public-camera-trap`).<br>Evaluator: [`sam3/scripts/eval/veval`](https://github.com/facebookresearch/sam3/tree/main/scripts/eval/veval).<br>Model: [SAM 3 / 3.1](https://github.com/facebookresearch/sam3) (HF `facebook/sam3`, `facebook/sam3.1`, gated; 848M params).<br>No SA-FARI-fine-tuned checkpoint found. | [hjy0518/DepthSAM](https://github.com/hjy0518/DepthSAM): README cites the paper, so it appears official (not confirmed). Code is Apache-2.0; weights and prediction maps on Google Drive/Baidu.<br>An independent [Jittor port](https://github.com/Lxlorange/Jittor-DepthSAM) reproduces close MAE values. | Code and data both marked **"Coming Soon"**.<br>The method is re-implementable: BioCLIP 2 ([HF](https://huggingface.co/imageomics/bioclip-2), MIT) and LILA data are public. | **None found** (no code, data, arXiv or supplement).<br>I could not read the PDF automatically: CVF blocks fetching it, and the browser tried to download it. |
| **License** [V] | SA-FARI: **CC-BY-NC 4.0** (non-commercial).<br>SAM 3: "SAM License" (royalty-free; trade-control and acceptable-use limits). | Code: Apache-2.0.<br>**Depends on Depth-Anything-V2-Large, which is CC-BY-NC-4.0** ([DA-V2](https://github.com/DepthAnything/Depth-Anything-V2)); DA-V2-Small is Apache-2.0. | Paper: arXiv license.<br>BioCLIP 2: MIT.<br>LILA licenses vary by dataset [A]. | n/a |
| **Compute** | SAM 3 inference in bf16 on the 20 GB RTX 4000 Ada [A: measure peak VRAM and fps in W1; split long clips into chunks if needed].<br>Fine-tuning is out of scope. | Training uses 4 GPUs (`torchrun --nproc_per_node=4`); inference on 1 GPU. | Embeddings plus LoRA or linear heads: fits easily on 1 GPU. | 7–8B VLM per clip: slow. |
| **Relevance** | **Direct**: the core video module and video benchmark. | **Complementary**: COD benchmarks are mostly animals, but there is no species ID, no camera-trap evaluation and no video. | **Direct**: species classification, deployment over time, and calibration. | Direct for night/empty clips, but the venue is ineligible and nothing is reproducible from released code. |
| **Key limitations** | NC license.<br>Weak on small masks: cgF1 25.3 vs 63.4 for large masks.<br>Only 4 continents.<br>**The public SAM 3 may already include SA-FARI train data**: the paper's "SAM 3 (SA-FARI)" variant was trained with it [U], so never evaluate SAM 3 on SA-FARI train. | Single-image only.<br>Not promptable.<br>NC dependency.<br>Repo has no pinned requirements.<br>Training code for a related part (CORAL) is "after publication". | No released code or benchmark yet: we must rebuild a smaller version ("StreamTrap-mini").<br>BioCLIP 2's pretraining may overlap LILA images [A]. | — |
| **Integration** | Video expert (SAM 3) plus the SA-FARI test set for evaluation. | Optional "refine mask" expert on hard crops (P1); evaluated offline. | Classifier adaptation and calibration core; drives the research contribution. | Its prompting recipe stays as a cited **P2** VLM expert. |

**Paper 1 benchmark numbers to reproduce** [V] (SA-FARI test set; Tables 3–5 re-read from the arXiv HTML on 18 Oct 2026. The HTML flattens the tables, so confirm the column grouping against the PDF in W1):
- **Species prompts** (cgF1 / pHOTA / TETA):

  | Model | cgF1 | pHOTA | TETA |
  |---|---|---|---|
  | SAM 3 | 14.0 | 48.5 | 39.6 |
  | SAM 3 (SA-FARI) | 39.0 | 63.1 | 52.1 |
  | SAM 3 FT | 46.9 | 68.1 | 58.7 |
  | LLMDet + SAM 3 tracker | 2.6 | 41.3 | 30.4 |
  | GLEE | −0.2 | 7.5 | 22.0 |

- **"animal" prompt** (IDF1 / HOTA):

  | Model | IDF1 | HOTA |
  |---|---|---|
  | MegaDetector + ByteTrack | 38.6 | 39.5 |
  | MegaDetector + OC-SORT | 33.9 | 45.5 |
  | MegaDetector + BoostSort++ | 47.2 | 38.3 |
  | SAM 3 (SA-FARI) | 71.1 | 64.4 |

- **Test subsets** (Table 5, SAM 3 FT, cgF1): large masks 63.4, small masks 25.3, multiple animals 45.9, challenging 36.8, night 44.1, all 46.9.
- **Night subset:** the paper says it "includes all the videos recorded at night" and reports it on the test split, so it most likely means 598 of the 833 test videos [V, one reading of the HTML; confirm in the PDF]. My first reading (5% of all videos) was probably wrong.

### 2.2 Can they integrate?

**Yes, as a cascade of experts around a calibrated triage policy** [R]:
- The cheap path (MegaDetector, then BioCLIP 2) handles most media.
- SAM 3 handles video clips that contain animals.
- DepthSAM is an optional refiner for low-confidence, small or camouflaged detections.
- The triage policy decides **auto-accept / review / empty** for each item.

The papers are not merged into one model. Each is a module with a typed adapter interface (§11).

### 2.3 Gaps in the selection and how we fill them (with reusable, non-2026 tools; these do not count toward the three papers)

| Gap | Fill with | Status |
|---|---|---|
| Image detection | [MegaDetector](https://github.com/agentmorris/MegaDetector): `pip install megadetector`; MDv1000-redwood recommended for accuracy, or MDv5a | [V] repo MIT. Per-weight licenses not stated; MDv5 is YOLOv5-based, and YOLOv5 code is AGPL [A]. |
| Species ID (supervised reference) | [SpeciesNet](https://github.com/google/cameratrapai): 2,000+ labels, EfficientNetV2-M, geofencing | [V] repo Apache-2.0. Trained on 65M images *including public LILA data*, so it is **not zero-shot on LILA**. |
| Aerial imagery (**none of the 3 papers**) | [MegaDetector-Overhead / OWL](https://github.com/microsoft/MegaDetector-Overhead) (point-based; OWL-C/T/D; legacy HerdNet), [Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) | [V] code MIT; **model/data release CC BY-NC-SA 4.0** |
| Aerial data | LILA: Izembek Lagoon Waterfowl (9,267 images, 521,270 boxes), NOAA Arctic Seals 2019 (~80k RGB+thermal, ~28k boxes).<br>Delplanque et al. multispecies (DOI 10.58119/ULG/MIRUU5).<br>[WAID](https://github.com/xiaohuicui/WAID) (~14k images, 6 species).<br>MMLA ([arXiv 2504.07744](https://arxiv.org/abs/2504.07744)) | [V] the LILA listing; Delplanque license [U] (ULiège server refused the connection; an HF mirror says CC-BY-NC-SA); WAID has no license file [U] |
| Tracking | SAM 3 tracker (P1 paper) plus ByteTrack baseline via `supervision` (MIT) | [V] SA-FARI baselines |
| Population / activity | Our code: independent events (30-min rule), relative abundance index (RAI) per 100 trap-days, diel activity, per-clip MaxN | Standard camera-trap ecology; *not* density estimation |
| Interchange standard | [Camtrap DP](https://camtrap-dp.tdwg.org/) (TDWG; deployments/media/observations; GBIF-publishable) | [V] |

**Other CVPR 2026 main-conference papers found** (I scanned all 4,042 titles by keyword; these are backup candidates only):
- Discover-Segment-Select, zero-shot camouflage ([arXiv 2602.19944](https://arxiv.org/abs/2602.19944)): DINOv2 + SAM 2-L + Qwen2.5-VL-7B; about 42 s per image on an RTX 3090; no code found.
- SDDF, open-vocabulary COD ([arXiv 2603.26109](https://arxiv.org/abs/2603.26109)).
- ProM3E (ecology embeddings).
- EthoCLIP (animal behaviour).
- RealBirdID.
- CountGD++ (open-world counting).
- Thermal-Det.
- A UAV multi-object-tracking benchmark ("Breaking Smooth-Motion Assumptions").

I found **no CVPR 2026 main-conference paper on aerial wildlife detection** in the title scan.

### 2.4 Final recommendation on papers

- **Keep P1 (SA-FARI)** as the anchor.
- **Keep P2 (DepthSAM)**, framed honestly as a *complementary robustness module*.
- **Swap P3 to the ECCV 2026 StreamTrap paper** (your decision).
- Cite Real-Wild-VLM as related work and as a P2 technique.
- **[U] Get instructor approval for the swap in W1.** Confirm ECCV is on your approved list.

### 2.5 How much of each paper was actually read

| Paper | What I read | What I did not |
|---|---|---|
| SA-FARI | arXiv HTML text and Tables 3–5, dataset card, project page, SAM 3 repo and license | The PDF itself; the supplementary material; the SA-FARI dataset files |
| DepthSAM | CVF abstract page, the official-looking GitHub README, an independent Jittor port's README | The PDF (CVF blocks fetching); no arXiv version found; the code was not run |
| StreamTrap (ECCV) | arXiv abstract and HTML text including limitations and open questions | Appendix details; no code or benchmark is released yet |
| Real-Wild-VLM (dropped) | Abstract; the authors' CamAgent paper's citation of it | The PDF (blocked) |

Everything marked [V] in this plan means "seen on the linked page", not "reproduced". Reproduction happens in W2–W5.

---

## 3. Research direction and novelty

### 3.1 Directions considered

| Direction | Gap evidence | 2-month feasibility (2 people, 1 GPU) | Verdict |
|---|---|---|---|
| **A. Review-budgeted, risk-controlled triage + site adaptation under temporal shift** | StreamTrap: zero-shot fails at many sites; naive updates underperform zero-shot at about 40% of traps. It calls predicting *when* adaptation helps "largely unexplored": its own confidence-based adapt-or-skip heuristics reach 76–77% balanced accuracy, against 87.8% for the oracle.<br>Related human-in-the-loop work routes by fixed confidence thresholds (see §3.3). | **High.** Works on cached embeddings; cheap training; public data. | **Chosen (P0)** |
| B. Calibrated routing between cheap and expensive *video* experts (MegaDetector + tracker vs SAM 3) | SA-FARI: SAM 3 far stronger than MegaDetector + trackers, but costly. | Medium: needs SAM 3 runs on the whole test set; SA-FARI test videos all contain animals, which limits the "skip" savings. | P1 extension (Exp 2b) |
| C. Camouflage-aware mask refinement | DepthSAM is not evaluated on camera traps; SA-FARI small/challenging subsets are weak. | Medium-low: DepthSAM is whole-image and not promptable, so applying it to crops is our adaptation. | P1 offline experiment |
| D. VLM empty-clip rejection (Real-Wild-VLM recipe) | The workshop paper reports VLMs biased toward non-empty answers. | Low: slow; needs empty-heavy labelled video. | P2 |

### 3.2 The chosen contribution (A)

**Problem.** A camera-trap deployment produces a stream of images. At each interval, an ecologist can review only a budget *b* of items. All other predictions are either auto-accepted or left unverified.

**What we propose and evaluate** [R]. A triage loop with four parts:
1. **Calibrated confidence.** Temperature scaling, fitted only on labels already reviewed at that site.
2. **Risk-controlled auto-acceptance.**
   - Split-conformal / conformal-risk-control thresholds that target error ≤ α among auto-accepted items.
   - **Adaptive conformal inference (ACI)** variants that track α online under temporal shift.
   - Conditioning by **day vs night/IR** ("Mondrian" groups), which ties to the lighting requirement.
3. **Budgeted review selection.** Uncertainty × class-balance quotas: review the items most likely to be wrong, with minority species protected.
4. **Site adaptation from verified labels.** StreamTrap recipe: Balanced Softmax with a linear head (or LoRA on BioCLIP 2), plus WiSE interpolation to protect zero-shot behaviour.

**What is and is not new** [R, after the related-work search in §3.3]. Every ingredient is existing work: temperature scaling, conformal risk control, Learn-then-Test, adaptive conformal inference, uncertainty-based and class-balanced selection, Balanced Softmax, WiSE. **We do not claim a new method.** What we propose is the *combination, evaluated under realistic conditions*:
- chronological streams at many real camera sites (StreamTrap-style), not a single i.i.d. split;
- a fixed review budget that is split between certification (random labels) and learning (selected labels);
- whether the auto-accept error target holds on *future* intervals, with day and night treated separately;
- how many selected labels close the gap to the all-labels oracle that StreamTrap uses.

The honest framing is an **empirical study with negative results allowed**. Whether it is publishable depends on results and on the literature check below.

**Novelty status** [U]. In about ten searches I found no camera-trap paper that sets review thresholds with a finite-sample guarantee (conformal risk control or Learn-then-Test). Absence from a few searches is not proof; W1 repeats the search properly (arXiv, Google Scholar, Semantic Scholar, ecology journals) and the claim stays unmade until then.

**Feasibility.** BioCLIP 2 embeddings for about 200k crops take minutes to roughly an hour on the RTX 4000 Ada [A]. Linear heads take seconds and LoRA takes hours. Every policy is simulated over the cached predictions.

**Data: "StreamTrap-mini".** Rebuilt from LILA following the StreamTrap protocol:
- Sites with ≥6 months and ≥1,000 animal images.
- Crops from **LILA's precomputed MegaDetector results** ([V] MDv5a/b, MDv1000-redwood, and repeat-detection-eliminated versions), with boxes enlarged by 50%.
- Chronological monthly intervals.
- Target **30 sites**: 10 dev and 20 test, from ≥3 LILA datasets.

**Design input from the simulator** (synthetic data, checks the mechanics only; *not a result*). `research/experiments/exp1_triage.py --synthetic` shows that **the size of the random calibration slice decides whether anything can be auto-accepted.**
- With α = 5% and δ = 0.1, Learn-then-Test needs at least ln δ / ln(1 − α) ≈ **45 error-free accepted calibration labels per group** before it can certify a threshold.
- Splitting calibration by day/night therefore costs coverage: 4% vs 12% in the smoke run.
- Exp 1 must therefore vary the calibration fraction ({0.2, 0.3, 0.5}). It must also test **pooling calibration across sites** against per-site calibration (pooling assumes sites are exchangeable).

**Academic value:** a clear empirical study with guarantees-vs-reality analysis and negative results allowed.

**Commercial value:** the headline product metric, "% of images auto-accepted at ≤α error" and "labels needed per site".

**Secondary (P0 reproduction, P1 analysis):**
- **Exp 2: SA-FARI video counting.** SAM 3 vs MegaDetector + ByteTrack per-clip MaxN error, by subset (night, small, multi-animal, challenging).
- **Exp 2b (P1):** clip-level routing (direction B).
- **Exp 3 (P1):** DepthSAM reproduction plus hard-crop refinement.
- **Exp 4 (P1):** aerial counting with OWL/HerdNet.

### 3.3 Related work found (searched 18 Oct 2026)

| Work | What it does | How it relates | Read how |
|---|---|---|---|
| Dussert et al., *Being confident in confidence scores* (Remote Sensing in Ecology and Conservation 2025; [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.11.10.566512)) | Calibration of camera-trap classifiers at image and sequence level. Averaging **logits** over a sequence before the softmax was the most effective; temperature scaling suggested. | Direct baseline for our calibration and sequence aggregation. **Changed our code** (see below). | Search summary only; the page blocked fetching |
| Huebner et al. 2024, [*Human supervision is key…*](https://theoryandpractice.citizenscienceassociation.org/articles/10.5334/cstp.752) (Snapshot Safari) | Human-in-the-loop with confidence-based routing: error 34.89% (AI alone) vs 8.73% (with volunteers) on a 5,314-point Serengeti gold standard. | The closest *deployed* version of confidence routing. Heuristic thresholds, no guarantee, one site and one time window. | Page read |
| Norouzzadeh et al. 2021, [deep active learning](https://arxiv.org/html/1910.09716v1) (Methods in Ecology and Evolution) | Active selection of images to label; reports matching full-label accuracy on a 3.2M-image set with 14,100 labels (>99.5% fewer). | Baseline for our selection strategies. Single dataset. | Search summary |
| Miao et al. 2021, [iterative human + automated identification](https://www.nature.com/articles/s42256-021-00393-0) (Nature Machine Intelligence) | Model picks low-confidence items for humans, accepts the rest. | Same idea without risk control. | Search summary |
| Fermanian, Hebiri, Salmon, [class-conditional conformal prediction for multiple inputs](https://arxiv.org/abs/2507.07150) (NeurIPS 2025) | Conformal sets that combine several photos of the same individual, with class-conditional coverage. Evaluated on Pl@ntNet (plants). | Directly relevant to bursts and sequences. A **P2 baseline** for our sequence handling. | Abstract |
| Cohen et al. 2026, [Bayesian fusion of AI confidences and manual annotations](https://arxiv.org/abs/2605.13660) (stat.AP preprint) | Calibrates AI confidences with annotations for ecological inference (deer body condition). Annotations chosen *at random*; active selection is named as future work. | Shows the "which images to annotate" question is open in ecology statistics. Different task. | Abstract + most of the text |
| StreamTrap, [arXiv 2603.20509](https://arxiv.org/abs/2603.20509) (ECCV 2026) | Adapt-or-skip heuristics (random, mean max-softmax, feature prototypes) reach 77.4 / 75.9 / 76.4% balanced accuracy vs 87.8% oracle; images treated as independent (no sequence modelling). | Our baseline and the open question we address. **We must differentiate from its MSP heuristic.** | Text read, last section partly |
| Hockerts et al. 2026, [label uncertainty](https://arxiv.org/abs/2608.30789) | Effect of volunteer-label noise on classifiers. No conformal or calibration component. | Context for reviewer noise (our ablation of 0% vs 2%). | Page read |
| Conformal prediction for acoustic indices, [ScienceDirect 2026](https://www.sciencedirect.com/science/article/pii/S266597272600276X); Poisot, species distribution models ([EcoEvoRxiv](https://ecoevorxiv.org/repository/object/8041/download/15118/)) | Conformal methods applied to other ecological monitoring data. | Shows conformal methods are entering ecology; none for camera-trap review. | Titles and snippets only (page blocked) |

**What changed in the project because of this search:**
1. **Sequence aggregation** now defaults to averaging **logits** before the softmax (Dussert et al.), with probability-mean and max kept as ablations. `ml/wildtriage_ml/pipeline/aggregate.py` was updated and tested.
2. **The novelty claim is now modest** (see above): a combination and an evaluation, not a new method.
3. **StreamTrap's MSP adapt-or-skip rule is an explicit baseline** in Exp 1, besides random and least-confidence review.
4. **Conformal sets for bursts** (Fermanian et al.) are a P2 baseline, not P0.
5. **Cite:** Dussert 2025, Huebner 2024, Norouzzadeh 2021, Miao 2021, Angelopoulos et al. (conformal risk control; Learn-then-Test), Gibbs and Candès (ACI), Ren et al. (Balanced Softmax), Wortsman et al. (WiSE-FT).

---

## 4. Product specification

**Target users:**
- conservation NGOs and protected-area teams running camera-trap grids;
- university ecology labs;
- environmental consultancies doing biodiversity monitoring for impact assessments [A: market hypothesis].

**Problem.** Manual review of camera-trap media is the bottleneck. Generic AI labels have unknown, site-dependent error and give no indication of which labels to trust.

**Core workflows:**
1. Create a project with a region and species list.
2. Register deployments (sites/cameras and their dates).
3. Upload a card dump (folder or zip of images/videos).
4. Processing runs asynchronously with live progress.
5. The system triages each item: auto-accept, review queue or empty.
6. Reviewers verify the queue by keyboard.
7. *(P1)* Adapt the site model from verified labels and re-triage.
8. Analytics and export (CSV, COCO, Camtrap DP P1).

**Supported inputs** (P0 unless marked):

| Input | Details |
|---|---|
| Images | JPG/PNG; EXIF timestamp. |
| Videos | MP4/MOV/AVI (H.264/H.265 via ffmpeg); ≤10 min and ≤2 GB each (configurable). |
| Aerial stills | JPG/TIFF ≤ 100 MP (P1), tiled. |
| Annotation import | COCO / COCO-Camera-Traps JSON, the MegaDetector batch JSON, SA-FARI (YTVIS-like) JSON, CSV species lists. |
| Modality | **Declared per dataset** (camera_trap \| aerial); no automatic modality detection (P2). |

**Feature status:**

| Status | Features |
|---|---|
| **Implemented (P0)** | Ingest, validation and SHA-256 dedup.<br>Quality flags (night/IR, blur, darkness).<br>MegaDetector; BioCLIP 2 species top-5 with calibrated confidence.<br>Sequence/clip aggregation.<br>SAM 3 video masklets/tracks and MaxN.<br>Triage policy; review queue; audit log.<br>Analytics (counts, diel activity, detections per deployment); CSV/COCO export.<br>Model and evaluation registry. |
| **Experimental (P1)** | DepthSAM refine; aerial counting; site-adaptation loop; Camtrap DP; RAI/independent events; reports; perceptual-hash near-duplicate detection. |
| **Future (P2)** | VLM adjudication, aerial video, re-ID, density estimation, edge, OCR timestamps, map/PostGIS, SaaS. |

**Handling bad or uncertain inputs:**
- **Unsupported or corrupt files** are rejected per file with error codes (`UNSUPPORTED_TYPE`, `DECODE_FAILED`, `TOO_LARGE`, `DUPLICATE`).
- **Low-quality media is never auto-accepted.** Low-quality means very dark, blurred, or with a tiny bounding box.
- **Low confidence or disagreement between models** sends the item to review, with explicit reason codes.
- **Humans detected:** frames flagged; blurred in exports; hidden from viewers.
- **Endangered-species coordinates** are obfuscated (rounded) except for admins.

### 4b. Market, licensing and commercial readiness

**Competing tools** (checked 18 Oct 2026 against the sources linked; "not verified" means not on the pages I read, not proof of absence):

| Tool | What it is | What I verified | Not verified |
|---|---|---|---|
| [Wildlife Insights](https://www.worldwildlife.org/our-work/wildlife/wildlife-technology/wildlife-insights/) | Hosted platform; AI sorts uploads into people, animals, empty, then species with SpeciesNet | Human review is built in (accept the AI label or pick from a list, thumbnail identification, burst mode); about 200 million human-verified images reported; access is free but gated, with approval needed for non-commercial users with fewer than 50 cameras ([Google Research](https://research.google/blog/where-wild-things-roam-identifying-wildlife-with-speciesnet/), [WildLabs](https://wildlabs.net/en/inventory/products/wildlife-insights)) | Video support; whether it uses calibrated or risk-controlled thresholds |
| [AddaxAI](https://github.com/PetervanLunteren/AddaxAI) (formerly EcoAssist) | Desktop app: detection, species recognition, verification, dashboards, exports | Open source, MIT, Windows/macOS/Linux; docs list "Confidence and verification", "Check the labels" and Timelapse-export pages | Video; export formats; calibration |
| Timelapse | Desktop image reviewer that shows MegaDetector and classifier output | Human review with ML output (per [Morris's survey](https://agentmorris.github.io/camera-trap-ml-survey/)) | Licensing; video |
| [Zamba / Zamba Cloud](https://zamba.drivendata.org) | Open-source package plus a no-code cloud service for **video** and image classification | Video support; models for 32 African and 11 European species plus blanks; reported 82% top-1 (African) and 79% (European) | Pricing; review workflow |
| TrapTagger, Agouti, TRAPPER, Camelot, WildTrax | Web or desktop platforms with MegaDetector-based filtering and annotation | TrapTagger and TRAPPER are open source (per the survey) | Features beyond one-line descriptions |

**What the competing tools do not appear to offer** [R, from the pages above]: none of the pages I read describe *calibrated* confidence, *error-rate-controlled* auto-acceptance, or *per-site adaptation from verifications* as a measured feature. That is the gap our product bets on. It may exist in places I did not read, so the plan makes it a testable claim, not a marketing line.

**Differentiation** [R]. Video-native segmentation and counting, **calibrated, error-controlled triage with measured review savings**, and per-site adaptation from verifications. All of it runs self-hosted on one GPU workstation, with no field data leaving the organization.

**Honest limits of that pitch:** Wildlife Insights is free for most research users and already has human review; AddaxAI is free, local and open source. A researcher will ask why they should switch. The answer has to be a measured saving in review time or error, not interface polish.

**Deployment options:**
1. A self-hosted Docker Compose stack on a GPU workstation; works offline at field stations.
2. A lab server.
3. *(Later)* managed hosting.

**Licensing split: two model manifests** [R].
- **Research edition** may use NC components: anything trained on SA-FARI, DA-V2-Large inside DepthSAM, and the OWL/HerdNet weights.
- **Commercial edition** uses license-clean components only:
  - BioCLIP 2 (MIT), SpeciesNet (Apache), DA-V2-Small, RF-DETR N/S/M/L (Apache);
  - SAM 3 (SAM License: legal review needed);
  - MegaDetector (weight/AGPL status [U]).
- Every `model_versions` row stores its license; a license-check script blocks NC models when commercial mode is on.

**Demonstrable benefit (only if Exp 1 shows it):** "X% of items auto-accepted at ≤5% realized error; Y labels per site to reach within 2 points of oracle accuracy." Report confidence intervals and failure sites.

**Not deliverable in 2 months:** customer validation, a security audit, SLAs, a hosted service, or legal clearance of licenses.

---

## 5. System architecture

```
 Browser: React SPA (Projects · Data/Upload · Jobs · Media & Clip viewer · Review · Analytics · Models/Eval · Export)
        │  REST/JSON + SSE (job progress)                        ▲ signed media URLs
        ▼                                                        │
 ┌──────────────────────────────┐   files   ┌────────────────────┴─────┐
 │ api: FastAPI                 │──────────▶│ Media store              │
 │ auth/RBAC · validation ·     │           │ local volume (P0) behind │
 │ CRUD · triage/review · audit │           │ StorageBackend; S3 (P1)  │
 │ analytics · exports          │           └────────────▲─────────────┘
 └──────┬───────────────────────┘                        │
        │ SQLAlchemy (CRUD + INSERT job row = enqueue)    │
 ┌──────▼──────────────────────┐   claim: SELECT … FOR UPDATE SKIP LOCKED
 │ PostgreSQL 16               │◀──────────────┬───────────┴───────────────────────────┐
 │ metadata · predictions ·    │               │                                        │
 │ reviews · audit · evals ·   │  ┌────────────┴──────────────────┐  ┌─────────────────┴──────────────────┐
 │ jobs table = the job queue  │  │ worker (queue "cpu"): ingest,  │  │ GPU workers, one process per GPU,  │
 └─────────────────────────────┘  │ ffprobe/ffmpeg, thumbnails,    │  │ each with its own VRAM budget:     │
                                  │ quality, dedup, exports        │  │ "gpu": MegaDetector · BioCLIP 2    │
                                  └────────────────────────────────┘  │   (+site head, calibrator) · Triage│
                                                                      │   → lab PC or T1000 (fp16, 3.5 GB) │
                                                                      │ "gpu-heavy": SAM 3 video, [P1]     │
                                                                      │   DepthSAM (own env), OWL/HerdNet  │
                                                                      │   → lab PC only (bf16, 18 GB)      │
                                                                      └────────────────────────────────────┘
 Offline research workspace: same `ml/` package → data prep → cached embeddings → experiments
                             → evaluation_runs rows (DB) + figures; git commit recorded per run
```

**Deployment shapes** (same code):
- **(a) Native Windows on the lab PC.** No admin rights needed: `uv` virtual environments, a portable PostgreSQL started with `pg_ctl`, the API and workers as plain processes, and the frontend served by Vite in development or as static files.
- **(b) Docker Compose** on any machine that has Docker (your laptop has it). The GPU worker is optional (the `gpu` profile).

**Machine roles at run time:**

| Machine | Runs | Settings (`.env`) |
|---|---|---|
| Lab PC (demo host) | Postgres, API, frontend, `cpu` worker, one GPU worker on `gpu,gpu-heavy` | `WT_MODEL_BACKEND=real`, `WT_GPU_VRAM_BUDGET_GB=18` |
| T1000 PC | A GPU worker on `gpu` only, pointed at the lab PC's database when on the campus network; otherwise its own local stack for development | `WT_MODEL_BACKEND=real`, `WT_GPU_VRAM_BUDGET_GB=3.5` |
| MacBook | A GPU worker on `gpu` (MPS), or its own local stack for development | `WT_MODEL_BACKEND=real`, `WT_DEVICE=mps`, `WT_GPU_VRAM_BUDGET_GB=12` |
| Laptop | The full stack in Docker or natively, using stub models | `WT_MODEL_BACKEND=stub` |

Media files must be reachable by the worker that processes them. Remote workers therefore only run when they share storage with the API (same machine, or a network share). Otherwise each machine runs its own local stack. [R] For the demo, run everything on the lab PC.

### Inference pipeline (per media item)

1. **Validate.** Check magic bytes; read metadata (EXIF, ffprobe); compute SHA-256 to catch duplicates; compute quality features:
   - night/IR flag: low chroma, mean |R−G|+|G−B| below a threshold;
   - blur: variance of the Laplacian;
   - brightness.
2. **Image path.**
   - Run MegaDetector (confidence ≥ τ_det).
   - Crop each detection with the box enlarged 50%.
   - Run BioCLIP 2 against the project's species prompts, or the site head if one exists.
   - Apply temperature scaling.
   - Aggregate over the camera's burst/sequence (LILA's `seq_id` or a time gap under 30 s).
3. **Video path.**
   - Run MegaDetector on frames sampled at 1 fps. If there is no animal, mark the clip as an empty candidate (this saves GPU).
   - Otherwise run SAM 3 at 6 fps (matches SA-FARI) with the prompt "animal" plus the top-k species names from the classifier.
   - Output masklets → tracks → per-track species vote → **MaxN** (the most animals visible in any one frame).
4. **Triage policy** (versioned). Outputs:
   - `auto_accept` / `review(priority, reasons[])` / `auto_empty`;
   - α and the budget come from project settings.
5. **Review** writes `verification_decisions`, which are append-only and audited.
6. **Adaptation** (P1 in-app; P0 in research) creates a new `model_version` from verified labels, then re-triages.

**Dependency isolation** [R]:
- SAM 3 needs Python ≥3.12 and PyTorch ≥2.7 / CUDA ≥12.6 [V].
- DepthSAM ships old SAM-v1 code with unpinned requirements [V].
- So DepthSAM gets its **own virtual environment (or image)**, and a worker started only for `gpu-heavy` camouflage jobs.
- If MegaDetector's dependencies conflict, it gets its own environment too. Each job row names its queue, and each worker claims only its own queues.

**VRAM policy** (20 GB lab PC, 4 GB T1000) [R]:
- Use **bf16 where the GPU supports it** (Ada); fall back to **fp16** (Turing T1000, which has no bf16) or fp32 on CPU. `wildtriage_ml.registry.pick_dtype` picks automatically.
- Each GPU worker's `ModelRegistry` has a **VRAM budget** (18 GB on the lab PC, 3.5 GB on the T1000). It evicts the least-recently-used model before loading another, and refuses models that can never fit.
- **What fits on the T1000** [A: measure in W1]: MegaDetector (about 1.5 GB) and BioCLIP 2 ViT-L/14 in fp16 (about 2 GB, small batches), one at a time if needed. SAM 3, DepthSAM and VLMs run on the lab PC only (`gpu-heavy`).
- Small models stay resident: MegaDetector and BioCLIP 2 together are about 3 GB [A].
- SAM 3 loads on demand.
- Run DepthSAM and any VLM **only in their own job batches, never resident at the same time as SAM 3.**
- Split long videos into chunks of ≤ 30 s at 6 fps.
- Log `torch.cuda.max_memory_allocated()` on every job, so the efficiency table comes for free.

**Why a Postgres queue instead of Celery + Redis** [R]:
- Redis has no official Windows build, and the lab PC may not allow Docker or WSL2.
- Celery's Windows support is limited.
- A `jobs` table claimed with `SELECT … FOR UPDATE SKIP LOCKED` removes a service. It also makes "create job + enqueue" a single transaction, and works natively and in Docker alike.
- Volume is low: one GPU worker, thousands of jobs rather than millions.
- If you later need more scale, the same interface can be backed by a Postgres task-queue library such as Procrastinate, or by Celery.

---

## 6. Technology stack

| Layer | Choice | Why | Simpler alternative |
|---|---|---|---|
| Backend | Python 3.12, **FastAPI**, Pydantic v2, Uvicorn | Same language as the ML code; async; OpenAPI generates the TypeScript client | Flask |
| ORM / migrations | SQLAlchemy 2.0, Alembic | Typed models; repeatable migrations | — |
| Database | **PostgreSQL 16** (JSONB for model outputs and config) | Concurrency with workers; JSONB; PostGIS later | SQLite (single-user demos only) |
| Jobs | **Postgres-backed queue** (`jobs` table + `FOR UPDATE SKIP LOCKED`, ~200 lines in `backend/app/workers/`): named queues, retries with backoff, heartbeat, progress columns; one long-lived GPU worker process loads models once and never forks CUDA | No Redis; works natively on Windows and in Docker; enqueue is in the same transaction as the job row | Procrastinate (Postgres-based) or Celery + Redis if scale demands it |
| Storage | Local volume behind a `StorageBackend` interface; S3-compatible adapter (boto3) in P1 | Zero infrastructure; swappable later | — ([A] check MinIO's current community-edition status before adopting it) |
| Media | ffmpeg/ffprobe, PyAV, OpenCV, Pillow, pycocotools (RLE masks) | Standard | — |
| ML | PyTorch ≥2.7 (CUDA 12.6+), `megadetector`, `open_clip` (BioCLIP 2), `sam3`, `peft` (LoRA), scikit-learn, numpy/pandas/pyarrow | Reuse first | — |
| Evaluation | pycocotools, TrackEval (HOTA/IDF1), the sam3 `veval` (cgF1/pHOTA/TETA), `py-sod-metrics` (S-α, F-β^w, E-φ, MAE), our own unit-tested ECE and conformal code | Matches each paper's protocol | — |
| Experiment tracking | MLflow (local file store) + YAML configs + `evaluation_runs` table | Free; offline | CSV + git |
| Data versioning | Parquet manifests (path, sha256, site, timestamp, seq_id, split) committed to git; DVC optional | Reproducible splits without heavy tooling | — |
| Frontend | React 19 + Vite + TypeScript; React Router; TanStack Query/Table; Tailwind (v4) + shadcn/ui components; ECharts; `<video>` + canvas overlays; `openapi-typescript` | Polished, typed, fast; fewer moving parts than a framework | Plain CSS instead of Tailwind |
| Auth | Session cookie (HTTP-only) + argon2 password hashes; roles admin / reviewer / viewer | Simple and sufficient | Basic auth (demo only) |
| Tests | pytest (+httpx, pytest-cov), Vitest + React Testing Library, Playwright | — | — |
| Operations | Two run paths: **native** (uv + portable PostgreSQL on the managed lab PC) and **Docker Compose** (NVIDIA Container Toolkit where allowed); `pydantic-settings` + `.env`; structlog JSON logs; a jobs dashboard in the app; `/health` (P0) and Prometheus `/metrics` (P1); nightly `pg_dump` + media copy to a second disk; GitHub Actions CI (CPU, stub adapters) | Works whether or not IT allows Docker | — |

---

## 7. Dataset and model strategy

### Models (reuse, not training)

| Role | Model / identifier | How we use it | License [V] |
|---|---|---|---|
| Detector | MegaDetector **MDv1000-redwood** / MDv5a (`pip install megadetector`; `python -m megadetector.detection.run_detector_batch MDV5A <dir> <out.json> --recursive --output_relative_filenames`) | As-is. Use LILA's precomputed results for experiments. | MIT repo; weights [U] |
| Species ID | **BioCLIP 2** (`open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')`) | Zero-shot with a taxonomic-name prompt per project species, plus **our** per-site heads | MIT (card) |
| Species reference | **SpeciesNet** (`pip install speciesnet`; `python -m speciesnet.scripts.run_model --folders <dir> --predictions_json out.json --country <ISO3>`) | Comparison baseline only; LILA leakage caveat | Apache-2.0 |
| Video segmentation / tracking | **SAM 3** / **SAM 3.1** (`git clone https://github.com/facebookresearch/sam3 && pip install -e .`; `hf auth login`; `build_sam3_video_predictor()`) | As-is with text prompts. Fine-tuning is P2. | SAM License; gated |
| Camouflage (P1) | **DepthSAM** (`git clone https://github.com/hjy0518/DepthSAM`; weights per README; `python test.py`) | Inference only | Apache code + NC DA-V2-L |
| Aerial (P1) | **OWL-C** (CPU-capable) / **OWL-D**; legacy HerdNet (MegaDetector-Overhead; `uv sync --no-default-groups --group gpu`) | As-is; tiled inference | MIT code; CC BY-NC-SA weights |
| VLM (P2) | Qwen3-VL-4B-Instruct, or 8B quantized to 4-bit (8B in bf16 needs ~17 GB, too tight next to other models on 20 GB) [A] (transformers ≥4.57) | Real-Wild-VLM-style prompts | Apache-2.0 |
| Detector fine-tune fallback (P2) | RF-DETR N/S/M/L | Only if the aerial module needs boxes | Apache-2.0 |

**Our own work, separate from what we reuse:**
- the triage policy (calibration, conformal/ACI, selection);
- per-site adaptation code and Balanced Softmax;
- sequence/clip aggregation, MaxN counting and quality features;
- every pipeline, backend, frontend and evaluation harness;
- StreamTrap-mini construction;
- all experiments.

### Datasets

| Dataset | Used for | Access [V] | License |
|---|---|---|---|
| **SA-FARI test** (833 videos, 1,083 masklets, 62,341 annotations, 83 species, 91 locations) | Exp 2 + video pipeline QA | HF gated annotations (`sa_fari_test(_ext).json`); frames via `gcloud storage cp -r gs://cxl-public-camera-trap/sa_fari/test/JPEGImages_6fps <dst>` ([A] check the exact path) | CC-BY-NC 4.0 |
| SA-FARI train | Qualitative development only. **Never** for SAM 3 evaluation (possible training overlap). | Same | CC-BY-NC 4.0 |
| **LILA** camera-trap datasets (Snapshot Serengeti, Caltech CT, Idaho, NACTI, Orinoquía, Island Conservation, …) + LILA precomputed MegaDetector JSON | Exp 1 StreamTrap-mini (~100–200k crops) | Public cloud buckets; per-dataset metadata CSV | Per dataset [A: many CDLA-Permissive; record each] |
| COD10K / CAMO / NC4K / CHAMELEON | Exp 3 reproduction | Links in the DepthSAM README | Research use [A] |
| MoCA-Mask (87 videos, 22,939 frames) | P2 video-camouflage stress test | [project page](https://xueliancheng.github.io/SLT-Net-project) | [U] |
| Aerial: LILA Izembek / NOAA seals; Delplanque; WAID | Exp 4 (P1) | See §2.3 | NC/unclear: research only |
| **Team "hard set"** (~300 items: night, blur, camouflage, empty) curated from the above | Robustness, demo, UI tests | Ours | Inherits source licenses |

### Preprocessing and safeguards

- **Manifests** with SHA-256 for every file.
- **Exact-duplicate removal**; perceptual-hash near-duplicates in P1.
- **Sequence grouping:** never split a burst across calibration and test.
- **Site-disjoint dev/test splits.**
- **Chronological intervals** within each site; calibration and thresholds use **only past intervals**.
- **One final test-set run**, from a tagged commit.

**Training we do:** per-site linear heads or LoRA on BioCLIP 2, temperature scaling, policy thresholds. **Not trained:** detectors, SAM 3, DepthSAM.

---

## 8. Eight-week roadmap

**Roles:**
- **A = ML & research lead:** data, models, experiments, report.
- **B = Product & platform lead:** backend, database, frontend, operations, tests.

**Shared work:** Friday integration and demo; Monday 30-minute planning; a shared GPU calendar, with A's experiments running at night and B's dev work using CPU stub adapters.

**Critical path:**
1. Access requests (SAM 3, SA-FARI) → W3 video pipeline.
2. StreamTrap-mini prep → Exp 1 (W3–5) → frozen final test run (W7).
3. Schema + jobs (W1–2) → review UI (W4) → adaptation loop (W5).

| Week | Deliverables (acceptance criteria) | A | B | Risks / fallback |
|---|---|---|---|---|
| **W1** 19–25 Oct | Stack boots; access granted; research protocol frozen. **Accept:** the stack starts (native path on the lab PC; Docker on the laptop); uploading 20 images creates DB rows plus stored files; SAM 3 produces masklets on 3 SA-FARI clips; a 2-page protocol lists RQs, splits and metrics. | Request SAM 3 and SA-FARI access **on day 1**. Read all 3 papers in full (resolve the [U] items). Literature check for novelty. Confirm what the lab PC allows (§0). Install CUDA PyTorch with uv. Smoke-test MegaDetector, BioCLIP 2 and SAM 3. Choose LILA sites (criteria in §3). Measure SAM 3 peak VRAM and fps on the 20 GB card. Check the T1000 (sm_75, fp16) runs MegaDetector and BioCLIP 2. Run the molab fallback test (§0 item 6). | Repo, Compose + native run scripts, schema v1 + Alembic, auth, project/dataset/deployment CRUD, upload + validation + SHA-256 dedup, storage abstraction, CI. | Access delayed → start with LILA work; SAM 2.1 fallback. |
| **W2** 26 Oct–1 Nov | **Walking skeleton (images).** **Accept:** upload → async job → detections + species + calibrated scores in DB → visible in the UI grid and viewer. SA-FARI reproduction harness runs on the test set. | Adapters (MegaDetector, BioCLIP 2) behind protocols. StreamTrap-mini download and crops. Cache embeddings. sam3 `veval` harness. Reproduce the "animal" prompt and MegaDetector + ByteTrack rows. | Postgres job queue + workers, with progress over SSE (polling fallback). Prediction persistence. Media browser + image viewer (boxes, top-5). | Reproduction gap → document the protocol difference; do not tune toward the paper's numbers. |
| **W3** 2–8 Nov | Exp 1 baselines + video in the app. **Accept:** table of zero-shot / naive fine-tune / recipe / oracle on dev sites. ECE by interval. One uploaded video shows tracked overlays and MaxN. | Exp 1 baselines; temperature scaling; split-conformal and ACI. SA-FARI species-prompt runs (overnight, if the lab PC allows unattended runs). | ffprobe/ffmpeg ingest; SAM 3 video job (chunked, within the VRAM budget); tracks table; clip viewer (server-rendered mask-overlay MP4 + canvas boxes + track timeline). | SAM 3 too slow → shorter clips, fewer prompts, 3 fps. |
| **W4** 9–15 Nov | **MVP** (images + video + triage + review + analytics). **Checkpoint:** apply the scope ladder if behind. **Accept:** full loop demo; review decisions audited; Exp 1 selection strategies run on dev sites. | Selection strategies (random, least-confidence, margin, entropy, k-center, ours); budget loop simulator; package the policy as a `model_version`. | Review queue (keyboard, timer, reason codes); triage badges; audit log; analytics v1 (counts, diel). | Behind → cut aerial and in-app DepthSAM now. |
| **W5** 16–22 Nov | Research depth + adaptation loop. **Accept:** ablation table on dev sites. "Adapt site" button creates a new model version and re-triages. | Ablations (§9). DepthSAM setup + COD10K reproduction (P1). Freeze the Exp 1 protocol before test. | Adaptation job; models and evaluations pages (risk–coverage, reliability diagrams); CSV/COCO export. | DepthSAM fails → offline-only or drop. |
| **W6** 23–29 Nov | Robustness, efficiency, hardening. **Accept:** robustness and efficiency tables. CI green. A batch of 1,000 images + 50 videos finishes with 0 unhandled errors. | Day/night and synthetic blur/darkness tests. Cross-dataset policy transfer. Latency/VRAM profiling. Aerial OWL on one dataset (P1). Exp 2 subset analysis. | Playwright E2E; API/DB tests; load test; error UX; Camtrap DP (P1); security pass (RBAC, upload limits, person blur, coordinate obfuscation). | Aerial → P2 if not working by Wed. |
| **W7** 30 Nov–6 Dec | **Feature freeze (Wed 2 Dec).** Final results. **Accept:** final test-set numbers from a tagged commit, with bootstrap CIs. Report draft. Review-effort mini-study done. | Single final test run. Statistics. Figures. Mini-study: 2–4 reviewers × 200 items, risk-ordered vs random order, measuring time per item and accuracy. | UI polish; demo seed script; backups; model/dataset cards in the app; README and deployment docs. | Only bug fixes after the freeze. |
| **W8** 7–13 Dec | Report, demo, hand-in. **Accept:** §13 checklist passes; demo rehearsed twice; backup video recorded. | Technical report (8-page paper style) + reproducibility appendix. | Demo script; slides; final deployment from a clean clone. | 2 buffer days. |

**MVP definition:** the W4 end state.

**Scope ladder** (cut in this order if behind):
1. aerial module;
2. in-app DepthSAM;
3. LoRA (keep linear heads);
4. Camtrap DP;
5. SSE (fall back to polling);
6. Exp 2b.

**Never cut:** Exp 1, SA-FARI reproduction, the review loop, tests, docs.

**Resources:**
- **Lab PC** (RTX 4000 Ada 20 GB, i7-14700, 32 GB, ~1.7 TB free): heavy GPU work and all raw datasets (≥300 GB, outside OneDrive). Native run path if Docker/WSL2 is not allowed.
- **Teammate's PC** (Quadro T1000 4 GB, i7 9th gen, 32 GB): light GPU development; MegaDetector/BioCLIP 2 runs; data prep.
- **Your laptop** (i5-1235U, 20 GB, no CUDA, ~180 GB free, Docker): platform development with stubs; Exp 1 simulations on cached embeddings.
- **molab account** (free, preview) as a tested fallback for GPU-heavy batch jobs; Colab or Kaggle as a second option.
- **A shared team drive** for cached embeddings and result files. BioCLIP 2 embeddings for ~200k crops in fp16 come to ~300 MB [A]. Manifests, configs and summary results go in git.
- HF account (gated access for SAM 3 and SA-FARI);
- `gcloud` CLI;
- GitHub (private repo);
- **cloud cost $0**.

---

## 9. Research and evaluation plan

**Research questions:**
- **RQ1 (calibration under drift).** Does post-hoc calibration fitted on past intervals stay calibrated on future intervals, per site and for day vs night?
- **RQ2 (risk control).** Does conformal / ACI auto-acceptance keep realized error ≤ α on future intervals, and at what coverage?
- **RQ3 (label efficiency).** At a fixed review budget, how close do uncertainty × class-balanced selection plus the StreamTrap recipe get to oracle accuracy?
- **RQ4 (video counting).** Does SAM 3 beat MegaDetector + ByteTrack at per-clip counts, and on which SA-FARI subsets?

**Hypotheses** (to test, not claims):
- **H1:** temperature scaling lowers ECE on the next interval compared with uncalibrated scores.
- **H2:** ACI with day/night groups has fewer α-violations than split-conformal across test sites.
- **H3:** at a 10% budget, our policy gives higher final-label macro-F1 than random or least-confidence review (paired bootstrap over sites, p < 0.05).
- **H4:** SAM 3 has lower MaxN error than MegaDetector + ByteTrack, overall and on night clips.

**Baselines:**
- zero-shot BioCLIP 2 with no review;
- a raw-softmax threshold;
- random review, least-confidence, margin, entropy and k-center review at the same budget;
- naive fine-tuning;
- StreamTrap's "accumulated with recipe" (all labels; the oracle);
- SpeciesNet (reference; possible leakage);
- for video: MegaDetector + ByteTrack / OC-SORT (SA-FARI rows).

**Metrics, chosen per task:**

| Task | Metrics |
|---|---|
| Classification | Average per-class accuracy (StreamTrap's metric), macro-F1, per-species F1, confusion matrices |
| Calibration / triage | ECE and adaptive ECE; coverage; realized error among auto-accepted (bootstrap CI); α-violation rate across sites; AURC; human labels used; labels to reach oracle −2 points |
| Video | cgF1, pHOTA, TETA (species prompts); HOTA, IDF1, DetA, AssA ("animal"), via the sam3 `veval` and TrackEval |
| Counting | MaxN MAE, RMSE and exact-match % per clip |
| Camouflage (P1) | S-α, F-β^w, E-φ, MAE (COD10K); IoU/Dice on SA-FARI small/challenging crops |
| Aerial (P1) | Point precision/recall/F1 within a pixel radius; count MAE/RMSE |
| Efficiency | images/s; s per clip; GPU-ms per stage; peak VRAM (`torch.cuda.max_memory_allocated`); end-to-end time for 1,000 images |
| Human | s per item; correction rate; agreement between reviewers on 50 shared items |

**Ablations (Exp 1):**
- calibration on/off;
- split-conformal vs ACI;
- Mondrian day/night grouping on/off;
- class-balance quota on/off;
- Balanced Softmax on/off;
- linear head vs LoRA;
- WiSE on/off;
- budget ∈ {1, 5, 10, 25}%;
- α ∈ {2, 5, 10}%;
- sequence aggregation on/off;
- calibration fraction ∈ {0.2, 0.3, 0.5} and per-site vs pooled calibration;
- simulated reviewer noise of 0 or 2%.

**Robustness:**
- day vs night;
- synthetic blur and darkness on test crops;
- cross-dataset: policy tuned on dataset X's sites, tested on dataset Y.

**Reproducibility:**
- configs in git;
- manifests with hashes;
- locked environments (`uv.lock` per image);
- Docker image digests;
- seeds (≥3 for anything that trains);
- every result row in `evaluation_runs` records the git commit;
- one frozen test run;
- negative results reported.

**Leakage safeguards:**
- SAM 3 is evaluated only on the SA-FARI test set;
- SpeciesNet on LILA is a non-zero-shot reference;
- sites are disjoint between dev and test;
- intervals are chronological;
- bursts are kept together;
- there is a written leakage statement in the report.

### Application testing strategy

| Layer | What is tested |
|---|---|
| Model adapters | "Golden" outputs on 5 fixed inputs (within a tolerance); CPU stub adapters for CI |
| Metric code | Unit tests on hand-computed toy cases (ECE, conformal coverage on synthetic streams, MaxN) |
| API | pytest + httpx against Postgres in Docker; authentication and roles; bad, corrupt and oversized uploads |
| Database | Alembic upgrade and downgrade; foreign keys; unique (dataset, sha256) |
| Workers | Idempotent re-runs (no duplicate predictions); failure leaves a `failed` status with an error |
| Frontend | Vitest + React Testing Library for components |
| End to end | Playwright: login → project → upload → job done → review → export |
| Acceptance | Scripted demo dataset run; 1,000-image / 50-video load test |

---

## 10. UI/UX blueprint

**Navigation.** A left sidebar for each project: Overview · Data · Jobs · Media · Review · Analytics · Models & Evaluation · Exports · Settings. The top bar has a project switcher and the user menu.

**Screens:**

| Screen | Contents and interactions |
|---|---|
| **Overview** | KPIs (processed, % auto-accepted, review backlog, species verified); recent jobs; alerts (failed items, low-quality share). |
| **Data** | Deployments table. Drag-and-drop folder/zip upload with a per-file validation list and duplicate warnings. Dataset versions ("freeze snapshot"). |
| **Jobs** | Progress by stage; throughput; errors with retry; cancel. |
| **Media browser** | Virtualized thumbnail grid. Filters: species, triage status, calibrated-confidence range, day/night, quality flag, deployment, date. |
| **Clip/Image viewer** | Overlay toggles (boxes, masks, track IDs, labels). Track timeline. MaxN. Side panel: top-5 species with calibrated %, **reason codes** for review (low confidence, night, tiny object, model disagreement, camouflage), and the model versions used. "Refine (DepthSAM)" button (P1). |
| **Review queue** | Risk-ordered, one item at a time. Keys: `1–9` pick a top-k label, `A` accept, `E` empty, `X` reject box, `U` unsure, `Z` undo, `/` species search. Burst batch-accept (P1). Session stats (items/hour). Time per item is recorded. |
| **Analytics** | Species counts (toggle verified-only vs including auto-accepted). Diel activity. Detections per deployment and RAI (P1). Trends. Every chart labels where its data comes from. |
| **Models & Evaluation** | Versions with license and metrics. Risk–coverage curves, reliability diagrams, confusion matrices, per-species F1. Activate or compare versions. Adaptation history. |
| **Exports** | CSV, COCO, Camtrap DP (P1), HTML/PDF report (P1). |
| **Settings** | Users and roles; α target; review budget; coordinate obfuscation; geofence country. |

**UX rules:**
- Confidence is always shown as a **calibrated %** plus a High/Med/Low badge, using colour + icon + text so it works for colour-blind users.
- "Predicted" vs "Verified" is always visible.
- Every view has empty, loading and error states.
- Long jobs never block the UI.
- Desktop-first (≥1280 px); tablet usable; analytics readable on a phone.

**User journey:**
1. A field ecologist uploads an SD-card dump.
2. Duplicates are dropped and files validated.
3. Live progress shows while it processes.
4. About X% is auto-accepted at the chosen error target.
5. A reviewer clears the risk-ordered queue.
6. They click "Adapt site", and the queue for the next upload shrinks.
7. They export Camtrap DP or a report.

**Final demo** (~10 minutes):
1. The full journey above on a mixed demo set (night, camouflage and video clips).
2. A model-comparison page.
3. Exp 1 and Exp 2 result slides.
4. A limitations slide.

---

## 11. Repository and database design

```
wildtriage/
  README.md  .env.example  docker-compose.yml  .github/workflows/ci.yml
  docs/            PROJECT_PLAN.md, adr/, model_cards/, dataset_cards/, report/
  backend/         pyproject.toml, alembic/ (versions/0001_initial_schema.py), app/{main.py, cli.py, schemas.py,
                   core/(config, security, logging), db/(base, session), models/(core, data, ml, predictions, review,
                   evaluation, enums), api/deps.py, api/v1/(auth, projects [+deployments, datasets, uploads], media,
                   jobs, review, models [+evaluations], analytics), services/(ingest, audit), storage/(base, local),
                   workers/(queue.py, runner.py, tasks/(ingest, inference))}, tests/
  ml/              pyproject.toml, models.yaml (license manifest), wildtriage_ml/{adapters/(base, stubs, megadetector,
                   bioclip, sam3, experimental), pipeline/(quality, sampling, aggregate, counting),
                   triage/(calibration, conformal, selection, policy), adaptation/(losses, heads, lora),
                   registry.py, runtime.py}, tests/
  research/        configs/, data_prep/streamtrap_mini.py, experiments/(exp1_triage [runs], exp2_video_counts,
                   exp3_camouflage, exp4_aerial), tests/
  frontend/        package.json, vite.config.ts, src/{App.tsx, api/, components/, hooks/, lib/, pages/}
  docker/          api.Dockerfile, worker-gpu.Dockerfile, frontend.Dockerfile, nginx.conf
  scripts/         native/ (setup, start-postgres, start-api, start-worker, start-frontend, backup .ps1),
                   license_check.py, seed_demo.py
  third_party/     (pinned clones of sam3, DepthSAM, MegaDetector-Overhead; gitignored)
```

### Key entities (PostgreSQL)

| Area | Tables |
|---|---|
| Users and projects | `users`; `projects` (settings JSONB: α, budget, country); `taxa` (scientific/common name, rank, parent); `project_species`; `deployments` (site, camera, start/end, lat/lon (obfuscated), modality) |
| Data | `datasets` (modality, source, license); `dataset_versions` (manifest URI + sha256, split spec, frozen_at); `media_items` (kind, storage URI, sha256 unique per dataset, dimensions, duration/fps, captured_at, is_night, quality JSONB, status, error); `frames` (sampled frames only) |
| Models and jobs | `model_versions` (task, weights URI + sha256, **license**, config, metrics, parent_id, is_active); `jobs` (also the queue: type, queue, priority, status, attempts/max_attempts, run_after, locked_by, heartbeat_at, progress, counts, params, error, timestamps) |
| Predictions | `detections` (media/frame/track, model_version, job, normalized bbox, mask URI, score, category); `tracks`; `classifications` (target, top-k JSONB, taxon, raw and calibrated score) |
| Triage and review | `triage_decisions` (policy version, decision, risk, reasons, features JSONB); `review_tasks` (priority, status, assignee); `verification_decisions` (append-only: action, new taxon/bbox, reviewer, time_spent_ms) |
| Ground truth, evaluation, audit | `annotations` (imported ground truth tied to a dataset version); `evaluation_runs` (model/policy, dataset version, task, config, metrics, artifacts, git commit); `exports`; `audit_log` (actor, action, entity, before/after JSONB) |

Masks are stored as RLE files in object storage, referenced by URI. Boxes and scores live in the database.

### API responsibilities (`/api/v1`)

| Endpoint | Responsibility |
|---|---|
| `auth` | Login, logout, `/me` |
| `projects`, `deployments`, `datasets` | CRUD |
| `datasets/{id}/uploads` | Multipart now; resumable uploads (tus) in P1 |
| `datasets/{id}/import` | COCO, MegaDetector JSON, SA-FARI |
| `datasets/{id}/versions` | Freeze a snapshot |
| `media` | Filtered list, detail, signed file URL, predictions |
| `jobs` | Create, get, SSE events, cancel |
| `review/queue`, `review/tasks/{id}/decision` | Review queue and decisions |
| `models` | List, activate, adapt |
| `evaluations` | Evaluation runs |
| `analytics/{project}/{summary\|activity\|events\|rai}` | Analytics |
| `exports` | Create and fetch exports |
| `health`, `metrics` | Health and metrics |

---

## 12. Risk register (top risks)

| # | Risk | Likelihood / Impact | Mitigation and fallback |
|---|---|---|---|
| 1 | SAM 3 / SA-FARI gated access delayed | M / H | Request on day 1. Meanwhile, LILA work. Fallback: SAM 2.1 (Apache-2.0) with MegaDetector box prompts. |
| 2 | Only one capable GPU (lab PC), shared by 2 people; the T1000 has 4 GB and the laptop has no CUDA | H / M | Lab-PC calendar. Heavy jobs batched overnight. Extract embeddings once and cache them, so Exp 1 runs on any CPU. Light GPU work on the T1000. CPU stubs for platform development. |
| 3 | Dependency conflicts (SAM 3 / DepthSAM / MegaDetector) | H / M | Separate environments/images and job queues; DepthSAM is P1 and isolated. |
| 4 | Exp 1 shows no gain | M / H | Analysis plan registered before the test run. Multiple budgets. Report negative results. The product metric (savings vs labelling everything) still holds. |
| 5 | StreamTrap code/data unreleased | H / M | Rebuild StreamTrap-mini; document differences; don't claim to match their numbers. |
| 6 | Leakage (SAM 3 ↔ SA-FARI train; SpeciesNet/BioCLIP 2 ↔ LILA) | H / H | The safeguards in §9; a leakage statement. |
| 7 | Licensing (NC data/weights; YOLOv5 AGPL) | H / M | License field on every model and dataset; research vs commercial manifests; `license_check.py`. |
| 8 | Video overlay complexity in the browser | M / M | Server-rendered overlay MP4 for P0; client-side RLE canvas in P1. |
| 9 | Windows/OneDrive environment | H / M | Repo and data outside OneDrive. Native Windows run path (uv + portable Postgres) works without Docker/WSL2. |
| 10 | 2-person bandwidth / exams | H / H | Scope ladder; W4 checkpoint; Friday integration; W7 freeze. |
| 11 | Exposure of endangered-species locations or human privacy | M / H | Coordinate obfuscation; person blur; role-based access; audit. |
| 12 | DepthSAM weights or reproduction issues | M / L | Use the Jittor port's reported MAE as a sanity check; move to P2. |
| 13 | Instructor rejects the P3 swap or ECCV isn't approved | L–M / M | Revert to Real-Wild-VLM (re-implement its prompts as the VLM expert); the architecture is unchanged. |
| 14 | Managed lab PC: no admin rights, no Docker/WSL2, re-imaging, shared GPU, no overnight access | M–H / H | Ask IT in W1. Native run path without admin. Keep code in git and data manifests reproducible, so a wiped disk only costs a re-download. Book GPU time. |
| 15 | 20 GB VRAM is not enough for SAM 3 on long or multi-object clips, or for co-resident models (4 GB on the T1000 is not enough for anything heavy) | M / M | bf16; chunk clips ≤30 s; VRAM budget with LRU eviction; never load DepthSAM or a VLM next to SAM 3; SAM 3.1 (the faster multi-object tracker) if needed; measure in W1. |
| 16 | The lab PC becomes unavailable (lab closed, exams, re-imaged) | M / H | Everything except SAM 3, DepthSAM and the final demo runs on the T1000 or laptop. Finish SA-FARI runs early (W2–W3) and keep their outputs. Record a backup demo video in W7. **Fallback for GPU-heavy batch jobs: molab** (see "Cloud fallback" below), with Colab or Kaggle as a second option. |

### Cloud fallback for GPU-heavy batch jobs (molab)

**What was found** [V from marimo's announcement, [blog](https://marimo.io/blog/reintroducing-molab) and [newsletter 25](https://marimo.io/blog/newsletter-25); one search, not cross-checked against molab's docs]:
- molab is marimo's cloud notebook service. It is free in public preview and runs on CoreWeave.
- Default notebook: 4 CPUs, 32 GB RAM. An opt-in GPU is available: an RTX Pro 6000 Blackwell with 96 GB VRAM. marimo says GPUs are free "as long as usage is reasonable" and does not define that.
- Sessions last at most 12 hours. Notebooks idle for more than 90 minutes are shut down automatically.

**What to run there** [R]:
- SAM 3 runs on the SA-FARI test set (Exp 2).
- DepthSAM reproduction (Exp 3).
- BioCLIP 2 embedding extraction for Exp 1.

**What not to run there** [R]:
- The app and the live demo (it needs a server that stays up).
- Anything that cannot be stopped and resumed.

**Rules that make the fallback safe:**
1. **Chunked, resumable jobs.** Every experiment script writes its results after each chunk (for example per site, per clip batch, or per 500 images) and skips chunks that already exist. A 12-hour limit or an idle shutdown then costs minutes, not days.
2. **Durable output.** Save embeddings, predictions and metrics somewhere that outlives the session (the team drive or a git-ignored results folder you copy out). [U: does molab keep files between sessions? Test in W1.]
3. **No secrets in notebooks.** Use the token mechanism molab provides for the Hugging Face token (SAM 3, SA-FARI are gated). Never paste it into a cell or commit it. [U: how molab stores secrets.]
4. **Pinned environment.** Run `uv pip install -e ml` plus pinned versions from the repo, so results match the lab PC. Record the molab GPU name and package versions in each `evaluation_runs` row.
5. **Same code path.** Notebooks only call `research/experiments/*.py` and `wildtriage_ml`, so there is one implementation to test.
6. **Licenses still apply.** SA-FARI and DepthSAM's depth model are non-commercial: fine for coursework. Do not upload anything the dataset terms forbid redistributing.

**Preview risks:** limits, quotas or pricing can change, "reasonable usage" is undefined, and a shared GPU may be unavailable at peak times. Treat molab as the *first* fallback and Colab or Kaggle (free tiers with limits) as the second. If neither works, scale down: fewer SA-FARI test clips (a fixed, documented subset) and SAM 3 at 3 fps.

**W1 test (acceptance):** one SA-FARI test clip and ten LILA crops processed end to end in molab, with these numbers written down: peak VRAM, seconds per clip, whether files persist, what happened on idle, and how the HF token was supplied.

**Unresolved [U]:**
- Whether the related-work search missed a paper that already does risk-controlled review for camera traps (repeat properly in W1).
- Parts of two sources I could not read: the Dussert et al. calibration paper and the conformal acoustic-indices paper (pages blocked).
- The Real-Wild-VLM and DepthSAM PDFs were not read (DepthSAM: abstract and code README only; Real-Wild-VLM: abstract only).
- ECCV on the approved venue list, and instructor approval of the swap.
- Whether the public SAM 3 includes SA-FARI training data.
- MegaDetector and SpeciesNet weight licenses.
- The Delplanque dataset license.
- The exact path of the SA-FARI GCS bucket.
- What the lab PC allows (admin rights, Docker/WSL2, overnight runs, remote access, disk persistence).

---

## 13. Final acceptance checklist

**Product:**
- [ ] A fresh clone, following the README, reaches the seeded demo in ≤30 minutes on the lab PC (RTX 4000 Ada, 20 GB).
- [ ] JPG/PNG/MP4/MOV/AVI are ingested. Unsupported or corrupt files are rejected with an error code per file.
- [ ] A batch of 1,000 images + 50 videos completes with 0 unhandled failures. Per-item errors are recorded.
- [ ] Every prediction references a `model_version` and a `job`. Every review decision has a reviewer, timestamp and before/after values in `audit_log`.
- [ ] The review loop works end to end. Analytics can switch between verified-only and predicted data. CSV and COCO exports validate.
- [ ] CI is green. Core backend and ML modules have ≥70% line coverage. The Playwright happy path passes.

**Research:**
- [ ] The SA-FARI reproduction table (SAM 3 + MegaDetector + ByteTrack) is reported, with differences in protocol explained.
- [ ] Exp 1: all baselines plus our method on held-out test sites, with 95% bootstrap CIs. The full ablation table. α-violation analysis. Negative results included.
- [ ] Calibration (ECE before/after) and day/night robustness are reported.
- [ ] An efficiency table (latency, throughput, VRAM per stage).
- [ ] The review-effort mini-study is reported with its caveat (small n).

**Documentation and demo:**
- [ ] README, architecture, model cards and dataset cards (with licenses), and the technical report (contributions vs reused components, leakage statement, limitations).
- [ ] The demo is rehearsed twice, and a backup video is recorded.

---

## 14. Repository skeleton status

The skeleton in `wildtriage/` follows §5, §6 and §11. What has been verified so far is recorded in `wildtriage/README.md` under "Skeleton status". In short:
- The math in the triage module and the backend's imports and schema are unit-tested.
- No models, datasets or weights are included or downloaded.
