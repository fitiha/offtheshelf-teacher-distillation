# Does an Off-the-Shelf Teacher Help Small Student Language Models? Evidence From TinyStories

Code and notebooks accompanying the empirical study testing whether a single frozen, off-the-shelf teacher (GPT-2 or GPT-Neo-125M) improves small GPT-2-architecture student models (7.24M–19.22M parameters) on TinyStories and WikiText-2.

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Paper License: CC BY 4.0](https://img.shields.io/badge/Paper%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper DOI](https://img.shields.io/badge/Paper%20DOI-TODO-orange.svg)](#how-to-cite)
[![Code DOI](https://img.shields.io/badge/Code%20DOI-TODO-orange.svg)](#how-to-cite)

> **Status:** This work is a **preprint** and has **not** been peer reviewed. Read the full report in [`paper/weldeananiya-2026-offtheshelf-teacher-distillation-tinystories.pdf`](paper/weldeananiya-2026-offtheshelf-teacher-distillation-tinystories.pdf).

## TL;DR

- **Distillation does not help in this setup:** Across five student sizes (7.24M–19.22M parameters) on TinyStories at $\alpha = 0.5$, students distilled from a frozen off-the-shelf GPT-2 teacher (`124M`) perform worse in validation cross-entropy than identical baselines trained without distillation.
- **Harm scales monotonically with teacher weight:** Sweeping $\alpha \in \{0.25, 0.5, 0.75, 1.0\}$ at three student sizes shows worse validation loss as more weight $(1 - \alpha)$ is placed on the teacher signal, converging to parity with the baseline only at $\alpha = 1.0$.
- **The pattern generalizes across teachers:** Replacing GPT-2 with GPT-Neo-125M (`125.2M` parameters, trained on the Pile) reproduces negative distillation benefit at all five student sizes.
- **Domain mismatch is a primary driver:** Holding each teacher fixed and swapping TinyStories for WikiText-2 cuts the harm roughly in half at every tested teacher and size (though benefit remains negative).
- **Evaluation-metric pitfall:** Hugging Face `Trainer.evaluate()` reuses `compute_loss`, which originally reported the weighted distillation loss rather than plain cross-entropy; re-evaluating with a plain `Trainer` removes this artifact.

## Results

All numbers below are copied directly from Tables 3–6 of the preprint (validation cross-entropy loss; `Benefit = Baseline Loss - Distilled Loss`, so negative values indicate harm).

### Student Size Sweep on TinyStories ($\alpha = 0.5$, GPT-2 teacher, two seeds; Table 3)

| Size | Params (M) | Baseline Loss | Distilled Loss | Benefit |
| :--- | ---: | ---: | ---: | ---: |
| Tiny | 7.24 | 5.850 | 6.258 | −0.408 |
| Zoom | 8.26 | 5.613 | 5.968 | −0.355 |
| XSmall | 9.3 | 5.402 | 5.785 | −0.383 |
| Small | 12.34 | 4.992 | 5.356 | −0.364 |
| Medium | 19.22 | 4.469 | 4.740 | −0.271 |

### Distillation Weight ($\alpha$) Sweep on TinyStories (GPT-2 teacher; Table 4)

| Size | $\alpha = 0.25$ | $\alpha = 0.5$ | $\alpha = 0.75$ | $\alpha = 1.0$ |
| :--- | ---: | ---: | ---: | ---: |
| Tiny | −0.744 | −0.408 | −0.172 | −0.061 |
| Zoom | −0.665 | −0.355 | −0.111 | +0.001 |
| Medium | −0.625 | −0.271 | −0.064 | +0.004 |

### Teacher Comparison on TinyStories ($\alpha = 0.5$, single seed for GPT-Neo-125M; Table 5)

| Size | Baseline Loss | GPT-2 Benefit | GPT-Neo Benefit |
| :--- | ---: | ---: | ---: |
| Tiny | 5.850 | −0.408 | −0.376 |
| Zoom | 5.613 | −0.355 | −0.326 |
| XSmall | 5.402 | −0.383 | −0.361 |
| Small | 4.992 | −0.364 | −0.346 |
| Medium | 4.469 | −0.271 | −0.231 |

### Target Domain Swap: TinyStories vs. WikiText-2 ($\alpha = 0.5$, single seed; Table 6)

| Size | Teacher | TinyStories Benefit | WikiText-2 Benefit |
| :--- | :--- | ---: | ---: |
| Tiny | GPT-2 | −0.408 | −0.206 |
| Tiny | GPT-Neo | −0.376 | −0.145 |
| Medium | GPT-2 | −0.271 | −0.083 |
| Medium | GPT-Neo | −0.231 | −0.110 |

## Repository Structure

| Path | Description / Paper Section Reproduced |
| :--- | :--- |
| `paper/weldeananiya-2026-offtheshelf-teacher-distillation-tinystories.pdf` | Preprint PDF (August 2026) |
| `notebooks/01_main_sweep_and_alpha.ipynb` | Sections 5, 6.1–6.2, 8 (Tables 1, 3, 4; Figures 1–2; padding-mask check) |
| `notebooks/02_gptneo_teacher_generalization.ipynb` | Section 6.4 (Table 6, Figure 3): GPT-Neo-125M on WikiText-2 *(note: filename is swapped with `04_`)* |
| `notebooks/03_wikitext_domain_swap_gpt2.ipynb` | Section 6.4 (Table 6, Figure 3): GPT-2 on WikiText-2 |
| `notebooks/04_wikitext_domain_swap_gptneo.ipynb` | Section 6.3 (Table 5): GPT-Neo-125M on TinyStories *(note: filename is swapped with `02_`)* |
| `notebooks/README.md` | Per-notebook summary (purpose, inputs, outputs, runtime, tables/figures) |

## How to Reproduce

- **Hardware:** Single NVIDIA T4 GPU (Kaggle or Google Colab), 1 epoch, batch size 16, max sequence length 128, GPT-2 tokenizer, default AdamW in Hugging Face `Trainer`.
- **Running:** Open any notebook in Kaggle or Colab with a T4 GPU enabled (`CUDA_VISIBLE_DEVICES="0"`), run the `!pip install transformers datasets accelerate -q` cell (or `pip install -r requirements.txt` locally), and execute cells in order.
- **Approximate per-condition runtimes** (Table 2 of the preprint, 20,000 training examples, 1 epoch):
  - Tiny (7.24M): ~1.4 min baseline / ~5.5–6.0 min distilled
  - Zoom (8.26M): ~1.8 min baseline / ~6.0 min distilled
  - XSmall (9.3M): ~1.9 min baseline / ~6.0 min distilled
  - Small (12.34M): ~2.2 min baseline / ~6.3 min distilled
  - Medium (19.22M): ~2.9 min baseline / ~6.8 min distilled
- **Expected outputs:** Pure cross-entropy `eval_loss` dictionaries printed to stdout at the end of each experiment cell.
- **Random seeds:** Notebooks `02`, `03`, and `04`, as well as cells 37–93 of `01`, call `torch.manual_seed(...)` explicitly (`42`, `456`, `789`, `2024`, `999`). Earlier exploratory cells (11–35) in `01` do not set a seed, and Python/NumPy RNGs are not separately seeded.

## Data

Datasets are loaded directly from the Hugging Face Hub at runtime and are not stored in this repository:

- **TinyStories:** `load_dataset("roneneldan/TinyStories")`, using the first `20,000` examples from `"train"` and `2,000` from `"validation"`.
- **WikiText-2:** `load_dataset("wikitext", "wikitext-2-raw-v1")`, filtered with `len(x["text"].strip()) > 20` and capped at `20,000` training and `2,000` validation examples.

## Evaluation-Metric Pitfall

Hugging Face `Trainer.evaluate()` calls the same `compute_loss` method used during training. Subclassing `Trainer` to return $\alpha \mathcal{L}_{\text{CE}} + (1 - \alpha) T^2 D_{\text{KL}}$ caused `evaluate()` to report the weighted distillation loss rather than plain cross-entropy, making distilled runs incomparable to baselines. In the notebooks, every distilled model is re-evaluated immediately after training with a plain `Trainer` instance (from `notebooks/01_main_sweep_and_alpha.ipynb`, cell 88):

```python
trainer = DistillationTrainer(
    model=m, teacher_model=teacher, temperature=temperature, alpha=alpha_val,
    args=args, train_dataset=tokenized_train, eval_dataset=tokenized_val,
    data_collator=data_collator,
)
trainer.train()

# Pure CE eval, comparable to baseline
pure_trainer = Trainer(model=m, args=args, eval_dataset=tokenized_val, data_collator=data_collator)
pure_ce = pure_trainer.evaluate()["eval_loss"]
```

## Limitations

As reported in Section 8 of the preprint:
- **Domain and capacity mismatch not fully isolated:** WikiText-2 narrows rather than eliminates the domain gap to the teachers; a condition with a teacher fine-tuned on TinyStories was not tested due to time constraints.
- **Limited scale for generalization runs:** Teacher generalization (Section 6.3) used a single seed per size; domain generalization (Section 6.4) used a single seed at only two sizes (Tiny, Medium).
- **Only two teachers and two datasets:** Both teachers are in the 124M–125M range (`gpt2`, `EleutherAI/gpt-neo-125m`) and only TinyStories and WikiText-2 were tested.
- **Padding tokens in the KL term:** The KL term originally included padding positions; masking them out was checked on Tiny at $\alpha \in \{0.25, 0.5\}$ (differing by 0.002–0.007 loss points), but not repeated across all conditions.
- **Uneven seed coverage:** $\alpha = 0.75$ and $\alpha = 1.0$ used a single seed at three sizes; XSmall and Small were not included in the $\alpha$ sweep.
- **Limited training budget:** One epoch on 20,000 examples (~1% of TinyStories training data).

## How to Cite

```bibtex
@misc{weldeananiya2026offtheshelf,
  author       = {Weldeananiya, Natnael F.},
  title        = {Does an Off-the-Shelf Teacher Help Small Student Language Models? Evidence From TinyStories},
  year         = {2026},
  month        = aug,
  note         = {Preprint, not peer reviewed. Work done at Gheero},
  doi          = {TODO: PAPER_DOI}
}
```

See [`CITATION.cff`](CITATION.cff) for software citation metadata (`type: software`, `version: 1.0.0`, `date-released: 2026-11-26`, `doi: TODO: CODE_DOI`).

## License and Acknowledgements

- **Code & notebooks:** [MIT License](LICENSE) (Copyright (c) 2026 Natnael F. Weldeananiya).
- **Paper PDF:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

<!-- TODO: confirm acknowledgement wording -->
Conducted during the AI/ML residency at Gheero.
