# Spectral Certification for Class-Level Machine Unlearning

**Authors:** Yan Zhang, Chenggang Lu (Zhejiang University of Technology)

**Target journal:** Applied Intelligence (Springer)

This folder is a self-contained submission package built from the official
Springer `svjour3` template (`smallextended` option).

## Files

| File | Description |
|---|---|
| `manuscript.tex` | Main manuscript (LaTeX source) |
| `references.bib` | BibTeX bibliography (spmpsci style) |
| `svjour3.cls` | Springer journal class (from official template) |
| `svglov3.clo` | Springer option file (loaded by `svjour3.cls`) |
| `spmpsci.bst` | Springer math/physical-sciences bibliography style |
| `Makefile` | Build helper (from official template) |

## Compiling

Standard pdflatex + BibTeX workflow (run inside this folder):

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

or simply `make`.

The compiled document is `manuscript.pdf` (36 pages at time of writing).

## Reproducibility

Experiment scripts and multi-seed results (CIFAR-10 / Fashion-MNIST /
CIFAR-100 class-level benchmarks, TOFU and MUSE-style book unlearning on
LLaMA-2-7B) are public at
https://github.com/EurusDemerzel/rii_unlearning

## Notes

- The manuscript uses only standard packages (amsmath, amssymb, amsfonts,
  graphicx, url, multirow, booktabs) plus `fix-cm`.
- RII = Residual Irreversibility Index; MHPR = Multi-Held-Out Projection
  Residual; PSSD = Per-Sample State Disparity (appendix).
