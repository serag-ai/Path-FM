<p align="center">
  <img src="assets/Fig1.png" alt="Do Foundation Models Truly Outperform Domain-Specific Models? Evidence from Digital Pathology" width="100%">
</p>

<h1 align="center">Do Foundation Models Truly Outperform Domain-Specific Models? Evidence from Digital Pathology</h1>

<p align="center">
  <a href="https://www.sciencedirect.com/xxx"><img src="https://img.shields.io/badge/ScienceDirect-View%20Paper-orange" alt="Paper"></a>
  <a href="https://huggingface.co/serag-ai"><img src="https://img.shields.io/badge/Hugging%20Face-Models-blue" alt="Models"></a>
  <a href="http://creativecommons.org/licenses/by/4.0/"><img src="https://img.shields.io/badge/license-CC--BY--4.0-brightgreen" alt="License: CC BY 4.0"></a>
</p>

## Overview

This repository contains the code and training/inference scripts used for benchmarking seven general-purpose pathology foundation models (FMs) and three domain-specific FMs across 11 patch-level datasets spanning three clinically relevant domains: pediatric hematology, prostate cancer, and breast cancer, using both linear probing and last-layer fine-tuning adaptation strategies. To contextualize the performance of pretrained FMs, a suite of five deep learning architectures trained from scratch on domain-specific data without any external pretraining were also benchmarked.  

## Repository Structure

- **`patho-fm/Linear-Probing`** — Training/testing FMs under a linear probing approach where pre-trained FMs served as fixed feature extractors with all backbone parameters frozen during training.
- **`patho-fm/Partial-FT`** — Training/testing FMs under a last-layer fine-tuning approach by updating the classification head together with the final trainable layer of each backbone, while all earlier layers remained frozen.
- **`patho-fm/Robustness-Check`** — Training FMs on a primary dataset and evaluating them on one/multiple external test datasets.
- **`patho-fm/DL-models`** — Training/testing a consistent set of standard computer vision backbones from scratch.
- **`assets/`** — Repository media.

## Foundation models
- ** Hematology **


## Deep learning models

## Datasets



## Citation

If you use these scripts in your research, please cite:

```bibtex
@article{chaima2026,
  title   = {Do Foundation Models Truly Outperform Domain-Specific Models? Evidence from Digital Pathology},
  author  = {Ben Rabah, Chaima and Serag, Ahmed},
  journal = {MAKE},
  year    = {2026},
  url     = {xxxxx}
}
```

## License

Released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/) license.
