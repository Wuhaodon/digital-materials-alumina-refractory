# Digital Materials for Low-Carbon Alumina Refractory

*Evaluation of Micro-Mechanical Behavior for low Carbon-Containing Alumina Refractory assisted by Interpretable Machine Learning: Toward Development of Digital Materials*

![Graphical Abstract](graphical_abstract.png)

## Overview

This repository contains the code and model checkpoint for the manuscript workflow on low-carbon alumina refractory micromechanics and interpretable machine learning.

The repository includes scripts for clustering, cross-sample correspondence analysis, feature screening, classification benchmarking, interpretable TabPFN regression, and ablation analysis.


## Availability

This repository releases source code and model files only.

The experimental dataset used in the manuscript is not publicly distributed.
Users who wish to run the workflow must prepare their own compatible input data.

## Environment

- Python `>=3.10`
- Main dependencies are declared in `pyproject.toml`
- A typical setup is `pip install -e .`
- The local TabPFN checkpoint is stored at `external/tabpfn_cache/tabpfn-v2-classifier.ckpt`

## Repository Structure

- `src/nano2/`: reusable package code
- `configs/`: configuration files for the main workflow
- `scripts/`: runnable scripts for the manuscript and supplementary analyses
- `external/tabpfn_cache/`: local model checkpoint used by selected scripts

## Main Workflow Scripts

- `scripts/00_clustering_preprocess.py`: prepare specimen-specific clustering inputs
- `scripts/00_clustering_per_sample.py`: run per-sample clustering and silhouette-based K selection
- `scripts/00_cluster_correspondence.py`: compute weighted probabilistic correspondence across specimens
- `scripts/00_clustering_comparison.py`: export clustering summary tables and overview figures
- `scripts/01_feature_selection_table6_benchmark.py`: run the classification benchmark and selected-feature comparison
- `scripts/01_feature_selection_high_corr_scan.py`: scan highly correlated candidate features
- `scripts/01_feature_selection_key_correlation_pairs.py`: summarize key correlation pairs discussed in the paper
- `scripts/01_feature_selection_target_feature_correlation.py`: export target-feature correlation summaries
- `scripts/02_multitarget_regression_tabpfn_shap.py`: run the main TabPFN regression and SHAP workflow
- `scripts/03_ablation_stability.py`: compare baseline and ablation models
- `scripts/03_ablation_highcorr_shap.py`: export SHAP results for ablation feature sets
- `scripts/03_plot_ablation_combined.py`: build the combined ablation comparison figure
- `scripts/03_plot_ablation_final.py`: build the reordered ablation figure

## Supplementary Scripts

- `scripts/supplementary_material/01_clustering_checks.py`
- `scripts/supplementary_material/02_weight_sensitivity_original_style.py`
- `scripts/supplementary_material/03_weight_sensitivity_visuals.py`
- `scripts/supplementary_material/04_export_table_s2_from_docx.py`
- `scripts/supplementary_material/05_small_sample_all_models.py`
- `scripts/supplementary_material/06_small_sample_tabpfn_xgboost.py`

## Rights and Use

Copyright (c) 2026 Wuhaodong. All rights reserved.

No permission is granted to use, reproduce, modify, distribute, or commercialize this code or the associated model files without prior written permission from the copyright holder.


## Notes

- Paper notation and code naming follow the mapping `Ue = WE`, `Upp = WPP`, `S = Stiffness`, and `E = Modulus`
- Some scripts assume manuscript-specific file layouts and are provided primarily as a research workflow record
