# CT-CBM — concept selection heuristic

This folder is the copy of **CT-CBM** used for the experiments of
*Towards Faithful Multimodal Concept Bottleneck Models*. It is kept here so that the
paper's results can be reproduced from a single checkout.

The maintained version, its documentation and any updates live in the standalone
repository: **https://github.com/yann-Choho/CT-CBM**. Please open issues and pull
requests there.

## What it does

Given a concept-annotated dataset (train / val / test CSVs), the five steps of the
heuristic rank the concepts and select them cluster by cluster until a target coverage
is reached:

1. concept clustering (`scripts/clustering.py`);
2. black-box model training (`models/BaselineModel_*.py`);
3. CAV computation (`scripts/cav_computation_unified.py`);
4. TCAV and LIG rankings (`scripts/TCAVS_*.py`, `scripts/LIG_ranking_*.py`);
5. combined score, identifiability (R² / F1) and cluster-wise coverage
   (`scripts/identifiability_score_R2_unified.py`, `scripts/concept_coverage_analysis.py`).

The output is a ranked concept list per coverage level (`.pkl`), which `run_CBM` in
`../src/CBM_pipeline.py` consumes through `import_concept_list` / `concept_level`.

## Running it

Install the root `requirements.txt` of this repository (the `requirements.txt` of this
folder is the standalone subset). Then open one of `notebooks/Step_1_to_5_notebook*.ipynb`
(one per dataset / modality) and set the variables of the *SETUP ENVIRONMENT VARIABLES*
cell:

```python
dataset       = 'n24news'     # 'n24news', 'agnews', 'dbpedia'
annotation    = 'cb_llm'      # 'C3M', 'our_annotation', 'cb_llm'
combine_type  = 'concat'      # 'concat', 'combine', 'None'
model_name    = 'clip'        # 'clip', 'clip-large', 'blip', 'bert-base-uncased', ...
modality_mode = 'multimodal'  # 'text', 'image', 'multimodal'
path_to_input  = "data/datasets/N24News/concat"
path_to_output = "data/datasets/N24News/concat/cb_llm_annotation/outputs_concept_scoring"
```

Everything else is routed automatically by `run_experiments/unified_config.py` and
`run_experiments/scripts/pipeline_dispatcher.py`.

## Input format

Three CSVs per annotation in `path_to_input`: `train_df_{annotation}.csv`,
`val_df_{annotation}.csv` (the validation set does not need concept annotations) and
`test_df_{annotation}.csv`. Concepts are **all the columns prefixed with `concept_`**:

| Modality | Columns |
|---|---|
| `text` | `text`, `label`, `concept_*` |
| `image` | `path_to_image`, `label`, `concept_*` |
| `multimodal` | `text`, `path_to_image`, `label`, `concept_*` |

`path_to_image` is the file name of the image (without directory; `.jpg` is appended
when missing) inside the images folder given by `config.path_to_input_images`.

Annotation types: `C3M` and `our_annotation` are binary concepts (0/1); `cb_llm` is
continuous (0–1) and is used with the ACC filtering of CB-LLM.
