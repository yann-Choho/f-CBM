# f-CBM — Towards Faithful Multimodal Concept Bottleneck Models

Code release for the paper [*Towards Faithful Multimodal Concept Bottleneck Models*](https://arxiv.org/abs/2603.13163).

**f-CBM** is a concept bottleneck model for text, image and text+image classification.
Compared with a standard joint CBM it adds

- a **KAN prediction head** on top of the concept layer (`kan_layer=True`), whose
  concept-to-class response curves can be plotted directly;
- a **differentiable leakage loss** (`leakage_loss=True`) that penalises task
  information flowing around the concepts;
- an optional **concept selection heuristic (CT-CBM)** that ranks concepts by a combined
  TCAV / LIG / identifiability score and adds them cluster by cluster until a target
  coverage is reached.

Experiments are run on **N24News** (text + image), **CUB-200-2011** (image + captions),
**AG News** and **DBpedia** (text), with `clip` (ViT-B/32) and `clip-large`
(ViT-L/14) backbones.

---

## Repository layout

```
f-CBM/
├── src/                      f-CBM itself
│   ├── CBM_pipeline.py       run_baseline() and run_CBM(): the two entry points
│   ├── CBM_model.py          joint / sequential / independent CBMs, leakage loss, plots
│   ├── KAN.py                KAN head and response-curve plots
│   ├── baseline_model.py     black-box CLIP / BLIP classifiers
│   ├── intervention.py       test-time intervention evaluator
│   ├── hyperparam_search.py  grid search used for Section 4 (resumable)
│   ├── data_*.py             dataset loaders (N24News, CUB, AG News, DBpedia)
│   └── path_info.py          PATH = 'data'  (root of the data folder, relative to f-CBM/)
├── CT-CBM/                   concept selection heuristic (steps 1-5), copy of github.com/yann-Choho/CT-CBM
├── notebooks/
│   ├── main_results/         Exp_baseline, Exp_sota_our_CBM, Exp_sota_CBLLM  (Tables of the paper)
│   ├── ablation/             Exp_sota_ablation (heuristic x KAN x leakage loss, random concepts)
│   ├── intervention/         run_interventions -> plot_intervention_results (Figure 7)
│   └── figures/              end-to-end N24News pipeline and KAN response curves (Figure 6)
├── scripts/download_data.py  fetches the datasets (see "Data")
├── requirements.txt
├── LICENSE                   MIT
├── NOTICE                    third-party attributions
└── data/                     NOT in git - created by scripts/download_data.py
```

All notebooks start with a bootstrap cell that locates the repository root, adds `src/`
to `sys.path` and `chdir`s to the root, so they can be launched from anywhere.

---

## Installation

Tested with Python 3.10, PyTorch 2.2.0 and CUDA 11.8. A GPU is required for training
(the paper's runs used 16 GB and larger GPUs; `clip-large` needs the larger ones).

```bash
conda create -n fcbm python=3.10
conda activate fcbm
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 11.8 wheels of PyTorch. For another CUDA version,
install PyTorch 2.2.0 first following https://pytorch.org, then the rest of the file.

The backbones (`openai/clip-vit-base-patch32`, `openai/clip-vit-large-patch14`,
`Salesforce/blip-image-captioning-base`) are downloaded from the Hugging Face Hub on
first use. No token is needed for them; if you use a gated model, copy `.env.example`
to `.env` and fill in `HF_TOKEN`.

---

## Data

The code reads everything from `data/` (path set in `src/path_info.py`). Nothing in
`data/` is versioned in git. Two kinds of files live there:

| | What | Where it comes from |
|---|---|---|
| 1 | Raw images of CUB-200-2011 and N24News | official sources (Caltech DATA, N24News authors) |
| 2 | **Our files**: concept annotations, train/val/test splits, concept scores of the heuristic, and optionally the trained checkpoints | Hugging Face Hub dataset *(upload in progress, see below)* |

AG News and DBpedia need no raw download: their annotation files already contain the
texts and labels.

### Datasets used in the paper

| Dataset | Modality | Classes |
|---|---|---|
| N24News | text + image | Food, Music, Sports, Technology |
| CUB-200-2011 | image (+ captions) | bird species |
| AG News | text | 4 topics |
| DBpedia | text | 6 topics |

Concepts come from the CB-LLM annotation procedure (`dataset_type='CBLLM'`, continuous
scores). N24News is also available with binary C3M concepts (`dataset_type='C3M'`).
The splits and the concept sets are exactly those of the paper; see the paper for the
per-dataset counts.

### Download

From the `f-CBM/` folder:

```bash
python scripts/download_data.py                              # everything (steps 1 and 2)
python scripts/download_data.py --datasets agnews dbpedia    # text datasets only
python scripts/download_data.py --with-checkpoints           # also fetch the trained models
python scripts/download_data.py --skip-annotations           # raw images only (step 1)
```

The script is idempotent and resumable: files already on disk are skipped and an
interrupted download restarts where it stopped. The CUB archive is checked against the
md5 published by Caltech.

> **Hugging Face dataset (step 2).** Our annotation files are being uploaded to a
> Hugging Face dataset repository; its identifier will be written in `HF_REPO_ID` at the
> top of `scripts/download_data.py` as soon as it is public. Until then, step 2 stops
> with an explicit message. Once the repository is public, you can also pass it by hand:
>
> ```bash
> python scripts/download_data.py --hf-repo <user>/f-cbm-data
> # or:  FCBM_HF_REPO=<user>/f-cbm-data python scripts/download_data.py
> ```
>
> The download is pinned to the tag `v1.0` of that repository, so everyone gets exactly
> the files used for the paper.

If Google Drive refuses the N24News archive (daily quota), the script tells you where to
download `N24News.zip` by hand (link in the authors' README,
https://github.com/billywzh717/N24News) and where to put it; rerun the script afterwards.

### Expected layout

After a full download `data/` looks like this (only the files the notebooks need are shown):

```
data/datasets/
├── N24News/
│   ├── imgs/                                                (official archive)
│   ├── news/nytimes_dataset.json                            (official archive)
│   ├── augmented_dataset_CBLLM/                CB-LLM concept scores, text and image
│   ├── augmented_dataset_C3M/                  C3M binary concepts
│   ├── combine/cb_llm_annotation/              train/val/test CSVs, label_dict, concept scores of the heuristic
│   └── models/                                 baseline checkpoints (--with-checkpoints)
├── CUB_200_2011/
│   ├── images/<class>/<image>.jpg                           (official archive)
│   └── combine/cb_llm_annotation/
├── agnews/
│   ├── df_with_topics_v4_CB_LLM*.csv
│   └── text/cb_llm_annotation/
└── dbpedia/
    ├── df_with_topics_v4_CB_LLM*.csv
    └── text/cb_llm_annotation/
```

To keep the data elsewhere (e.g. on a scratch disk), either create a symlink
`ln -s /path/to/storage data` inside `f-CBM/`, or change `PATH` in `src/path_info.py`
and pass the same location to `download_data.py --data-dir`.

### Licences

The raw datasets keep their own licences and are never redistributed by us: CUB-200-2011
(Caltech), N24News (Wang et al., LREC 2022; NYT content), AG News and DBpedia. Our
annotation files are derived from them.

---

## Running the experiments

Start Jupyter from the repository root and open the notebooks:

```bash
jupyter lab
```

| Notebook | What it does |
|---|---|
| `notebooks/main_results/Exp_baseline.ipynb` | black-box CLIP baselines on the four datasets (text, image, multimodal) |
| `notebooks/main_results/Exp_sota_our_CBM.ipynb` | **f-CBM** (heuristic + KAN + leakage loss) on the four datasets |
| `notebooks/main_results/Exp_sota_CBLLM.ipynb` | CB-LLM-style CBM (cos-cubed concept loss, linear head) |
| `notebooks/ablation/Exp_sota_ablation.ipynb` | 2x2x2 ablation on N24News, varying number of concepts, random concept subsets |
| `notebooks/intervention/run_interventions.ipynb` | trains the compared models, runs test-time interventions, saves JSONs to `saved_intervention_results/` |
| `notebooks/intervention/plot_intervention_results.ipynb` | plots Figure 7 from those JSONs |
| `notebooks/figures/Multimodal_CBM_N24News_pipeline.ipynb` | end-to-end pipeline on N24News, KAN response curves (Figure 6) |

The paper reports every result for both backbones. The `main_results` notebooks expose
a single variable at the top:

```python
backbone = 'clip'        # or 'clip-large'
```

### The two entry points

Everything goes through two functions of `src/CBM_pipeline.py`:

```python
import sys; sys.path.insert(0, "src")          # run from the f-CBM/ folder
from CBM_pipeline import run_baseline, run_CBM

# black-box baseline
history, model = run_baseline(dataset='N24', backbone='clip', modality='multi', num_epochs=10)

# f-CBM: KAN head + leakage loss + heuristic concept selection
scores = ('data/datasets/N24News/combine/cb_llm_annotation/clip_multimodal/outputs_concept_scoring/'
          'blue_checkpoints/clip/cavs/regression/sorted_macro_concepts_coverage_MJ_cb_llm_abs_all_LIG.pkl')
history, model = run_CBM(dataset='N24', dataset_type='CBLLM', combine_type='combine',
                         backbone='clip', concept_representation='importance', num_epochs=10,
                         kan_layer=True, leakage_loss=True, leakage_loss_activation='up',
                         import_concept_list=scores, concept_level=12)
```

Main arguments of `run_CBM` (full list in its docstring):

| Argument | Meaning |
|---|---|
| `dataset` | `'N24'`, `'CUB'`, `'agnews'`, `'dbpedia'` |
| `dataset_type` | `'CBLLM'` (continuous concept scores) or `'C3M'` (binary) |
| `combine_type` | `'combine'` (one concept vector for both modalities), `'concat'` (one per modality), `'text'`, `'image'` |
| `backbone` | `'clip'` or `'clip-large'` |
| `kan_layer` | KAN prediction head instead of a linear layer |
| `leakage_loss`, `leakage_loss_activation`, `lambda_XtoC_leakage` | leakage loss and its weight |
| `import_concept_list`, `concept_level` | concept subset produced by the CT-CBM heuristic (`.pkl`), and the level to use |
| `select_concepts`, `random_concepts`, `select_most_frequent` | other ways to choose the concept subset |
| `sequential`, `independant` | sequential / independent CBM training instead of joint |
| `loss_CBLLM` | concept loss: `'MSE'` (default) or `'cos_cubed'` |
| `lambda_XtoC`, `learning_rate`, `num_epochs` | usual training knobs |
| `load` | reload the checkpoint of a previous identical run instead of training |

`run_CBM` returns `(history, model)`; `history` holds the test accuracy, F1, concept
accuracy and leakage metrics reported in the paper.

### Hyper-parameter search

```bash
python src/hyperparam_search.py        # from the f-CBM/ folder
```

Grid over learning rate x epochs x `lambda_XtoC` on N24News with `clip-large`. Results
accumulate in `results/hyperparam_search.json`; configurations already present in that
file are skipped, so the search can be interrupted and resumed.



## Citation

```bibtex
@misc{moreau2026faithfulmultimodalconceptbottleneck,
      title={Towards Faithful Multimodal Concept Bottleneck Models}, 
      author={Pierre Moreau and Emeline Pineau Ferrand and Yann Choho and Benjamin Wong and Annabelle Blangero and Milan Bhan},
      year={2026},
      eprint={2603.13163},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.13163}, 
}
```

## Licence

The code is released under the [MIT licence](LICENSE): use it as you would the code of
any paper, with attribution. The datasets keep their own licences (see *Data*).
Third-party code included or adapted in `src/` is listed in [NOTICE](NOTICE).
