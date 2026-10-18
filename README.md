
# Image Captioning on Flickr8k

Projecte de l'assignatura **XNAP** (UAB): donada una imatge, el model genera una curta descripció en anglès.

Hem provat dues arquitectures:

1. **Baseline** — ResNet18 + GRU entrenat des de zero (`models/baseline.py`).
2. **ViT–GPT2** — fine-tuning del model preentrenat `nlpconnect/vit-gpt2-image-captioning` (`train.py`). És el que millor funciona: al test, BLEU-1 **0,724** i CLIPScore **30,18** (mateix protocol per a tots els models; detall a [docs/resultats.md](docs/resultats.md)).

També vam fer anàlisi de biaixos (gènere, diversitat lèxica) i algunes variants d'entrenament/decoding.

## Documentació

| Fitxer | Contingut |
|--------|-----------|
| [docs/test.md](docs/test.md) | Com avaluar amb `test.py` / `test.ipynb` |
| [docs/resultats.md](docs/resultats.md) | Resultats, corbes i comparativa entre models |
| [docs/arquitectures.md](docs/arquitectures.md) | Baseline vs ViT–GPT2 i variants |
| [docs/analisi_bias.md](docs/analisi_bias.md) | Biaixos i mitigació (debias, nucleus sampling) |
| [docs/conclusions.md](docs/conclusions.md) | Conclusions globals del projecte |

## Estructura del repo

```
├── train.py              # Entrenament ViT–GPT2 (model principal)
├── test.py               # Avaluació al test split
├── analyze.py            # Export CSV per-mostra (anàlisi de biaixos)
├── train_baseline.py     # Entrenament baseline ResNet+GRU
├── test_baseline.py      # Avaluació del baseline
├── sweep.yaml            # Config del sweep W&B per ViT–GPT2
├── models/baseline.py    # Model baseline
├── dataloaders/dataset.py
├── notebooks/            # bias_analysis, comparison, attention_viz
├── docs/                 # Informes i figures
└── environment.yml
```

Checkpoints (`checkpoints/`) i dades (`data/`) no es versionen; es generen en local.

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k): 8.091 imatges, 5 captions cadascuna.

```
data/flickr8k/
├── Images/
└── captions.txt
```

Split per imatge: 75 % train / 12,5 % val / 12,5 % test (`dataloaders/dataset.py`).

## Setup

```bash
conda env create --file environment.yml
conda activate xnap-example
```

El primer cop cal fer `wandb login` si vols loguejar entrenaments a [Weights & Biases](https://wandb.ai/site).

## Com executar-ho

### ViT–GPT2 (model principal)

```bash
# Entrenament
python train.py

# Sweep d'hiperparàmetres
wandb sweep sweep.yaml
wandb agent <sweep_id>

# Test (cal checkpoints/best_model.pt)
python test.py
```

### Baseline ResNet + GRU

```bash
python train_baseline.py
python test_baseline.py --checkpoint checkpoints/best_model_baseline.pt
```

### Anàlisi de biaixos

```bash
python analyze.py --checkpoint checkpoints/best_model.pt --output outputs/predictions.csv
jupyter lab notebooks/bias_analysis.ipynb
```

Veure [docs/test.md](docs/test.md) per flags (`--no-checkpoint`, `--top_p`, exportar CSV, etc.).

## Contribuïdors

- Aleix Artal — `1702608@autonoma.cat`
- Jan Munts — `1704562@autonoma.cat`
- Hèctor Sánchez — `1704248@autonoma.cat`

Xarxes Neuronals i Aprenentatge Profund  
Grau de Matemàtica Computacional i Analítica de Dades, UAB, 2026
