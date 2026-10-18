# `test.py` / `test.ipynb` — avaluar el model de captions

Aquests fitxers carreguen el checkpoint, reconstrueixen el tokenizer com a `train.py` i avaluen el model **ViT–GPT2** (`nlpconnect/vit-gpt2-image-captioning`) sobre el split test de Flickr8k.

Poden fer:

- **Avaluació completa** del test (loss, CLIPScore, BLEU i, opcionalment, un CSV de prediccions).
- **Demo d’una imatge** amb `--image`, sense recórrer el dataset.

La generació per defecte és **beam search** (`num_beams=4`).

Per al baseline ResNet18+GRU existeix el script paral·lel `test_baseline.py` (mateix tipus d’avaluació, altre model i checkpoint).

## Abans de res

```bash
conda activate xnap-example
```

Cal tenir `data/flickr8k/` (`Images/`, `captions.txt`) i `checkpoints/best_model.pt` (generat amb `train.py`).

El checkpoint només desa pesos del **decoder**; l’encoder ViT es recarrega des de Hugging Face. El split test és el 12,5 % final d’imatges (`dataloaders/dataset.py`).

## Què calcula l’avaluació

**Loss:** cross-entropy amb teacher forcing sobre les tokens de referència de cada fila del batch.

**CLIPScore:** es genera una caption per imatge (beam search) i es mesura l’alineament imatge–text amb CLIP (cosinus, escala ×100) sobre tot el loader. A `train.py` la validació només usa els primers 50 batches per CLIP; al test es passa per complet.

**BLEU-1 … BLEU-4:** corpus BLEU amb una predicció per imatge (deduplicada per nom de fitxer) i les cinc referències de `captions.txt`. Mateix protocol que la validació de `train.py`.

El `DataLoader` té una fila per parell (imatge, caption) (~5000 files). La loss es calcula sobre totes; CLIP i BLEU compten una vegada per imatge.

## Script (`test.py`)

### Mode normal

```bash
python test.py
```

Imprimeix mètriques i, per defecte, cinc exemples aleatòries al terminal (seed 42).

### Exportar prediccions

```bash
python test.py --save_predictions outputs/test_predictions.csv
```

CSV amb columnes `image`, `reference`, `prediction`.

### Només mètriques

```bash
python test.py --no_samples
```

### Guardar graella d’exemples

```bash
python test.py --output_dir docs/figures
```

Escriu `test_samples.png` dins la carpeta indicada.

### Demo: una foto

```bash
python test.py --image data/flickr8k/Images/1000268201_693b08cb0e.jpg
```

### Zero-shot

Model pretrained, sense carregar checkpoint:

```bash
python test.py --no-checkpoint
```

### Nucleus sampling

Substitueix beam search per sampling:

```bash
python test.py --top_p 0.9 --temperature 1.0
```

### Paràmetres


| Flag                 | Defecte                     | Descripció                         |
| -------------------- | --------------------------- | ---------------------------------- |
| `--checkpoint`       | `checkpoints/best_model.pt` | Checkpoint del decoder             |
| `--data_dir`         | `data/flickr8k`             | Arrel Flickr8k                     |
| `--max_len`          | 30                          | Longitud màxima de caption         |
| `--num_beams`        | 4                           | Beam search (1 = greedy)           |
| `--batch_size`       | 32                          | Batch per l’avaluació              |
| `--seed`             | 42                          | Seed per triar exemples            |
| `--n_samples`        | 5                           | Quants exemples mostrar            |
| `--no_samples`       | —                           | Omet exemples al final             |
| `--save_predictions` | —                           | Camí CSV                           |
| `--output_dir`       | —                           | Carpeta per guardar PNG d’exemples |
| `--no-checkpoint`    | —                           | Sense fine-tuning                  |
| `--top_p`            | —                           | Nucleus sampling                   |
| `--temperature`      | 1.0                         | Temperatura (amb `--top_p`)        |


## Notebook (`test.ipynb`)

Obre el notebook des de l’arrel del repo i executa les cel·les en ordre.

Importa funcions des de `test.py` amb `importlib` (evita el conflicte amb el paquet estàndard `test` de Python). La cel·la de configuració copia `DEFAULT_CONFIG`; pots descomentar `save_predictions` per exportar el CSV.

Ordre de cel·les:

1. Config i device.
2. Tokenizer i model (`ckpt_info`: epoch, `val_clip_score`).
3. Avaluació del test (loss + CLIPScore). El notebook no imprimeix BLEU; el script sí.
4. Graella amb `plot_samples` (predicció al títol, referència sota).
5. Exemples en text amb `show_samples`.
6. Cel·la opcional: una imatge amb `generate_captions`.

El notebook mostra sis exemples per defecte (`n_samples: 6`); el script en mostra cinc.

## Baseline (`test_baseline.py`)

```bash
python test_baseline.py --checkpoint checkpoints/best_model_baseline.pt
```

Mateixa idea: avaluar loss, CLIPScore i BLEU al test, amb el vocabulari i el model del baseline.