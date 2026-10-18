# Arquitectures

Al llarg del projecte hem passat per dues arquitectures principals: un **baseline** CNN+RNN entrenat des de zero i el **fine-tuning** d’un ViT–GPT2 preentrenat (el model final). Més endavant vam afegir variants d’entrenament i decoding sobre el ViT sense canviar la xarxa. Els números de cada fase estan a [resultats.md](resultats.md); aquí expliquem com està muntat cada cosa i com ho relacionem amb el que vam observar.

---

## 1. Baseline: ResNet18 + GRU

**Codi:** [`models/baseline.py`](../models/baseline.py), entrenament a [`train_baseline.py`](../train_baseline.py).

És l’arquitectura clàssica encoder–decoder per captions:

```
Imatge → ResNet18 (pool global) → vector 512 → Linear → h₀ de la GRU
Token → Embedding → GRU (2 capes) → Linear → logits
```

Detalls rellevants:

- ResNet18 preentrenat a ImageNet; **congelat** durant tot l’entrenament.
- Pool global: cada imatge es resumeix en **un sol vector** de 512 dimensions.
- Aquest vector projectat inicialitza l’estat ocult de la GRU; després el decoder genera token a token amb teacher forcing.
- Vocabulari propi amb `<START>`, `<STOP>`, `<PAD>`, `<UNK>` (`utils/vocab.py`); ~15,8 M paràmetres entrenables.

### Què vam veure als resultats

Al [resultats.md](resultats.md) es veu que el model aprèn (train loss baixa) però la **validació empitjora** passada l’epoch 4: sobreajustament. Amb el protocol antic (greedy) el BLEU corpus era ~0,05; amb el protocol unificat al test, el baseline puja a BLEU-1 ~0,63 però queda per sota del ViT fine-tuned.

Interpretació arquitectural: la GRU **només “veu” la imatge al primer pas**. Quan ha de descriure diversos objectes o detalls fina, ha de recollir-ho tot des de `h₀`. Això encaixa amb errors on agafa un sol motiu de l’escena o inventa. No és l’únic factor (dataset petit, vocabulari limitat), però és el límit estructural més clar del disseny.

---

## 2. ViT–GPT2 fine-tuned (model principal)

**Codi:** [`train.py`](../train.py), avaluació a [`test.py`](../test.py). Model base: `nlpconnect/vit-gpt2-image-captioning`.

```
Imatge → ViT (encoder) → seqüència de tokens visuals
                              ↕ cross-attention
Text  → GPT-2 (decoder) → logits del següent token
```

Canvis respecte al baseline:

| | Baseline | ViT–GPT2 |
|--|----------|----------|
| Encoder | ResNet18, 512-D global | ViT, representació seqüencial |
| Decoder | GRU 2 capes | GPT-2 (12 capes) |
| Atenció imatge | No (només `h₀`) | Cross-attention a cada capa del decoder |
| Vocabulari | ~4600 paraules pròpies | Tokenizer BPE de GPT-2 |
| Entrenament | Des de zero | Fine-tuning sobre pesos preentrenats |

### Per què vam canviar

Els resultats del baseline (BLEU baix, sobreajustament, captions genèriques) ens van fer buscar un model amb **més capacitat visual i lingüística** sense entrenar tot des de zero. El ViT–GPT2 ja porta encoder i decoder preentrenats i cross-attention entre imatge i text — just el que al baseline ens faltava estructuralment.

### Estratègia d’entrenament

No entrenem tot de cop:

1. **Encoder congelat** al principi; fine-tuning del decoder GPT-2.
2. **8 capes inferiors del decoder congelades** (`frozen_decoder_layers=8`).
3. A l’**època 10**, descongelem l’encoder ViT amb lr 10× més petit que el del decoder.
4. **LLRD** i augmentació lleugera a train; label smoothing a la loss.
5. Early stopping segons **`val/clip_score`** (paciència 15 en el run post-sweep).

### Què vam veure als resultats

Al test amb protocol unificat, el fine-tuned assoleix BLEU-1 **0,724** i CLIPScore **30,18** (baseline 0,634 / 24,46; zero-shot 0,626 / 28,70). A validació, `val/clip_score` supera **30** amb pics ~**30,2**; val loss baixa fins ~**3,41**. Detall a [resultats.md](resultats.md).

Limitacions que **no** soluciona sol l’arquitectura: col·lapse lèxic (plantilles repetides), biaix de gènere heretat del GPT-2 pretrained. Això ho vam abordar amb intervencions addicionals (secció següent), no canviant la topologia de la xarxa.

---

## 3. Variants sobre el mateix ViT–GPT2

Aquestes execucions **no canvien l’arquitectura** (mateix ViT, mateix GPT-2). Són canvis de loss o de decoding; les xifres estan resumides a [resultats.md](resultats.md) i el detall de biaixos a [analisi_bias.md](analisi_bias.md).

### v2 / v3 — re-pes a tokens femenins

A `train.py`, la `CrossEntropyLoss` pot ponderar més tokens com *woman*, *girl*, *she*, etc. (`female_token_weight`).

- **v2 (1,8×):** redueix el biaix de gènere però no prou.
- **v3 (2,5×):** asimetria dona↔home de 2,6× a ~1,1×, sense penalitzar BLEU/CLIP de forma rellevant.

És una intervenció d’**entrenament**, no un canvi d’encoder ni de decoder.

### Beam vs nucleus sampling

A **inferència** (`test.py`, `analyze.py`):

- **Beam search** (`num_beams=4`): per defecte; maximitza BLEU i CLIP.
- **Nucleus sampling** (`top_p=0.9`): mateix checkpoint v3, més diversitat lèxica (+47 % vocabulari ús) però ~−0,10 BLEU i ~−1 CLIP.

Dos modes de generar text sobre el mateix model; útil segons si prioritzem mètriques o captions més variades.

---

## 4. Resum de l’evolució

| Fase | Què és | Estat al repo | Motiu del canvi següent |
|------|--------|---------------|-------------------------|
| ResNet18 + GRU | CNN global + GRU | Implementat | BLEU baix, sobreajustament, imatge només al `h₀` |
| ViT–GPT2 | Fine-tuning encoder–decoder | Implementat (model final) | Resultats molt millors en CLIP i BLEU |
| v3 + debias / nucleus | Variants train/decode | Implementat | Biaix de gènere i diversitat lèxica |

En una frase: vam començar amb un baseline petit i interpretable, vam veure que comprimir tota la imatge en un vector no n’hi havia prou, i vam passar a un transformer preentrenat amb cross-attention; després vam afinar loss i decoding sobre aquesta base.

Per com executar cada part, [test.md](test.md) (avaluació) i el README (entrades d’entrenament).
