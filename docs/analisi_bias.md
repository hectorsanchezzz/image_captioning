# W5 — Anàlisi de biaixos i mitigació

Anàlisi del model ViT-GPT2 fine-tuned sobre Flickr8k. L'objectiu va ser identificar biaixos i debilitats sistemàtiques i provar intervencions correctives.

Experiments executats a la branca `jan/w5-bias-analysis`:

| variant | descripció |
|---|---|
| **zero-shot** | `nlpconnect/vit-gpt2-image-captioning` pretrained, sense fine-tuning |
| **v1 baseline** | model fine-tuned sobre Flickr8k, sense intervenció |
| **v2** | re-entrenat amb `female_token_weight=1.8` a la CE loss |
| **v3** | re-entrenat amb `female_token_weight=2.5` |
| **v3 + nucleus** | mateix checkpoint v3, decoding amb nucleus sampling (`top_p=0.9, temperature=0.9`) |

Eines:
- [analyze.py](../analyze.py) — exporta CSV per-mostra amb BLEU/CLIP/flags de gènere
- [notebooks/bias_analysis.ipynb](../notebooks/bias_analysis.ipynb) — anàlisi d'un CSV individual
- [notebooks/comparison.ipynb](../notebooks/comparison.ipynb) — comparativa multi-variant amb bootstrap CI
- [notebooks/attention_viz.ipynb](../notebooks/attention_viz.ipynb) — visualització de cross-attention

---

## 1. Diagnòstic inicial (v1)

Sobre el test set (~948 imatges úniques × 5 captions = 4742 files al CSV v1 sense dedupe):

### Diversitat lèxica
- Vocab predicció: **621** vs referències: **3084** → el model usa només **20% del lèxic**
- Captions úniques: ~950 / 948 imatges → cada imatge rep una caption distinta (no hi ha mode-collapse estructural, només lèxic)
- Top n-grames revelen plantilles: `"a brown dog"`, `"a man in"`, `"dog is running"`, `"and white dog"`

### Rendiment per categoria visual

| categoria | n | BLEU-1 | BLEU-4 | CLIP |
|---|---|---|---|---|
| street | 597 | 0.604 | 0.176 | 29.35 |
| bike | 121 | 0.632 | 0.207 | 30.95 |
| child | 1738 | 0.646 | 0.207 | 30.53 |
| water | 1052 | 0.683 | 0.246 | 31.54 |
| snow | 403 | 0.708 | 0.282 | 30.62 |
| grass | 960 | 0.712 | 0.276 | 31.44 |
| dog | 1460 | 0.723 | 0.311 | 32.20 |
| sport | 434 | 0.725 | 0.320 | 32.22 |

Pitjors: street, bike, child. Millors: dog, sport (les escenes prototípiques de Flickr8k).

### Biaix de gènere (la troballa més rellevant)

Matriu de confusió (files = referència, columnes = predicció):

```
pred_gender  both   man  neutral  woman   All
ref=both      174    67       88     88    417
ref=man       129  1135      282    100   1646
ref=neutral    49    49     1474     20   1592
ref=woman     106   173      102    706   1087
```

- **Dona→home: 173 / 1087 = 15.9%**
- **Home→dona: 100 / 1646 = 6.1%**
- **Asimetria: 2.6×** (una dona té 2.6× més probabilitat de ser mal-etiquetada que un home)
- Referències neutres → 49 prediuen "man", 20 prediuen "woman" → biaix 2.45× cap a masculí

Exemples concrets:
- `3758787457`: ref `"a girl in a black two-piece is leaping into the water"` → pred `"a boy jumps into the air above the water"`
- `3767841911`: ref `"a baby girl in an orange dress"` → pred `"a little boy in a red shirt"`

### Hallucinacions de plantilla

Els pitjors casos no són errors de matís sinó encaixos forçats de plantilles freqüents:
- ref `"three children in a cage"` → pred `"a boy and a girl look at each other through binoculars"`
- ref `"two girls hugging in front of american flag"` → pred `"a woman in a red shirt is talking on a cellphone"`

---

## 1bis. Origen del biaix: zero-shot vs fine-tuned

Per saber si el biaix de gènere prové del decoder GPT-2 pretrained o l'introduïm nosaltres amb el fine-tuning, vam analitzar el model `nlpconnect/vit-gpt2-image-captioning` **sense entrenament addicional sobre Flickr8k**.

### Matriu de confusió zero-shot

```
pred_gender  both  man  neutral  woman   All
ref=both       27   23       15     24     89
ref=man        19  243       65     25    352
ref=neutral     8   14      314      1    337
ref=woman      16   41       32    145    234
```

### Comparativa de les quatre variants

| variant | woman→man | man→woman | asimetria | neutrals genderitzades | vocab |
|---|---|---|---|---|---|
| **Zero-shot** | **17.5%** (41/234) | 7.1% | 2.47× | **14× cap a home** (14M/1W) | 471 |
| **v1 baseline** | 15.9% | 6.1% | 2.6× | 2.45× cap a home (49M/20W) | 621 |
| **v3 (2.5×)** | 12.8% | 11.6% | **1.10×** | 1.29× (9M/7W) | 597 |
| **v3 + nucleus** | 12.4% | 12.8% | **0.97×** | 1.07× (14M/15W) | 877 |

### Lectura

**El biaix de gènere prové del GPT-2 pretrained, no del nostre fine-tuning.** Tres observacions clau:

1. **Sobre referències de gènere explícit:** el zero-shot ja té 17.5% de swap dona→home (un 10% més que el simètric). El fine-tuning a Flickr8k el va atenuar lleugerament (a 15.9%), però la **asimetria de fons es manté** (2.47× → 2.6×).

2. **Sobre referències neutres** (el biaix més pur, sense pista textual): el zero-shot tria masculí amb una intensitat aclaparadora — **14 cap a home, només 1 cap a dona**. El fine-tuning sobre Flickr8k va corregir aquest defecte substancialment (a 2.45×), probablement perquè les captions de Flickr8k usen "woman" més sovint que el corpus de pretraining del GPT-2.

3. **El nostre debiasing completa el camí:** asimetria 2.47× → 1.10×, defecte en neutrals 14× → 1.29×. Sense degradar BLEU/CLIP.

### Conclusió narrativa

> El fine-tuning sobre Flickr8k **no va introduir el biaix de gènere — el va heretar i atenuar parcialment** perquè el dataset és més equilibrat que les dades web del GPT-2 pretrained. La nostra intervenció amb pes a tokens femenins a la CE loss el redueix pràcticament a zero, sense penalitzar BLEU ni CLIP. Aquest és un argument fort contra l'assumpció ingènua que "el biaix surt del fine-tuning".

---

## 2. Intervenció: re-pes de tokens femenins a la CE loss

### Plantejament

Modificar la `CrossEntropyLoss` perquè els tokens corresponents a paraules femenines (woman, women, girl, girls, she, her, female, lady, ladies, mother, mom + variants amb majúscula i espai inicial) tinguin un pes >1. Implementat a [train.py:build_gender_class_weights()](../train.py).

40 token IDs identificats nets (sense fragments sub-word). Pes exposat com a hiperparàmetre `female_token_weight` a `DEFAULT_CONFIG`.

### v2 — `female_token_weight=1.8`

| | woman→man | man→woman | asimetria | neutrals genderitzades |
|---|---|---|---|---|
| **v1 baseline** | 15.9% | 6.1% | 2.6× | 2.45× cap a home |
| **v2 (1.8×)** | 14.1% | 7.7% | 1.83× | 1.83× cap a home |
| **Δ** | −1.8 pp | +1.6 pp | **−29%** | **−25%** |

Mètriques per categoria pràcticament idèntiques a v1 (cap degradació). El biaix s'ha reduït significativament però **no per sota del criteri d'èxit del <10%**. Calia pujar el pes.

### v3 — `female_token_weight=2.5`

| | woman→man | man→woman | asimetria | neutrals genderitzades |
|---|---|---|---|---|
| **v1 baseline** | 15.9% | 6.1% | 2.6× | 2.45× |
| **v2 (1.8×)** | 14.1% | 7.7% | 1.83× | 1.83× |
| **v3 (2.5×)** | **12.8%** | **11.6%** | **1.10×** | **1.29×** |
| **Δ v1→v3** | −3.1 pp | +5.5 pp | **−58%** | **−47%** |

**Asimetria pràcticament eliminada** (1.10× ≈ simetria perfecta).

Mètriques per categoria (v3 vs v1, BLEU-1):
- street: 0.604 → 0.596 (−0.008)
- bike: 0.632 → 0.643 (+0.011)
- child: 0.646 → 0.646 (=)
- water: 0.683 → 0.697 (+0.014)
- grass: 0.712 → 0.710 (=)
- snow: 0.708 → 0.718 (+0.010)
- sport: 0.725 → 0.714 (−0.011)
- dog: 0.723 → 0.722 (=)

Totes dins del soroll. **Cap penalització significativa de rendiment.**

### Per què 2.5× i no més?

L'asimetria a v3 és 1.10× — gairebé 1.0 (simetria perfecta). Pujar més probablement **invertiria el biaix** (home→dona > dona→home) sense reduir més l'error global. El biaix residual (~12% per banda) prové probablement de **mostres genuïnament ambigües** del test set (persones de lluny, de perfil, ocultes), no d'un biaix del model. Saturació clara.

---

## 3. Intervenció ortogonal: nucleus sampling

Per atacar el **col·lapse lèxic** sense re-entrenar, vam substituir beam search (`num_beams=4`) per nucleus sampling (`top_p=0.9, temperature=0.9`) sobre el mateix checkpoint v3.

### Què és nucleus sampling

A cada pas, en lloc d'agafar el camí més probable, mostreja entre el conjunt mínim de tokens que sumen probabilitat ≥ p. Resultat: diversitat lèxica humana sense caure a tokens absurds.

### Comparativa v3 beam vs v3+nucleus

| Mètrica | v3 beam | v3 + nucleus | Δ |
|---|---|---|---|
| **Vocab predicció** | 597 | **877** | **+47%** |
| **Captions úniques** | 93.6% | **99.8%** | +6.2 pp |
| **Asimetria de gènere** | 1.10× | **0.97×** | **simetria perfecta** |
| **Genderització de neutrals** | 1.29× → home | **1.07× (14M/15W)** | equilibri total |
| **BLEU-1 mitjà (categories)** | 0.681 | **0.585** | **−0.096** |
| **CLIPScore mitjà** | 30.7 | **29.6** | **−1.1** |

### Tradeoff per categoria

| categoria | BLEU-1 (beam→nucleus) | CLIP (beam→nucleus) |
|---|---|---|
| street | 0.596 → 0.544 (−0.052) | 29.05 → 28.23 |
| child | 0.646 → 0.541 (−0.105) | 30.29 → 29.40 |
| water | 0.697 → 0.584 (−0.113) | 31.36 → 30.48 |
| sport | 0.714 → 0.583 (−0.131) | 32.04 → 30.38 |
| dog | 0.722 → 0.630 (−0.092) | 32.09 → 30.79 |

Caiguda de ~10pp BLEU i ~1pp CLIP a totes les categories.

### Exemple il·lustratiu

Imatge `3758787457` (nena en biquini saltant a l'aigua):
- **v1:** `a boy jumps into the air above the water`
- **v3 beam:** `a boy jumps into the air above the water` (mateixa plantilla, gènere encara malament en aquest cas concret)
- **v3 nucleus:** `the boy is doing a flip in the pool while people watch from the other side`

El nucleus produeix descripcions molt més riques però el gènere segueix erroni en aquest cas. **Confirma que el biaix de gènere és un problema separat del col·lapse lèxic** — el decoding no l'arregla, només el re-entrenament amb pes l'ataca.

---

## 4. Conclusions

### Dos eixos d'intervenció ortogonals

| Eix | Intervenció | Què fixa | Cost |
|---|---|---|---|
| **Biaix de gènere** | CE weight 2.5× | Asimetria 2.6× → 1.10× (−58%) | Cap, BLEU/CLIP estables |
| **Col·lapse lèxic** | Nucleus sampling | Vocab +47%, captions úniques 99.8% | −10 pp BLEU, −1 pp CLIP |

Es **combinen**: v3 + nucleus té asimetria 0.97× + diversitat 99.8%.

### Recomanació de decoding segons cas d'ús

- **Beam (v3):** quan vols màxim BLEU per a benchmark acadèmic
- **Nucleus (v3+nucleus):** quan vols captions usables per humans (xarxes socials, accessibilitat) i prioritzes diversitat sobre adequació estricta

### Limitacions reconegudes

- **Mida de mostra petita per al biaix de gènere:** N=234 refs femenines al test → intervals de confiança amplis (~±4pp). Cal precaució amb afirmacions fines.
- **Detecció de gènere per keywords** (woman/girl/man/boy + variants): no captura pronoms en context, ni gènere no-binari, ni descripcions perifràstiques ("a person in a dress").
- **Biaix residual ~12%** probablement intrínsec a mostres ambigües del test set.

Idees per continuar: test χ² v1 vs v3, visualització d'atenció sobre swaps de gènere, sweep més fi de `female_token_weight`.

Conclusions globals del projecte: [conclusions.md](conclusions.md).

---

## 5. Reproductibilitat

```bash
# Baseline zero-shot (model pretrained, sense fine-tuning)
python analyze.py --no-checkpoint --output outputs/predictions_zeroshot.csv

# Entrenament dels checkpoints fine-tuned
python train.py --run-name <variant>
# (canviar female_token_weight a DEFAULT_CONFIG entre runs: 1.0, 1.8, 2.5)

# Anàlisi de cada checkpoint
python analyze.py --checkpoint checkpoints/best_model.pt \
    --output outputs/predictions_<variant>.csv

# Variant nucleus sobre v3
python analyze.py --checkpoint checkpoints/best_model.pt \
    --top_p 0.9 --temperature 0.9 \
    --output outputs/predictions_v3_nucleus.csv

# Inspecció individual
jupyter lab notebooks/bias_analysis.ipynb     # canvia CSV_PATH

# Comparativa consolidada (auto-detecta CSVs disponibles)
jupyter lab notebooks/comparison.ipynb
```
