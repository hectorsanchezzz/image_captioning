# Conclusions

Resum del que hem après fent image captioning sobre Flickr8k. Els detalls (corbes, taules, exemples) estan a [resultats.md](resultats.md), [arquitectures.md](arquitectures.md) i [analisi_bias.md](analisi_bias.md).

## Evolució del projecte

Vam començar amb un **baseline ResNet18 + GRU** entrenat des de zero. Funciona com a referència i és fàcil d’entendre, però sobreajusta aviat (val loss puja des de l’epoch 4) i les captions són genèriques (*a man in a red shirt*, *in the grass*). Amb el protocol de test unificat (§3 de [resultats.md](resultats.md)) el baseline assoleix BLEU-1 ~0,63 i CLIP ~24,5 — lluny del ViT fine-tuned.

El salt important va ser el **fine-tuning de ViT–GPT2**: després d’un sweep curt a W&B vam entrenar 50 èpoques amb els millors hiperparàmetres. Al test, amb **el mateix protocol per a tots els models** (beam, BLEU corpus amb 5 refs, CLIPScore), el fine-tuned arriba a BLEU-1 **0,724** i CLIP **30,18**, molt per sobre del baseline (0,634 / 24,46) i del zero-shot (0,626 / 28,70). El millor checkpoint es guarda segons **`val/clip_score`**, no segons val loss.

## Què funciona i què no

**Funciona bé:**
- Fine-tuning progressiu (decoder primer, ViT descongelat a l’època 10, regularització moderada).
- Avaluar tots els models amb el **mateix protocol** al test (comparació justa).
- Beam search per mètriques (BLEU/CLIP altes, captions coherents en anglès).
- Re-pes a tokens femenins (`female_token_weight=2.5`, v3): redueix l’asimetria de gènere de 2,6× a ~1,1× sense castigar BLEU/CLIP de forma rellevant.

**Encara falla:**
- **Repetició de plantilles:** el model usa una fracció petita del vocabulari de referència i repeteix plantilles.
- **Biaix de gènere residual:** ~12–13 % de swaps en alguns casos; part prové del GPT-2 pretrained (ho vam veure comparant zero-shot i v1), no només del nostre entrenament.


## Biaix i decoding

L’anàlisi de biaixos (vegeu [analisi_bias.md](analisi_bias.md)) va mostrar dos problemes **separats**:

1. **Gènere** → cal intervenir a l’entrenament (CE loss amb pes 2,5×). Nucleus sampling sol no ho arregla.
2. **Diversitat lèxica** → nucleus sampling (`top_p=0.9`) sobre el mateix checkpoint v3 augmenta molt el vocabulari ús (+47 %), però costa ~0,10 BLEU i ~1 CLIP.

Per això tenim dos modes segons el que prioritzem:
- **v3 + beam:** benchmark acadèmic, màxim BLEU/CLIP.
- **v3 + nucleus:** captions més variades; acceptable si preferim text menys “de plantilla”.

## Limitacions del treball

- Flickr8k és petit; les conclusions del sweep (10 trials) són orientatives.
- CLIPScore és útil al train/val però no substitueix BLEU ni una lectura humana de les captions.
- La detecció de gènere als anàlisis va per paraules clau; no cobreix tots els casos (pronoms en context, gènere no binari, etc.).
- N mostra petita per alguns subgrups (p.ex. refs amb “woman” al test) → cal anar amb compte amb afirmacions molt fines.

## Conclusió final

En conjunt, el projecte confirma que per a captioning en un dataset petit com Flickr8k **no n’hi ha prou amb entrenar un CNN+RNN des de zero**, encara que ens serveixi per entendre el pipeline. El pas a **ViT–GPT2 fine-tunat** millora de forma clara les mètriques (BLEU-1 0,724, CLIP 30,18) i ens situa al nivell dels papers clàssics de Flickr8k i prop de treballs recents amb ensemble.

El model que considerem **final** per a mètriques de captioning és el **ViT–GPT2 fine-tuned** (run post-sweep) amb **beam search** a inferència: BLEU-1 **0,724**, CLIP **30,18** al test. Per al biaix de gènere, a més vam entrenar la variant **v3** (`female_token_weight=2.5`); vegeu [analisi_bias.md](analisi_bias.md). Nucleus sampling queda com a alternativa quan prioritzem diversitat lèxica.

A nivell de metodologia, el que més ens ha servit ha estat **triar bé la mètrica de validació** (CLIP en lloc de loss), **avaluar tots els models amb el mateix protocol** al test abans de comparar arquitectures, **comparar zero-shot amb fine-tuned** abans d’atribuir biaixos només al nostre entrenament, i **separar problemes** (gènere vs col·lapse lèxic) abans d’aplicar solucions. Això ens ha permès corregir el biaix amb un canvi relativament simple a la loss i abordar la diversitat amb decoding, sense barrejar-ho tot.

No hem tancat el projecte amb un model “resolt”: queden errors de gènere residuals, plantilles freqüents i limitacions del dataset i de les mètriques automàtiques. Tot i així, ens sembla que el treball mostra un camí raonable — de baseline interpretable a transformer fine-tunat, amb anàlisi crítica dels fallos — i deixa obertes millores concretes (més dades, avaluació humana, sweep del pes de debias) sense invalidar el que ja hem obtingut.
