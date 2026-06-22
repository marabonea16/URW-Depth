---
title: URW-Depth Demo
emoji: 🌧️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
---

# URW-Depth: Uncertainty-guided Robust Weather Depth

Demo interactiv pentru lucrarea de licență **URW-Depth**, construită peste arhitectura
[TinyDepth](https://www.sciencedirect.com/science/article/pii/S0952197624014714)
(TinyViT-5M, 5M parametri, estimare de adâncime monoculară auto-supervizată).

## Ce poți face în acest demo

1. **Demo live** — încarci o imagine proprie, alegi modelul și condiția meteo simulată
   (ceață/ploaie/zăpadă, severitate controlabilă), vezi adâncimea prezisă și harta de
   incertitudine ($\sigma$).
2. **Progresie cronologică** — vezi cum evoluează predicția de-a lungul celor 6 etape
   de dezvoltare ale arhitecturii, adăugate progresiv: baseline → cap de incertitudine
   → automascare ghidată → suprimare de caracteristici → URW-Depth-S2 → URW-Depth-HiRes.
3. **Robustețe la vreme** — grid cu severitate crescătoare a aceleiași condiții meteo,
   pentru un singur model, demonstrând degradarea controlată a performanței.

Toate modelele sunt încărcate live de pe
[mara-bonea-16/tinydepth-experiments](https://huggingface.co/mara-bonea-16/tinydepth-experiments).
