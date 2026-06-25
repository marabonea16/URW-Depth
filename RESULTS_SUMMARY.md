# URW-Depth — Rezultate consolidate (referință pentru documentație)

Toate rezultatele sunt logate și verificabile în wandb, proiect `tinydepth`
(`marabonea16-universitatea-politehnica-timi-oara`). Acest fișier e doar un
rezumat de referință rapidă.

**Modelul final ales pentru prezentare: `URW-Depth-S2` (sigma colapsat).**

## Tabel 1 — KITTI Eigen Clean (fără TTA)

| Model | abs_rel ↓ | sq_rel ↓ | rmse ↓ | rmse_log ↓ | a1 ↑ | a2 ↑ | a3 ↑ |
|---|---|---|---|---|---|---|---|
| TinyDepth-b6 | 0.0991 | 0.7082 | 4.3241 | 0.1737 | 0.9007 | 0.9672 | 0.9844 |
| +Cap incertitudine | 0.1023 | 0.7147 | 4.3848 | 0.1752 | 0.8939 | 0.9661 | 0.9846 |
| +Automascare | 0.0977 | 0.6713 | 4.2897 | 0.1718 | 0.9014 | 0.9677 | 0.9849 |
| +Suprimare caracteristici | 0.0991 | 0.6846 | 4.2963 | 0.1732 | 0.9013 | 0.9672 | 0.9844 |
| **URW-Depth-S2 (final)** | **0.0976** | **0.6837** | 4.3235 | 0.1727 | **0.9010** | 0.9672 | 0.9844 |
| URW-Depth-HiRes | 0.1122 | 0.7703 | 4.6746 | 0.1866 | 0.8734 | 0.9620 | 0.9831 |

## Tabel 2 — KITTI Eigen Clean cu TTA (gradient-based, --use_tta)

| Model | abs_rel ↓ | sq_rel ↓ | rmse ↓ | rmse_log ↓ | a1 ↑ | a2 ↑ | a3 ↑ |
|---|---|---|---|---|---|---|---|
| +Cap incertitudine | 0.1107 | 0.8062 | 4.5320 | 0.1858 | 0.8821 | 0.9618 | 0.9824 |
| +Automascare | 0.1060 | 0.7620 | 4.4420 | 0.1824 | 0.8888 | 0.9636 | 0.9830 |
| +Suprimare caracteristici | 0.1074 | 0.7825 | 4.4500 | 0.1840 | 0.8896 | 0.9629 | 0.9824 |
| **URW-Depth-S2** | **0.1055** | 0.7830 | 4.4590 | 0.1834 | 0.8901 | 0.9630 | 0.9823 |

> TTA degradează ușor performanța pe date curate la toate modelele (cauza:
> colapsul sigma, vezi secțiunea de analiză critică).

## Tabel 3 — KITTI-C (18 corupții × 5 severități)

| Model | abs_rel ↓ | sq_rel ↓ | rmse ↓ | rmse_log ↓ | a1 ↑ | a2 ↑ | a3 ↑ |
|---|---|---|---|---|---|---|---|
| TinyDepth-b6 | 0.2342 | 2.1846 | 7.2676 | 0.3245 | 0.6497 | 0.8357 | 0.9192 |
| +Suprimare caracteristici | 0.2237 | 1.9914 | 7.1920 | 0.3175 | 0.6614 | 0.8432 | 0.9241 |
| **URW-Depth-S2** | **0.2157** | **1.9228** | **7.0051** | **0.3071** | **0.6781** | **0.8525** | **0.9278** |
| URW-Depth-HiRes | 0.2863 | 2.6918 | 8.9794 | 0.3991 | 0.5194 | 0.7737 | 0.8932 |

## Tabel 4 — Vreme simulată (fără TTA), abs_rel ↓ / a1 ↑

| Model | Ceață | Ploaie | Zăpadă | Medie abs_rel |
|---|---|---|---|---|
| TinyDepth-b6 | 0.1228 / 0.8668 | 0.1142 / 0.8759 | 0.1090 / 0.8848 | 0.1153 |
| +Cap incertitudine | 0.1212 / 0.8667 | 0.1217 / 0.8629 | 0.1118 / 0.8788 | 0.1182 |
| +Automascare | 0.1172 / 0.8712 | 0.1149 / 0.8740 | 0.1068 / 0.8874 | 0.1130 |
| +Suprimare caracteristici | 0.1090 / 0.8862 | 0.1082 / 0.8869 | 0.1077 / 0.8884 | 0.1083 |
| **URW-Depth-S2** | 0.1084 / 0.8844 | **0.1070** / 0.8863 | **0.1064** / 0.8876 | **0.1073** |
| URW-Depth-HiRes | 0.1330 / 0.8324 | 0.1233 / 0.8538 | 0.1240 / 0.8524 | 0.1268 |

## Tabel 5 — Vreme simulată cu TTA, abs_rel ↓

| Model | Ceață | Ploaie | Zăpadă |
|---|---|---|---|
| +Cap incertitudine | 0.1205 | 0.1213 | 0.1117 |
| +Automascare | 0.1170 | 0.1159 | 0.1066 |
| +Suprimare caracteristici | 0.1090 | 0.1083 | 0.1076 |
| **URW-Depth-S2** | **0.1080** | **0.1068** | **0.1059** |

## Tabel 6 — Sinteză câștiguri URW-Depth-S2 vs baseline

| Benchmark | Metrică | b6 | URW-Depth-S2 | Câștig |
|---|---|---|---|---|
| KITTI Eigen | abs_rel | 0.0991 | 0.0976 | −1.5% |
| KITTI-C | abs_rel | 0.2342 | 0.2157 | −7.9% |
| Weather medie | abs_rel | 0.1153 | 0.1073 | −6.9% |

## Tabel 7 — NYU Depth v2 (zero-shot, fără fine-tuning)

Protocol: 654 imagini test oficiale, eigen crop `[45:471, 41:601]`, max_depth=10m,
scalare mediană (modelele sunt antrenate exclusiv pe KITTI, testate fără
fine-tuning pe NYU — generalizare cross-domeniu, ca în paper-ul TinyDepth).

| Model | abs_rel ↓ | a1 ↑ |
|---|---|---|
| TinyDepth-b6 | 0.3151 | 0.5118 |
| +Cap incertitudine | 0.2995 | 0.5316 |
| +Automascare | **0.2821** | **0.5660** |
| +Suprimare caracteristici | 0.2995 | 0.5392 |
| URW-Depth-S2 | 0.3171 | 0.5120 |
| URW-Depth-HiRes | 0.2771 | 0.5731 |

> Observație: pe NYU (generalizare cross-domeniu), `URW-Depth-HiRes` și
> `+Automascare` generalizează cel mai bine, nu `URW-Depth-S2` (care e
> optimizat specific pentru KITTI+vreme). Rezoluția mai mare (HiRes) ajută
> probabil la generalizare cross-domeniu, opus față de ce am văzut pe
> KITTI-C/weather (unde HiRes era cel mai slab).

---

# Investigația de calibrare a incertitudinii (Fix1–Fix7, Diag1–Diag9)

**Context:** sigma (capul de incertitudine) colapsează la ~0 în toate modelele
de mai sus (`raw_uncert ≈ -27`). Mecanismul de suprimare a caracteristicilor
și TTA-ul ghidat de incertitudine sunt practic inactive în modelele curente.

## Cauza colapsului (confirmată)

`loss = (1-sigma.detach())·photo·mask + λ·sigma` — sigma primește gradient
doar descendent (din `λ·sigma`), niciun gradient ascendent din partea
fotometrică (detașată). Colaps inevitabil spre 0.

## Cronologia tentativelor de fix

| # | Configurație | Rezultat KITTI clean | Sigma | Verdict |
|---|---|---|---|---|
| Fix1 | mască `sigma<0.8` + MSE calib w=1.0 | 0.1462 | stabil ~0.4 | Regresie severă |
| Fix2 | fără mască, w=1.0, fără seed | 0.163→0.182 (degradare) | stabil | Regresie progresivă |
| Fix3 | + `detach()` izolare backbone | 0.443 (catastrofal) | stabil | Colaps degenerat |
| Diag1 | w=0.3, fără augmentare | **0.098** (=baseline) | colapsează lent | Depth OK, sigma nu rezistă |
| Diag2 | w=1.0, fără seed | 0.557 | stabil | Catastrofal |
| Diag3 | w=1.0, seed=42 | 0.443 (identic Fix3!) | stabil | Confirmă determinism, nu ghinion |
| Diag4 | w=0.5, seed=42 | 0.124 | stabil ~0.10-0.11 | Compromis moderat |
| **Fix4** | w=0.5, 15 epoci complete, augmentare | **0.112** | **stabil ~0.05-0.09** | **Cel mai bun compromis funcțional** |
| Diag5 | w=0.5, fără gating suprimare | 0.123 | stabil | Gating NU e cauza unică |
| Diag6 | w=0.5, reset Adam selectiv | 0.111 | stabil | Adam NU e cauza |
| Diag7 | fără reset, calib=0 (control) | **0.099** | colapsat (normal) | Confirmă fine-tuning normal e OK |
| Diag8 | reset=True, calib=0 | **0.098** | colapsat (normal) | Confirmă reset_uncert_head NU e cauza |
| Fix7 | w=0.3, 15 epoci, augmentare | 0.099 (epoca 0) apoi colaps lent | colapsează | Reproduce Diag1 la scară completă |
| Diag9 | w=0.3 + uncert_weight=0.1 | **0.443** (catastrofal) | stabil ~0.45 (prea mare!) | Suprimare excesivă (55% atenuare) |

## Rezultate complete Fix4 (modelul cu sigma calibrat funcțional)

| Benchmark | URW-Depth-S2 (original) | Fix4 (calibrat) | Δ |
|---|---|---|---|
| KITTI clean | 0.0976 | 0.112 | +15% |
| KITTI-C | 0.2157 | **0.1566** | **−27% (mult mai bun)** |
| Weather medie | ~0.1073 | ~0.1208 | +13% |
| TTA (toate) | marginal | marginal | fără diferență |

## Concluzia mecanicistă finală

Suprima de caracteristici (`d_refined = d×(1-sigma)`) e un mecanism
**substractiv**: elimină informație proporțional cu magnitudinea medie a
sigma. Pe date curate/deja-cunoscute (vreme antrenată direct), nu există
"informație coruptă" de eliminat — doar regiuni natural dificile dar utile
(ocluzii, textură slabă, obiecte îndepărtate) — deci suprimarea costă fără
beneficiu compensator. Pe corupții diverse/nevăzute (KITTI-C), suprimarea
elimină informație genuin nesigură — beneficiu real (−27%).

**Testat și exclus explicit ca cauză unică a degradării:** masca dură,
gating-ul de suprimare, starea optimizatorului Adam, `reset_uncert_head`,
varianța de seed aleator. Cauza e magnitudinea sigma în echilibru (controlată
de `calib_weight` și `uncert_weight` împreună), care scalează direct
intensitatea suprimei — o limitare structurală a mecanismului, nu un bug
izolat reparabil prin hyperparametri.

**Decizie finală:** `URW-Depth-S2` (sigma colapsat) rămâne modelul prezentat
pentru toate rezultatele numerice (câștigă pe KITTI, prioritatea declarată).
Investigația de calibrare e prezentată ca secțiune de analiză critică/limitări
— 9 experimente de diagnostic riguros, cauză mecanicistă identificată complet.
