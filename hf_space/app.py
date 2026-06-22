"""
URW-Depth — demo interactiv pe HuggingFace Spaces.

Trei tab-uri:
  1. Demo live: incarca o imagine, alege model + tip/severitate vreme, vezi adancimea + incertitudinea
  2. Progresie cronologica: arata cum evolueaza predictia pe parcursul celor 6 etape de ablatie
  3. Robustete la vreme: grid cu severitate controlabila (ceata/ploaie/zapada)
"""
import os
import gradio as gr
from PIL import Image

from model_utils import run_inference, ABLATION_MODELS
from weather_aug import apply_weather

MODEL_CHOICES = [m[0] for m in ABLATION_MODELS]
MODEL_LOOKUP = {m[0]: m for m in ABLATION_MODELS}

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def _ex(fname):
    return os.path.join(EXAMPLES_DIR, fname)


def _static_ablation_gallery(scene):
    items = []
    for idx, (name, _, _, _) in enumerate(ABLATION_MODELS):
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        items.append((_ex(f"{scene}_ablation_{idx}_{safe}.png"), name))
    return items


def _static_weather_gallery(weather_type):
    sev_tags = [("clean", "curat"), ("sev25", "severitate 0.25"),
                ("sev45", "severitate 0.45"), ("sev65", "severitate 0.65"),
                ("sev85", "severitate 0.85")]
    return [(_ex(f"weather_{weather_type}_{tag}_depth.png"), label) for tag, label in sev_tags]


def demo_live(image, model_name, weather_type, severity):
    if image is None:
        return None, None, "Incarca o imagine."
    _, model_dir, weights_dir, use_fs = MODEL_LOOKUP[model_name]

    img = image.convert("RGB")
    if weather_type != "none":
        img = apply_weather(img, weather_type, severity)

    depth_color, uncert_color, _ = run_inference(img, model_dir, weights_dir, use_fs)
    info = f"Model: **{model_name}**"
    if weather_type != "none":
        info += f" | Vreme: **{weather_type}** (severitate {severity:.2f})"
    return Image.fromarray(depth_color), (
        Image.fromarray(uncert_color) if uncert_color is not None else None
    ), info


def ablation_progression(image, weather_type, severity):
    if image is None:
        return [None] * len(ABLATION_MODELS)
    img = image.convert("RGB")
    if weather_type != "none":
        img = apply_weather(img, weather_type, severity)

    results = []
    for name, model_dir, weights_dir, use_fs in ABLATION_MODELS:
        depth_color, _, _ = run_inference(img, model_dir, weights_dir, use_fs)
        results.append((Image.fromarray(depth_color), name))
    return results


def weather_severity_grid(image, model_name, weather_type):
    if image is None:
        return []
    _, model_dir, weights_dir, use_fs = MODEL_LOOKUP[model_name]
    severities = [0.0, 0.25, 0.45, 0.65, 0.85]
    results = []
    for sev in severities:
        img = image.convert("RGB")
        if sev > 0:
            img = apply_weather(img, weather_type, sev)
        depth_color, _, _ = run_inference(img, model_dir, weights_dir, use_fs)
        label = "curat (severitate 0)" if sev == 0 else f"severitate {sev:.2f}"
        results.append((Image.fromarray(depth_color), label))
    return results


with gr.Blocks(title="URW-Depth Demo") as demo:
    gr.Markdown("""
    # URW-Depth: Estimare de Adancime Monoculara Robusta la Conditii Meteo

    Demo interactiv pentru lucrarea de licenta **URW-Depth** (Uncertainty-guided Robust Weather Depth),
    construita peste arhitectura TinyDepth (TinyViT-5M, 5M parametri).
    """)

    with gr.Tab("Exemple precalculate"):
        gr.Markdown("""
        Rezultate generate offline (fara timp de calcul la incarcarea paginii) — pentru o
        privire rapida fara sa fie nevoie de upload sau de asteptare.
        """)
        gr.Markdown("### Progresie cronologica a ablatiei — Scena 1")
        gr.Gallery(value=_static_ablation_gallery("scene1"), columns=3, height=450, label=None)
        gr.Markdown("### Progresie cronologica a ablatiei — Scena 2")
        gr.Gallery(value=_static_ablation_gallery("scene2"), columns=3, height=450, label=None)
        gr.Markdown("### Robustete la ceata (URW-Depth-S2, severitate crescatoare)")
        gr.Gallery(value=_static_weather_gallery("fog"), columns=5, height=350, label=None)
        gr.Markdown("### Robustete la ploaie (URW-Depth-S2, severitate crescatoare)")
        gr.Gallery(value=_static_weather_gallery("rain"), columns=5, height=350, label=None)
        gr.Markdown("### Robustete la zapada (URW-Depth-S2, severitate crescatoare)")
        gr.Gallery(value=_static_weather_gallery("snow"), columns=5, height=350, label=None)

    with gr.Tab("Demo live"):
        gr.Markdown("Incarca o imagine, alege modelul si conditiile meteo, vezi predictia de adancime si harta de incertitudine.")
        with gr.Row():
            with gr.Column():
                inp_image = gr.Image(type="pil", label="Imagine de intrare")
                model_select = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[-2], label="Model")
                weather_select = gr.Radio(["none", "fog", "rain", "snow"], value="none", label="Conditie meteo")
                severity_slider = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Severitate vreme")
                run_btn = gr.Button("Ruleaza inferenta", variant="primary")
            with gr.Column():
                out_depth = gr.Image(label="Adancime prezisa")
                out_uncert = gr.Image(label="Incertitudine (sigma)")
                out_info = gr.Markdown()
        run_btn.click(demo_live, [inp_image, model_select, weather_select, severity_slider],
                      [out_depth, out_uncert, out_info])

    with gr.Tab("Progresie cronologica (ablatie)"):
        gr.Markdown("""
        Evolutia predictiei de-a lungul celor 6 etape de dezvoltare ale arhitecturii,
        in ordine cronologica: baseline -> +incertitudine -> +automascare -> +suprimare caracteristici
        -> URW-Depth-S2 -> URW-Depth-HiRes.
        """)
        with gr.Row():
            abl_image = gr.Image(type="pil", label="Imagine de intrare")
            with gr.Column():
                abl_weather = gr.Radio(["none", "fog", "rain", "snow"], value="none", label="Conditie meteo")
                abl_severity = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Severitate")
                abl_btn = gr.Button("Ruleaza toate modelele", variant="primary")
        abl_gallery = gr.Gallery(label="Progresie (in ordine)", columns=3, height=500)
        abl_btn.click(ablation_progression, [abl_image, abl_weather, abl_severity], abl_gallery)

    with gr.Tab("Robustete la vreme"):
        gr.Markdown("Compara un singur model la severitati crescatoare ale aceleiasi conditii meteo.")
        with gr.Row():
            wx_image = gr.Image(type="pil", label="Imagine de intrare")
            with gr.Column():
                wx_model = gr.Dropdown(MODEL_CHOICES, value=MODEL_CHOICES[-2], label="Model")
                wx_weather = gr.Radio(["fog", "rain", "snow"], value="fog", label="Conditie meteo")
                wx_btn = gr.Button("Genereaza grid de severitate", variant="primary")
        wx_gallery = gr.Gallery(label="Severitate crescatoare", columns=5, height=400)
        wx_btn.click(weather_severity_grid, [wx_image, wx_model, wx_weather], wx_gallery)

    gr.Markdown("""
    ---
    Modele si cod: [mara-bonea-16/tinydepth-experiments](https://huggingface.co/mara-bonea-16/tinydepth-experiments)
    """)

if __name__ == "__main__":
    demo.launch()
