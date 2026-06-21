"""
Etapa 3 — Puntúa las explicaciones del AV usando el NLA Activation Reconstructor.

Uso:
    python reconstructor.py \
        --verbalizer_csv  /ruta/a/explicaciones_nla.csv \
        --activations_dir /ruta/a/activaciones \
        --checkpoint_ar   /ruta/a/checkpoints/nla_ar \
        --output_csv      /ruta/a/resultados_nla.csv \
        [--load_in_8bit]
"""
import argparse, csv, json, os
from collections import defaultdict
import torch
import numpy as np
import yaml
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="Puntúa explicaciones NLA usando el Activation Reconstructor.")
    p.add_argument("--verbalizer_csv",  required=True, help="Ruta a explicaciones_nla.csv (salida de verbalizer.py)")
    p.add_argument("--activations_dir", required=True, help="Directorio que contiene los archivos .npy de activaciones")
    p.add_argument("--checkpoint_ar",   required=True, help="Ruta al directorio del checkpoint del NLA Reconstructor")
    p.add_argument("--output_csv",      required=True, help="Ruta donde guardar el CSV de resultados finales")
    p.add_argument("--load_in_8bit",    action="store_true", help="Cargar modelo en 8-bit (T4 / <20 GB VRAM)")
    return p.parse_args()


def load_ar(checkpoint_ar, load_in_8bit):
    tok = AutoTokenizer.from_pretrained(checkpoint_ar, trust_remote_code=True)
    print(f"Cargando AR en modo {'8-bit' if load_in_8bit else 'bfloat16'}...")
    if load_in_8bit:
        backbone = AutoModelForCausalLM.from_pretrained(
            checkpoint_ar,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto", trust_remote_code=True,
        )
    else:
        backbone = AutoModelForCausalLM.from_pretrained(
            checkpoint_ar, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )

    inner = backbone.model
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, torch.nn.Identity())
            print(f"✓ LayerNorm final ({attr}) → Identity")
            break

    d = backbone.config.hidden_size
    value_head = torch.nn.Linear(d, d, bias=False, dtype=torch.float32)
    value_head.load_state_dict(load_file(os.path.join(checkpoint_ar, "value_head.safetensors")))
    device = next(backbone.parameters()).device
    value_head = value_head.to(device).eval()
    backbone.eval()

    print(f"✓ AR cargado. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return tok, backbone, value_head, device


def make_scorer(tok, backbone, value_head, device, ar_template, mse_scale):
    @torch.inference_mode()
    def reconstruir(descripcion):
        prompt = ar_template.format(explanation=descripcion)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        h   = backbone.model(ids, use_cache=False).last_hidden_state
        return value_head(h[0, -1].float()).cpu()

    def score(descripcion, v_raw_np):
        pred = reconstruir(descripcion)
        gold = torch.as_tensor(v_raw_np, dtype=torch.float32)
        pred_n = pred / pred.norm().clamp_min(1e-12) * mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            pred.unsqueeze(0), gold.unsqueeze(0)
        ).item()
        return mse, cos

    return score


def interpretar(cos):
    if cos >= 0.9:    return "EXCELENTE"
    elif cos >= 0.75: return "BUENO"
    elif cos >= 0.5:  return "MEDIOCRE"
    else:             return "POBRE"


def run(args):
    with open(args.verbalizer_csv, encoding="utf-8") as f:
        filas_verb = list(csv.DictReader(f))

    with open(os.path.join(args.checkpoint_ar, "nla_meta.yaml")) as f:
        ar_meta = yaml.safe_load(f)
    mse_scale   = ar_meta["extraction"]["mse_scale"]
    ar_template = ar_meta["prompt_templates"]["ar"]

    tok, backbone, value_head, device = load_ar(args.checkpoint_ar, args.load_in_8bit)
    score = make_scorer(tok, backbone, value_head, device, ar_template, mse_scale)

    COLUMNAS = ["id", "lang", "grupo", "tema", "texto",
                "senales_colombianas", "hipotesis_nla",
                "explicacion_nla", "cos_sim", "mse", "fidelidad"]

    resultados, errores = [], []
    total = len(filas_verb)
    print(f"\nEvaluando {total} entradas...\n")

    for i, fila in enumerate(filas_verb):
        pid, lang = fila["id"], fila["lang"]
        print(f"[{i+1:3d}/{total}] {pid}_{lang} ...", end=" ", flush=True)
        try:
            v_raw = np.load(os.path.join(args.activations_dir, f"{pid}_{lang}.npy"))
            mse, cos = score(fila["explicacion_nla"], v_raw)
            resultados.append({
                "id":                  pid,
                "lang":                lang,
                "grupo":               fila["grupo"],
                "tema":                fila["tema"],
                "texto":               fila["texto"],
                "senales_colombianas": fila["senales_colombianas"],
                "hipotesis_nla":       fila["hipotesis_nla"],
                "explicacion_nla":     fila["explicacion_nla"],
                "cos_sim":             round(cos, 4),
                "mse":                 round(mse, 4),
                "fidelidad":           interpretar(cos),
            })
            print(f"cos={cos:.3f} | {interpretar(cos)}")
        except Exception as e:
            errores.append({"id": pid, "lang": lang, "error": str(e)})
            print(f"✗ ERROR: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()
        writer.writerows(resultados)

    por_grupo = defaultdict(list)
    for r in resultados:
        por_grupo[r["grupo"]].append(r["cos_sim"])

    print(f"\n{'='*55}")
    print("RESUMEN POR GRUPO")
    for grupo, vals in sorted(por_grupo.items()):
        print(f"  {grupo}: n={len(vals)}  media={sum(vals)/len(vals):.4f}  "
              f"max={max(vals):.4f}  min={min(vals):.4f}")
    print(f"\n✓ Resultados guardados : {len(resultados)}")
    print(f"✗ Errores              : {len(errores)}")
    print(f"✓ CSV                  → {args.output_csv}")
    return args.output_csv


if __name__ == "__main__":
    run(parse_args())
