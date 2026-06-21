"""
Etapa 1 — Extrae activaciones de Qwen para todos los prompts.

Uso:
    python inference.py \
        --prompts_json  /ruta/a/prompts_espanol_ingles.json \
        --checkpoint    /ruta/a/checkpoints/qwen_sujeto \
        --output_dir    /ruta/a/activaciones \
        [--layer 20] \
        [--load_in_8bit]
"""
import argparse, json, os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="Extrae activaciones de Qwen por capa para todos los prompts.")
    p.add_argument("--prompts_json",  required=True, help="Ruta al archivo JSON de prompts")
    p.add_argument("--checkpoint",    required=True, help="Ruta al directorio del checkpoint de Qwen")
    p.add_argument("--output_dir",    required=True, help="Directorio donde guardar los .npy y el JSON de metadatos")
    p.add_argument("--layer",         type=int, default=20, help="Índice de capa a extraer (por defecto: 20)")
    p.add_argument("--load_in_8bit",  action="store_true", help="Cargar modelo en 8-bit (T4 / <20 GB VRAM)")
    return p.parse_args()


def load_model(checkpoint, load_in_8bit):
    print("Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    print(f"Cargando modelo en modo {'8-bit' if load_in_8bit else 'bfloat16'}...")
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    print(f"✓ Modelo cargado — {model.config.num_hidden_layers} capas")
    return tokenizer, model


def extract_activation(texto, tokenizer, model, layer):
    almacen = {}

    def hook_fn(module, inp, out):
        almacen["act"] = out.detach().float().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    chat = [{"role": "user", "content": texto}]
    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )["input_ids"].to("cuda")

    n_tokens = input_ids.shape[1]
    with torch.no_grad():
        _ = model(input_ids=input_ids)

    handle.remove()

    tensor  = almacen["act"][0]                    # [T, d_model]
    inicio  = min(10, n_tokens - 1)                # omitir tokens iniciales con normas anómalas
    act_vec = tensor[inicio:].mean(dim=0).numpy()  # mean pool → [d_model]
    return act_vec, n_tokens


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prompts_json, encoding="utf-8") as f:
        prompts = json.load(f)
    print(f"{len(prompts)} prompts cargados — {len(prompts) * 2} activaciones a extraer (ES + EN)\n")

    tokenizer, model = load_model(args.checkpoint, args.load_in_8bit)
    d_model = model.config.hidden_size

    metadatos, errores = [], []
    total = len(prompts) * 2

    for i, entrada in enumerate(prompts):
        pid = entrada["id"]
        for lang, campo in [("es", "prompt_es"), ("en", "prompt_en")]:
            op_num = i * 2 + (0 if lang == "es" else 1) + 1
            texto  = entrada[campo]
            print(f"[{op_num:3d}/{total}] {pid}_{lang} ...", end=" ", flush=True)
            try:
                act_vec, n_tok = extract_activation(texto, tokenizer, model, args.layer)
                nombre_npy = f"{pid}_{lang}.npy"
                np.save(os.path.join(args.output_dir, nombre_npy), act_vec)
                metadatos.append({
                    "id":                  pid,
                    "lang":                lang,
                    "grupo":               entrada["grupo"],
                    "tema":                entrada["tema"],
                    "texto":               texto,
                    "senales_colombianas": entrada.get("senales_colombianas", []),
                    "hipotesis_nla":       entrada.get("hipotesis_nla", ""),
                    "n_tokens":            n_tok,
                    "capa":                args.layer,
                    "d_model":             d_model,
                    "archivo_npy":         nombre_npy,
                })
                print(f"✓ ({n_tok} tokens)")
            except Exception as e:
                errores.append({"id": pid, "lang": lang, "error": str(e)})
                print(f"✗ ERROR: {e}")

    meta_path = os.path.join(args.output_dir, "metadatos_activaciones.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"✓ Activaciones guardadas : {len(metadatos)}")
    print(f"✗ Errores                : {len(errores)}")
    print(f"✓ Metadatos              → {meta_path}")
    return meta_path


if __name__ == "__main__":
    run(parse_args())
