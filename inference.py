"""
Etapa 1 — Extrae activaciones de Qwen para los primeros 5 prompts de cada grupo.

Por cada prompt se realizan 4 extracciones: último token de cada cuartil de posición.
Total: 15 historias × 2 idiomas × 4 cuartiles = 120 vectores [d_model].

Uso:
    python inference.py \
        --prompts_json  data/prompts/prompts_espanol_ingles.json \
        --checkpoint    /ruta/a/checkpoints/qwen_sujeto \
        --output_dir    outputs/activaciones \
        [--layer 20] \
        [--load_in_8bit]
"""
import argparse, json, os
from collections import defaultdict
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

GRUPOS = ["explicito_colombiano", "implicito_colombiano", "control_neutral"]
N_POR_GRUPO = 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_json", required=True, help="Ruta al JSON de prompts pareados ES/EN")
    p.add_argument("--checkpoint",   required=True, help="Ruta al checkpoint de Qwen")
    p.add_argument("--output_dir",   required=True, help="Directorio donde guardar .npy y metadatos")
    p.add_argument("--layer",        type=int, default=20, help="Capa a extraer (default: 20)")
    p.add_argument("--load_in_8bit", action="store_true", help="Cargar modelo en 8-bit (T4)")
    return p.parse_args()


def seleccionar_prompts(prompts):
    por_grupo = defaultdict(list)
    for entrada in prompts:
        por_grupo[entrada["grupo"]].append(entrada)
    seleccionados = []
    for grupo in GRUPOS:
        seleccionados.extend(por_grupo[grupo][:N_POR_GRUPO])
    return seleccionados


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


def extract_quartile_activations(texto, tokenizer, model, layer):
    """
    Forward pass único sobre el prompt.
    Retorna 4 vectores [d_model] (último token de cada cuartil) y sus índices.
    """
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

    tensor = almacen["act"][0]  # [N, d_model]

    # Último token de cada cuartil: índices N//4-1, N//2-1, 3N//4-1, N-1
    token_indices = [
        n_tokens // 4 - 1,
        n_tokens // 2 - 1,
        3 * n_tokens // 4 - 1,
        n_tokens - 1,
    ]
    vectors = [tensor[idx].numpy() for idx in token_indices]

    return vectors, token_indices, n_tokens


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prompts_json, encoding="utf-8") as f:
        todos = json.load(f)

    prompts = seleccionar_prompts(todos)
    print(f"{len(prompts)} prompts seleccionados ({N_POR_GRUPO} × {len(GRUPOS)} grupos)")
    print(f"→ {len(prompts) * 2} pares ES/EN × 4 cuartiles = {len(prompts) * 2 * 4} activaciones\n")

    tokenizer, model = load_model(args.checkpoint, args.load_in_8bit)
    d_model = model.config.hidden_size

    metadatos, errores = [], []
    total = len(prompts) * 2
    idx = 0

    for entrada in prompts:
        pid = entrada["id"]
        for lang, campo in [("es", "prompt_es"), ("en", "prompt_en")]:
            idx += 1
            texto = entrada[campo]
            print(f"[{idx:2d}/{total}] {pid}_{lang} ...", end=" ", flush=True)
            try:
                vectors, token_indices, n_tok = extract_quartile_activations(
                    texto, tokenizer, model, args.layer
                )

                cuartiles = {}
                for q, (vec, tok_idx) in enumerate(zip(vectors, token_indices), start=1):
                    nombre_npy = f"{pid}_{lang}_q{q}.npy"
                    np.save(os.path.join(args.output_dir, nombre_npy), vec)
                    cuartiles[f"q{q}"] = {
                        "archivo_npy": nombre_npy,
                        "token_idx":   int(tok_idx),
                    }

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
                    "cuartiles":           cuartiles,
                })
                print(f"✓ ({n_tok} tokens | cuartiles en: {token_indices})")
            except Exception as e:
                errores.append({"id": pid, "lang": lang, "error": str(e)})
                print(f"✗ ERROR: {e}")

    meta_path = os.path.join(args.output_dir, "metadatos_activaciones.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print(f"✓ Prompts procesados : {len(metadatos)}")
    print(f"✗ Errores            : {len(errores)}")
    print(f"✓ Metadatos          → {meta_path}")
    return meta_path


if __name__ == "__main__":
    run(parse_args())
