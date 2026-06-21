"""
Etapa 2 — Verbaliza activaciones con el NLA Activation Verbalizer.

Uso:
    python verbalizer.py \
        --metadata_json  /ruta/a/activaciones/metadatos_activaciones.json \
        --checkpoint_av  /ruta/a/checkpoints/nla_av \
        --output_csv     /ruta/a/explicaciones_nla.csv \
        [--load_in_8bit]
"""
import argparse, csv, json, os, re
import torch
import numpy as np
import yaml
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def parse_args():
    p = argparse.ArgumentParser(description="Verbaliza activaciones de Qwen usando el NLA Activation Verbalizer.")
    p.add_argument("--metadata_json", required=True, help="Ruta a metadatos_activaciones.json (salida de inference.py)")
    p.add_argument("--checkpoint_av", required=True, help="Ruta al directorio del checkpoint del NLA Verbalizer")
    p.add_argument("--output_csv",    required=True, help="Ruta donde guardar el CSV de explicaciones")
    p.add_argument("--load_in_8bit",  action="store_true", help="Cargar modelo en 8-bit (T4 / <20 GB VRAM)")
    return p.parse_args()


def load_embedding(checkpoint_dir, dtype):
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = json.load(open(index_path))["weight_map"]
        key   = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
        shard = os.path.join(checkpoint_dir, weight_map[key])
    else:
        shard = os.path.join(checkpoint_dir, "model.safetensors")
        with safe_open(shard, framework="pt") as f:
            key = next(k for k in f.keys() if k.endswith("embed_tokens.weight"))
    with safe_open(shard, framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)
    emb = torch.nn.Embedding(*weight.shape, _weight=weight)
    emb.requires_grad_(False)
    return emb.eval()


def load_av(checkpoint_av, load_in_8bit):
    tok = AutoTokenizer.from_pretrained(checkpoint_av, trust_remote_code=True)
    print(f"Cargando AV en modo {'8-bit' if load_in_8bit else 'bfloat16'}...")
    if load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_av,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_av, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True,
        )
    model.eval()
    dtype_emb  = torch.float16 if load_in_8bit else torch.bfloat16
    embed_layer = load_embedding(checkpoint_av, dtype_emb)
    print(f"✓ AV cargado. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return tok, model, embed_layer


def make_verbalizer(tok, model, embed_layer, nla_meta, load_in_8bit):
    injection_scale  = nla_meta["extraction"]["injection_scale"]
    injection_char   = nla_meta["tokens"]["injection_char"]
    injection_tok_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor    = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor   = nla_meta["tokens"]["injection_right_neighbor_id"]
    prompt_template  = nla_meta["prompt_templates"]["av"]
    device           = next(model.parameters()).device

    def verbalizar(v_raw, max_new_tokens=200):
        contenido = prompt_template.format(injection_char=injection_char)
        fmt = tok.apply_chat_template(
            [{"role": "user", "content": contenido}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tok.encode(fmt, add_special_tokens=False)

        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            embeds = embed_layer(ids_t).float()

        v = torch.as_tensor(v_raw, dtype=torch.float32)
        v_scaled = v / v.norm().clamp_min(1e-12) * injection_scale

        inyectado = False
        for p in range(1, len(input_ids) - 1):
            if (input_ids[p] == injection_tok_id and
                    input_ids[p - 1] == left_neighbor and
                    input_ids[p + 1] == right_neighbor):
                embeds[0, p] = v_scaled
                inyectado = True
                break
        assert inyectado, "Posición de inyección no encontrada — revisar prompt template"

        embeds_dev = embeds.to(device)
        if load_in_8bit:
            embeds_dev = embeds_dev.bfloat16()

        with torch.no_grad():
            out = model.generate(
                inputs_embeds=embeds_dev,
                max_new_tokens=max_new_tokens,
                temperature=1.0, do_sample=True,
                pad_token_id=tok.eos_token_id,
            )

        texto_gen = tok.decode(out[0], skip_special_tokens=False)
        m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", texto_gen, re.DOTALL)
        return m.group(1).strip() if m else texto_gen

    return verbalizar


def run(args):
    with open(args.metadata_json, encoding="utf-8") as f:
        metadatos = json.load(f)
    activations_dir = os.path.dirname(args.metadata_json)

    with open(os.path.join(args.checkpoint_av, "nla_meta.yaml")) as f:
        nla_meta = yaml.safe_load(f)

    tok, model, embed_layer = load_av(args.checkpoint_av, args.load_in_8bit)
    verbalizar = make_verbalizer(tok, model, embed_layer, nla_meta, args.load_in_8bit)

    COLUMNAS = ["id", "lang", "grupo", "tema", "texto",
                "senales_colombianas", "hipotesis_nla", "explicacion_nla"]

    filas, errores = [], []
    total = len(metadatos)
    print(f"\nVerbalizando {total} activaciones...\n")

    for i, entrada in enumerate(metadatos):
        pid, lang = entrada["id"], entrada["lang"]
        print(f"[{i+1:3d}/{total}] {pid}_{lang} ...", end=" ", flush=True)
        try:
            v_raw = np.load(os.path.join(activations_dir, entrada["archivo_npy"]))
            explicacion = verbalizar(v_raw)
            filas.append({
                "id":                  pid,
                "lang":                lang,
                "grupo":               entrada["grupo"],
                "tema":                entrada["tema"],
                "texto":               entrada["texto"],
                "senales_colombianas": "|".join(entrada.get("senales_colombianas", [])),
                "hipotesis_nla":       entrada.get("hipotesis_nla", ""),
                "explicacion_nla":     explicacion,
            })
            print(f"✓  {explicacion[:70]}...")
        except Exception as e:
            errores.append({"id": pid, "lang": lang, "error": str(e)})
            print(f"✗ ERROR: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        writer.writeheader()
        writer.writerows(filas)

    print(f"\n{'='*55}")
    print(f"✓ Explicaciones guardadas : {len(filas)}")
    print(f"✗ Errores                 : {len(errores)}")
    print(f"✓ CSV                     → {args.output_csv}")
    return args.output_csv


if __name__ == "__main__":
    run(parse_args())
