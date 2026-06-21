"""
Pipeline NLA — ejecuta inferencia → verbalizador en secuencia.

Etapa 3 (análisis) pendiente de confirmación de herramienta.

Uso:
    python pipeline.py \
        --prompts_json   data/prompts/prompts_espanol_ingles.json \
        --checkpoint     /ruta/a/qwen_sujeto \
        --checkpoint_av  /ruta/a/nla_av \
        --output_dir     outputs \
        [--layer 20] \
        [--load_in_8bit]

Archivos generados en --output_dir:
    activaciones/metadatos_activaciones.json
    activaciones/{id}_{lang}_q{1-4}.npy     (4 por prompt, 120 total)
    explicaciones_nla.csv                   (120 filas: 30 prompts × 4 cuartiles)
"""
import argparse, os
from types import SimpleNamespace

import inference as stage1
import verbalizer as stage2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts_json",  required=True, help="Ruta al JSON de prompts pareados ES/EN")
    p.add_argument("--checkpoint",    required=True, help="Ruta al checkpoint de Qwen")
    p.add_argument("--checkpoint_av", required=True, help="Ruta al checkpoint del NLA Verbalizer")
    p.add_argument("--output_dir",    required=True, help="Directorio raíz para archivos de salida")
    p.add_argument("--layer",         type=int, default=20, help="Capa de Qwen a extraer (default: 20)")
    p.add_argument("--load_in_8bit",  action="store_true", help="Cargar modelos en 8-bit (T4)")
    return p.parse_args()


def main():
    args = parse_args()
    activations_dir = os.path.join(args.output_dir, "activaciones")
    csv_verb        = os.path.join(args.output_dir, "explicaciones_nla.csv")

    print("=" * 60)
    print("ETAPA 1 — Inferencia (activaciones de Qwen, 4 cuartiles)")
    print("=" * 60)
    stage1.run(SimpleNamespace(
        prompts_json=args.prompts_json,
        checkpoint=args.checkpoint,
        output_dir=activations_dir,
        layer=args.layer,
        load_in_8bit=args.load_in_8bit,
    ))

    print("\n" + "=" * 60)
    print("ETAPA 2 — Verbalizador (120 llamadas al AV)")
    print("=" * 60)
    stage2.run(SimpleNamespace(
        metadata_json=os.path.join(activations_dir, "metadatos_activaciones.json"),
        checkpoint_av=args.checkpoint_av,
        output_csv=csv_verb,
        load_in_8bit=args.load_in_8bit,
    ))

    print("\n" + "=" * 60)
    print("ETAPAS 1-2 COMPLETADAS")
    print(f"  Activaciones  : {activations_dir}")
    print(f"  Explicaciones : {csv_verb}")
    print("  Etapa 3 (análisis): pendiente")
    print("=" * 60)


if __name__ == "__main__":
    main()
