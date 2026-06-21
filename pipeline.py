"""
Pipeline NLA completo — ejecuta inferencia → verbalizador → reconstructor en secuencia.

Uso:
    python pipeline.py \
        --prompts_json   /ruta/a/prompts_espanol_ingles.json \
        --checkpoint     /ruta/a/checkpoints/qwen_sujeto \
        --checkpoint_av  /ruta/a/checkpoints/nla_av \
        --checkpoint_ar  /ruta/a/checkpoints/nla_ar \
        --output_dir     /ruta/a/salida \
        [--layer 20] \
        [--load_in_8bit]

Archivos generados en --output_dir:
    activaciones/metadatos_activaciones.json
    activaciones/{id}_{lang}.npy  (uno por cada par prompt-idioma)
    explicaciones_nla.csv
    resultados_nla.csv
"""
import argparse, os
from types import SimpleNamespace

import inference as stage1
import verbalizer as stage2
import reconstructor as stage3


def parse_args():
    p = argparse.ArgumentParser(description="Ejecuta el pipeline NLA completo.")
    p.add_argument("--prompts_json",  required=True, help="Ruta al archivo JSON de prompts")
    p.add_argument("--checkpoint",    required=True, help="Ruta al checkpoint de Qwen")
    p.add_argument("--checkpoint_av", required=True, help="Ruta al checkpoint del NLA Activation Verbalizer")
    p.add_argument("--checkpoint_ar", required=True, help="Ruta al checkpoint del NLA Activation Reconstructor")
    p.add_argument("--output_dir",    required=True, help="Directorio raíz para todos los archivos de salida")
    p.add_argument("--layer",         type=int, default=20, help="Capa de Qwen a extraer (por defecto: 20)")
    p.add_argument("--load_in_8bit",  action="store_true", help="Cargar todos los modelos en 8-bit")
    return p.parse_args()


def main():
    args = parse_args()
    activations_dir = os.path.join(args.output_dir, "activaciones")
    csv_verb        = os.path.join(args.output_dir, "explicaciones_nla.csv")
    csv_final       = os.path.join(args.output_dir, "resultados_nla.csv")

    print("=" * 60)
    print("ETAPA 1 — Inferencia (activaciones de Qwen)")
    print("=" * 60)
    stage1.run(SimpleNamespace(
        prompts_json=args.prompts_json,
        checkpoint=args.checkpoint,
        output_dir=activations_dir,
        layer=args.layer,
        load_in_8bit=args.load_in_8bit,
    ))

    print("\n" + "=" * 60)
    print("ETAPA 2 — Verbalizador (explicaciones del AV)")
    print("=" * 60)
    stage2.run(SimpleNamespace(
        metadata_json=os.path.join(activations_dir, "metadatos_activaciones.json"),
        checkpoint_av=args.checkpoint_av,
        output_csv=csv_verb,
        load_in_8bit=args.load_in_8bit,
    ))

    print("\n" + "=" * 60)
    print("ETAPA 3 — Reconstructor (puntuación del AR)")
    print("=" * 60)
    stage3.run(SimpleNamespace(
        verbalizer_csv=csv_verb,
        activations_dir=activations_dir,
        checkpoint_ar=args.checkpoint_ar,
        output_csv=csv_final,
        load_in_8bit=args.load_in_8bit,
    ))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print(f"  Activaciones  : {activations_dir}")
    print(f"  Explicaciones : {csv_verb}")
    print(f"  Resultados    : {csv_final}")
    print("=" * 60)


if __name__ == "__main__":
    main()
