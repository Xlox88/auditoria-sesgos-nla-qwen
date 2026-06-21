# Auditoría de Sesgos Multilingües con NLA

Pipeline de auditoría que usa **Natural Language Autoencoders (NLA)** sobre Qwen-2.5-7B para detectar representaciones internas sesgadas ante prompts en español colombiano vs. inglés.

## Descripción

El pipeline extrae activaciones de la capa intermedia de Qwen-2.5-7B para cada prompt, las verbaliza con el Activation Verbalizer (AV) y evalúa la fidelidad de cada explicación con el Activation Reconstructor (AR). El resultado final es un CSV con las explicaciones generadas y sus métricas de calidad, listo para el análisis de sesgos.

```
prompts JSON  →  [inference.py]  →  .npy + metadatos
                                         ↓
                               [verbalizer.py]  →  explicaciones_nla.csv
                                                          ↓
                                          [reconstructor.py]  →  resultados_nla.csv
```

## Requisitos

- Python 3.10+
- GPU con mínimo ?? GB VRAM
- Checkpoints descargados desde HuggingFace:
  - `qwen_sujeto` — Qwen-2.5-7B
  - `nla_av` — Activation Verbalizer
  - `nla_ar` — Activation Reconstructor

```bash
pip install -r requirements.txt
```

## Uso

### Opción A — Pipeline completo (recomendado)

Ejecuta las tres etapas en secuencia con un solo comando.

El dataset está dividido en dos batches con el mismo formato — correr uno tras otro produce los 360 vectores completos (180 prompts × 2 idiomas):

```bash
# Batch 1: prompts A01–C30
python pipeline.py \
    --prompts_json   /ruta/a/prompts_espanol_ingles.json \
    --checkpoint     /ruta/a/checkpoints/qwen_sujeto \
    --checkpoint_av  /ruta/a/checkpoints/nla_av \
    --checkpoint_ar  /ruta/a/checkpoints/nla_ar \
    --output_dir     ./salida/batch1 \
    --load_in_8bit

# Batch 2: prompts A31–C60
python pipeline.py \
    --prompts_json   /ruta/a/prompts_espanol_ingles_2.json \
    --checkpoint     /ruta/a/checkpoints/qwen_sujeto \
    --checkpoint_av  /ruta/a/checkpoints/nla_av \
    --checkpoint_ar  /ruta/a/checkpoints/nla_ar \
    --output_dir     ./salida/batch2 \
    --load_in_8bit
```

> **Nota:** Los archivos `prompts_espanol*.json` y `prompts_ingles*.json` son versiones separadas (solo ES / solo EN) del mismo dataset y **no son necesarios** para el pipeline.

### Opción B — Etapas individuales

**Etapa 1 — Extraer activaciones de Qwen:**
```bash
python inference.py \
    --prompts_json  /ruta/a/prompts_espanol_ingles.json \   # o prompts_espanol_ingles_2.json
    --checkpoint    /ruta/a/checkpoints/qwen_sujeto \
    --output_dir    ./salida/activaciones \
    --load_in_8bit
```

**Etapa 2 — Verbalizar activaciones:**
```bash
python verbalizer.py \
    --metadata_json  ./salida/activaciones/metadatos_activaciones.json \
    --checkpoint_av  /ruta/a/checkpoints/nla_av \
    --output_csv     ./salida/explicaciones_nla.csv \
    --load_in_8bit
```

**Etapa 3 — Puntuar explicaciones:**
```bash
python reconstructor.py \
    --verbalizer_csv  ./salida/explicaciones_nla.csv \
    --activations_dir ./salida/activaciones \
    --checkpoint_ar   /ruta/a/checkpoints/nla_ar \
    --output_csv      ./salida/resultados_nla.csv \
    --load_in_8bit
```

## Parámetros

| Parámetro | Descripción | Por defecto |
|---|---|---|
| `--load_in_8bit` | Cuantización 8-bit para GPUs con <20 GB VRAM | desactivado |
| `--layer` | Capa de Qwen de la que se extraen activaciones | 20 |

## Archivos de salida

| Archivo | Descripción |
|---|---|
| `activaciones/{id}_{lang}.npy` | Vector de activación [3584] por cada prompt-idioma |
| `activaciones/metadatos_activaciones.json` | Metadatos de todos los prompts procesados |
| `explicaciones_nla.csv` | Explicaciones generadas por el AV para cada activación |
| `resultados_nla.csv` | CSV final con explicaciones + cosine similarity + MSE + fidelidad |

### Columnas de `resultados_nla.csv`

| Columna | Descripción |
|---|---|
| `id` | Identificador del prompt (e.g. `A01`, `B15`, `C30`) |
| `lang` | Idioma del prompt (`es` / `en`) |
| `grupo` | `explicito_colombiano` / `implicito_colombiano` / `control_neutral` |
| `tema` | Categoría temática (salud, transporte, cultura, etc.) |
| `texto` | Texto original del prompt |
| `senales_colombianas` | Marcadores regionales presentes (separados por `\|`) |
| `hipotesis_nla` | Hipótesis esperada sobre la activación interna |
| `explicacion_nla` | Explicación generada por el AV |
| `cos_sim` | Cosine similarity entre activación original y reconstruida [0–1] |
| `mse` | Error cuadrático medio normalizado |
| `fidelidad` | Interpretación: `EXCELENTE` / `BUENO` / `MEDIOCRE` / `POBRE` |

## Estructura del repositorio

```
repo/
├── requirements.txt
├── inference.py       # Etapa 1: extracción de activaciones
├── verbalizer.py      # Etapa 2: verbalización con el AV
├── reconstructor.py   # Etapa 3: puntuación con el AR
└── pipeline.py        # Ejecuta las tres etapas en secuencia
```

## Equipo

Universidad Autónoma de Occidente — Semillero de Investigación