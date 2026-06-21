# Auditoría de Sesgos Multilingües con NLA — Qwen-2.5-7B

Universidad Autónoma de Occidente · Semillero de Investigación · Track: IA Responsable

---

## Pregunta de investigación

¿Construye Qwen-2.5-7B representaciones internas distintas ante prompts en español colombiano versus inglés, aun cuando el contenido semántico es equivalente? Se busca evidencia de sesgos no verbalizados: supuestos sobre nacionalidad, nivel socioeconómico o intención del usuario codificados en las activaciones pero invisibles en la respuesta.

---

## Contexto técnico

El paper *Natural Language Autoencoders* (Anthropic, mayo 2026) demuestra que es posible convertir una activación interna en lenguaje natural y viceversa mediante un ciclo de ida y vuelta:

```
activación h_l  →  [AV]  →  texto z  →  [AR]  →  activación reconstruida
```

La métrica de fidelidad es `r(h_l, z) = −log‖h_l − AR_θ(z)‖²₂`. En la práctica se usan cosine similarity y MSE normalizado.

**Modelo bajo auditoría:** Qwen-2.5-7B-Instruct — 28 capas, d_model=3584. Se extrae de la **capa 20** (media). Mean-pool sobre tokens 10+ para producir un vector [3584] por prompt.

---

## Checkpoints (HuggingFace)

| Rol | HuggingFace ID |
|---|---|
| Modelo base (sujeto de auditoría) | `Qwen/Qwen2.5-7B-Instruct` |
| Activation Verbalizer (AV) | `kitft/nla-qwen2.5-7b-L20-av` |
| Activation Reconstructor (AR) | `kitft/nla-qwen2.5-7b-L20-ar` |

Los checkpoints del AV y AR incluyen `nla_meta.yaml` con parámetros críticos: `injection_scale=150.0`, `mse_scale=59.8665` (≈√3584), IDs de tokens vecinos del carácter de inyección (`㈎`, token_id=149705), y los templates de prompt para AV y AR.

**Entorno:** Google Colab, GPU T4 (16 GB VRAM), carga en 8-bit (`--load_in_8bit`).

---

## Dataset de prompts

180 prompts únicos × 2 idiomas = 360 activaciones a extraer.

### Grupos

| Grupo | IDs | Descripción |
|---|---|---|
| `explicito_colombiano` | A01–A60 | Prompts con referencias explícitas a Colombia (EPS, TransMilenio, tutela, etc.) |
| `implicito_colombiano` | B01–B60 | Contexto colombiano implícito sin marcadores geográficos directos |
| `control_neutral` | C01–C60 | Prompts neutros sin señales culturales |

### Archivos JSON

| Archivo | Contenido | Compatible con pipeline |
|---|---|---|
| `prompts_espanol_ingles.json` | Paired ES+EN, A01–C30 | ✓ |
| `prompts_espanol_ingles_2.json` | Paired ES+EN, A31–C60 | ✓ |
| `prompts_espanol.json` | Solo ES, A01–C30 | ✗ (split intermedio) |
| `prompts_ingles.json` | Solo EN, A01–C30 | ✗ (split intermedio) |
| `prompts_espanol_2.json` | Solo ES, A31–C60 | ✗ (split intermedio) |
| `prompts_ingles_2.json` | Solo EN, A31–C60 | ✗ (split intermedio) |

Los archivos `_espanol` e `_ingles` son versiones separadas usadas durante la generación; el pipeline solo necesita los paired (`*_ingles.json` y `*_ingles_2.json`).

Los prompts fueron generados con ayuda de LLMs (LatamGPT fue la intención original pero presentó problemas de inferencia; la generación final puede haber sido con GPT/Claude).

Cada entrada tiene la estructura:
```json
{
  "id": "A01",
  "grupo": "explicito_colombiano",
  "tema": "salud",
  "prompt_es": "...",
  "senales_colombianas": ["Colombia", "EPS"],
  "hipotesis_nla": "Activación de representación...",
  "prompt_en": "..."
}
```

---

## Estructura de archivos

```
HACKATHON/
├── CLAUDE.md                          ← este archivo
│
├── prompts_espanol_ingles.json        ← dataset batch 1 (A01–C30)
├── prompts_espanol_ingles_2.json      ← dataset batch 2 (A31–C60)
├── prompts_espanol.json               ← split ES batch 1
├── prompts_ingles.json                ← split EN batch 1
├── prompts_espanol_2.json             ← split ES batch 2
├── prompts_ingles_2.json              ← split EN batch 2
│
├── NLA_inferencia.ipynb               ← notebook original (1 prompt hardcodeado)
├── NLA_verbalizer.ipynb               ← notebook original (1 prompt)
├── NLA_reconstructor.ipynb            ← notebook original (1 prompt)
├── NLA_inferencia_batch.ipynb         ← batch: lee JSONs, guarda .npy + metadatos
├── NLA_verbalizer_batch.ipynb         ← batch: lee metadatos, guarda explicaciones CSV
├── NLA_reconstructor_batch.ipynb      ← batch: puntúa explicaciones, guarda resultados CSV
│
└── repo/                              ← repositorio ejecutable (sin notebooks)
    ├── CLAUDE.md                      ← (este mismo archivo vive aquí también)
    ├── README.md
    ├── requirements.txt
    ├── inference.py                   ← Etapa 1: extrae activaciones
    ├── verbalizer.py                  ← Etapa 2: verbaliza con el AV
    ├── reconstructor.py               ← Etapa 3: puntúa con el AR
    └── pipeline.py                    ← ejecuta las 3 etapas en secuencia
```

---

## Pipeline (repo/)

```
prompts_espanol_ingles*.json
        ↓
  [inference.py]  →  activaciones/{id}_{lang}.npy  +  metadatos_activaciones.json
        ↓
  [verbalizer.py]  →  explicaciones_nla.csv
        ↓
  [reconstructor.py]  →  resultados_nla.csv
```

Comando para correr el pipeline completo (hacer para batch 1 y batch 2):
```bash
python pipeline.py \
    --prompts_json   prompts_espanol_ingles.json \
    --checkpoint     /ruta/qwen_sujeto \
    --checkpoint_av  /ruta/nla_av \
    --checkpoint_ar  /ruta/nla_ar \
    --output_dir     ./salida/batch1 \
    --load_in_8bit
```

Salida final (`resultados_nla.csv`) tiene columnas: `id, lang, grupo, tema, texto, senales_colombianas, hipotesis_nla, explicacion_nla, cos_sim, mse, fidelidad`.

---

## Estado del proyecto

### Hecho
- [x] Entorno Colab validado con prompt único en inglés (notebooks originales)
- [x] Notebooks batch: inferencia, verbalizer, reconstructor
- [x] Dataset de 180 prompts pareados ES/EN en dos batches
- [x] Repositorio CLI (`repo/`) con los 4 scripts y README en español

### Pendiente
- [ ] **Correr el pipeline en Colab** sobre los 360 pares (180 prompts × 2 idiomas) — produce `resultados_nla.csv`
- [ ] **Análisis de resultados**: string matching sobre `explicacion_nla` buscando menciones a idioma, nacionalidad, nivel socioeconómico; comparar frecuencias por grupo (A vs B vs C) y por idioma (ES vs EN)
- [ ] **Reporte final**: 4–8 páginas con ejemplos concretos, cuantificación de frecuencias y recomendaciones

---

## Recursos externos

- Paper NLA: Anthropic, mayo 2026 — *Natural Language Autoencoders*
- Frontend sin GPU: Neuronpedia (respaldo si Colab falla)
- LatamGPT: `https://www.latamgpt.org/resources` — Llama 3.1 70B fine-tuned en ~297B tokens LatAm (usado para generación de prompts, no como modelo de auditoría)
