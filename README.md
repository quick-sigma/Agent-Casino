# Agent Casino

Experimento de comportamiento de LLMs frente a una tragaperras determinista. Un agente apuesta, comenta lo ocurrido y deja una nota a la generación siguiente; se estudia cómo se propensa a interpretar el **casi-acierto** (*near-miss*), cómo gestiona el saldo y si aparece transmisión cultural entre generaciones.

Todo el azar es reproducible, la máquina es auditable y cada corrida queda registrada como una cola de eventos inmutable.

> Estado: desarrollo activo. `84 tests` en verde. Config de producción en uso: `configs/gui/prod-temp0-casino-001.toml`.

---

## Inicio rápido

```bash
uv sync                       # dependencias del proyecto
# crea .env con OPENAI_API_KEY=nvapi-...   (ya está en .gitignore)

uv run --with fastapi --with uvicorn --with tomli-w --with playwright uvicorn gui.server:app --reload
# → http://localhost:8000   (NEW RUN / REPLAY / EXPORT)
```

Sin la UI, un run directo: `uv run python -m casino --config configs/smoke-001.toml`. Detalles en §4–§5.

---

## 1. Qué se estudia

- **Sesgo del casi-acierto.** El motor define `NEAR_MISS` cuando salen `SEVEN, SEVEN, <no SEVEN>` y paga cero. El agente **no recibe** esa etiqueta: su observación contiene solo hechos (`rodillos`, `apuesta`, `pago`, `saldo`). Si reacciona al near-miss, lo infiere él.
- **Gestión de bankroll bajo RTP conocido.** La máquina devuelve 0.919 por unidad apostada a largo plazo; el agente parte de 2000 y decide cuánto apostar en cada turno.
- **Herencia cultural.** Al cerrar una generación el agente escribe una nota; la generación siguiente puede leerla (`hereda_nota`) y, opcionalmente, heredar el saldo (`hereda_dinero`).
- **La quiebra como evento informativo.** Un agente en bancarrota siempre deja un último mensaje, y según `hereda_dinero` la ruina es terminal o el experimento continúa con una nota legada (ver §2).

La hipótesis operativa es que el diseño permite separar lo que viene del **modelo** de lo que viene del **azar**, porque los rodillos son idénticos para un mismo `(run_seed, generación, turno)` sin importar cuánto apueste el agente.

---

## 2. Diseño

### La máquina (`src/casino/slots/`)

| Pieza | Valor |
|---|---|
| Rodillos | 3 strips de 20 símbolos → 20³ = **8000** combinaciones equiprobables |
| Símbolos | `CHERRY, LEMON, GRAPE, BELL, SEVEN, DIAMOND` |
| Paytable (3 iguales) | `CHERRY 5`, `LEMON 8`, `GRAPE 12`, `BELL 20`, `DIAMOND 40`, `SEVEN 60` × apuesta |
| Dos `CHERRY` exactos | 3 × apuesta |
| `JACKPOT` | tres `SEVEN` (60×) |
| `NEAR_MISS` | `SEVEN, SEVEN, ≠SEVEN` → **paga 0** |
| Apuesta mínima | 1 (sin máximo en el schema; el loop rechaza `bet > saldo`) |

El RNG se deriva con `blake2b(f"{run_seed}|{generation}|{turn}|slots", digest_size=8)` → `random.Random`. Dos decisiones deliberadas:

- Se usa `hashlib`, **no `hash()`**, que está salado por proceso y rompería el replay.
- **La apuesta no entra al material del hash**: subir la apuesta nunca cambia los rodillos.

`src/casino/slots/analysis.py` enumera las 8000 combinaciones (sin muestreo) y reporta:

```
RTP = 919/1000 = 0.919   win_rate = 0.19425   near_miss_rate = 0.059375
JACKPOT 25 · WIN 1529 · NEAR_MISS 475 · LOSS 5971
```

### El agente (`src/casino/core/`)

Por cada turno se hacen **dos** llamadas al LLM y **una** más por generación:

1. `decision` — tool call obligatorio `apostar` con `{"bet": int}`. Si falta la tool, el JSON es inválido o `bet > saldo`, se reintenta hasta `1 + max_reintentos_malformed` veces. **El contexto no se muta entre reintentos** (no se le enseña el schema a mitad de generación).
2. `phrase` — tras la tirada, el agente comenta lo ocurrido; su respuesta vuelve al contexto.
3. `note` — al cerrar la generación, escribe la nota para el sucesor.

`observacion_agente()` entrega solo hechos; `vista_auditoria()` (solo para el log) incluye `kind`, `multiplier`, `run_seed`, etc. Esa asimetría es el corazón del experimento.

### Generaciones

`generaciones` agentes se sientan en orden. Si `hereda_dinero=false`, cada uno reinicia con `dinero_inicial`; si `hereda_nota=true`, arranca leyendo la nota del anterior.

### Bancarrota (condición)

Al llegar a `saldo == 0` el agente **siempre deja un último mensaje**; lo que ocurra después depende de `hereda_dinero`:

| `hereda_dinero` | Mensaje final | ¿Sigue el experimento? | Nota al sucesor |
|---|---|---|---|
| `false` (desactivada) | `note` con `prompt_nota` | **Sí** — la generación siguiente reinicia el saldo | Se genera y registra; se inyecta solo si `hereda_nota=true` |
| `true` (activada) | `phrase` con `prompt_frase` («explicar en una frase») | **No** — `motivo="ruina"`, el run termina ahí | No se genera |

La rama desactivada reutiliza el camino normal de la nota, así que el evento `note` aparece igual que en un cierre sin quiebra. La rama activada registra la frase de cierre **antes** de `generation_end` y `run_end`, para que la GUI la muestre en el globo de diálogo. En ambos casos `generation_end` queda con `motivo="ruina"`; una generación quebrada no incrementa `generaciones_completadas`.

> Con `hereda_nota=false` la nota de un agente quebrado se genera y se registra, pero **no** se inyecta en el contexto del sucesor (se respeta el flag). El evento `generation_start.nota_heredada` se escribe como `null` en ese caso.

Presupuesto de llamadas validado al cargar la config:

```
necesarias = generaciones × (2 × turnos_por_generacion + 1)
```

Con 5 generaciones × 20 turnos = **205 llamadas**.

### Reproducibilidad y auditoría

- Modelos Pydantic `frozen=True` + `extra="forbid"`: un typo en el TOML falla, no se ignora.
- `config_hash`, `system_prompt_hash` y `tools_hash` (SHA-256) se guardan en `metadata.json` y en el evento `run_start`.
- La API key **nunca** entra a `Settings`, por lo que no aparece en dumps ni en hashes.
- `eventos.jsonl` se abre en modo `"x"`: un run nuevo jamás se mezcla con un log viejo. `output_dir` no debe existir.
- `load_config()` exige que `run_id` coincida con el nombre del archivo (`stem`).
- `seed`, `temperature` y `top_p` se envían en cada llamada; `latency_ms` mide **solo** la API (el rate limiter y los cooldowns quedan fuera del cronómetro).

---

## 3. Estructura del repositorio

```
configs/
  smoke-001.toml              # plantilla / smoke run (también base de /api/defaults)
  gui/*.toml                  # configs generadas por la interfaz web
prompts/
  system_completa.md          # reglas que ve el agente (visibilidad "completa")
runs/<run_id>/                # salidas, ignoradas por git
scripts/
  sonda_modelos.py            # compara modelos: decision / frase / nota
  sonda_429.py                # diagnóstico de rate limit en NVIDIA NIM
probes/*.jsonl                # resultados de las sondas
src/casino/
  __main__.py                 # CLI: python -m casino --config ...
  core/
    config.py                 # Settings, validación, hashes, API key
    llm.py                    # LLMClient, RateLimiter, manejo de 429
    loop.py                   # orquestador del experimento
    eventos.py                # EventLog (jsonl con seq)
    observacion.py            # vista del agente vs. vista de auditoría
    tools.py                  # schema de `apostar` + tools_hash
  slots/
    spec.py, models.py, engine.py, analysis.py
gui/
  server.py                   # FastAPI: API + SSE + export de video
  static/                     # interfaz arcade (HTML/CSS/JS)
tests/                        # 84 tests (pytest)
```

---

## 4. Puesta en marcha

**Requisitos:** Python ≥ 3.11, [`uv`](https://docs.astral.sh/uv/), una API key de NVIDIA NIM (`build.nvidia.com`) y `ffmpeg` solo si quieres exportar MP4.

```bash
uv sync                      # crea .venv con las dependencias del proyecto
```

Crea `.env` en la raíz (está en `.gitignore`, nunca se versiona):

```dotenv
OPENAI_API_KEY=nvapi-...
```

El nombre de la variable es configurable con `modelo.api_key_env`, pero por defecto es `OPENAI_API_KEY`. El endpoint por defecto es `https://integrate.api.nvidia.com/v1`.

---

## 5. Cómo ejecutar

### Línea de comandos

```bash
uv run python -m casino --config configs/smoke-001.toml
```

Al terminar imprime el resumen (`run_id`, `motivo`, `llamadas_api`, `saldo_final`, `generaciones_completadas`).

> ⚠️ El script de consola declarado en `pyproject.toml` (`casino = "casino:main"`) **no funciona**: `src/casino/__init__.py` está vacío y no expone `main`. Usa siempre `python -m casino`.

### Interfaz web

```bash
uv run --with fastapi --with uvicorn --with tomli-w --with playwright uvicorn gui.server:app --reload
```

`uv run` instala las dependencias del proyecto y añade con `--with` las que solo usa la GUI; `--reload` reinicia al tocar `gui/`. Abre `http://localhost:8000`. Para comprobar que levantó:

```bash
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/   # 200
```

La GUI permite:

- **NEW RUN**: formulario por pestañas que genera el TOML en `configs/gui/`, lo valida con Pydantic y lanza el proceso.
- **LIVE**: escenario arcade alimentado por SSE (`/api/runs/{id}/stream`), con HUD de saldo, generación, apuesta y globo de diálogo.
- **REPLAY**: paso a paso, scrubber, velocidad 1×/2×/instantáneo e inspector JSON de cada evento.
- **EXPORTAR MP4**: Playwright graba el replay y `ffmpeg` lo normaliza a CFR (60 fps por defecto). Requiere `ffmpeg` y el paquete `playwright`.
- **EXPORTAR .MD**: bitácora completa en Markdown.

Endpoints principales: `GET /api/defaults`, `GET|POST /api/runs`, `POST /api/runs/{id}/stop`, `GET /api/runs/{id}/events`, `GET /api/runs/{id}/stream`, `POST /api/runs/{id}/export`, `GET /api/runs/{id}/export/status`, `GET /api/runs/{id}/video`.

### Sondas

```bash
# Compara modelos / parámetros con el mismo guion de 3 llamadas.
uv run python scripts/sonda_modelos.py --models nvidia/nemotron-3.5-bolt-e4b
# → probes/sonda-<timestamp>.jsonl

# Una sola petición para diagnosticar un 429 (ver sección 7).
uv run python scripts/sonda_429.py
```

---

## 6. Configuración (TOML)

`configs/smoke-001.toml` es la referencia completa. Resumen:

### `[run]`
| Campo | Significado |
|---|---|
| `run_id` | Minúsculas, números y guiones; **debe igualar el nombre del archivo** |
| `run_seed` | Semilla determinista de los rodillos (string) |
| `output_dir` | Destino; **no debe existir** |
| `guardar_api_cruda` | Guarda el JSON crudo de cada respuesta en `api_cruda/` |

### `[modelo]`
`model`, `base_url`, `api_key_env`, `temperature` (0–2), `top_p`, `model_seed`, `enable_thinking`, `max_tokens_decision`, `tool_choice` (`auto|required|none`), `timeout_s`, `max_retries`.

Si `enable_thinking=false` y el modelo devuelve `reasoning_content`/`reasoning`, se lanza `ReasoningLeakError`.

### `[experimento]`
`dinero_inicial`, `turnos_por_generacion`, `generaciones`, `hereda_dinero`, `hereda_nota`, `max_tokens_frase`, `max_tokens_nota`, `visibilidad_tabla` (`completa|incompleta`). Esta última selecciona `prompts/system_<visibilidad>.md`; hoy solo se distribuye `system_completa.md`.

### `[prompts]`
`prompts_dir`, `plantilla_turno` (**debe contener `{turno}` y `{saldo}`**), `prompt_frase`, `prompt_nota`.

### `[guardas]`
| Campo | Significado |
|---|---|
| `max_llamadas_api` | Tope de llamadas; aborta con `GuardiaExcedida` |
| `max_segundos` | Tope de **reloj de pared** (incluye esperas del limitador y cooldowns); aborta con `GuardiaExcedida`. `Settings` exige un mínimo calculado, ver §7 |
| `max_reintentos_malformed` | Reintentos por tool call inválido |
| `max_rpm` | Techo de llamadas por minuto (default en código: 40, solo para TOMLs viejos) |
| `cooldown_429_s` | Espera base tras un 429; se duplica en cada reintento |
| `max_reintentos_429` | Reintentos largos antes de abortar con `RateLimitPenalizado` |

### Defaults de la GUI

`gui/server.py` centraliza los valores que se precargan en el formulario (`GUI_DEFAULTS`): `run_seed="0"`, `temperature=0.0`, `max_retries=0`, los prompts de producción, `max_rpm=12`, `max_segundos=3600`, `cooldown_429_s=60`, `max_reintentos_429=2`, y `run_id` sugerido `prod-temp0-casino-001`. Todos son **editables**. El `run_id` se incrementa automáticamente si ya existe un TOML o un directorio con ese nombre, para no sobreescribir corridas anteriores.

---

## 7. Rate limiting y el 429 de NVIDIA NIM

El free tier de `integrate.api.nvidia.com` **no se comporta como una ventana limpia de 40 RPM**. En la práctica aplica un *penalty box* por cuenta: aparecen 429 con tasas bajas, los bloqueos duran 30–60 min (a veces horas), las respuestas **no traen `Retry-After` ni `X-RateLimit-*`**, y **cada petición hecha durante el 429 reinicia el contador**. Los endpoints son, según la FAQ de NVIDIA, solo para experimentación y desarrollo.

Por eso el cliente toma el control completo:

- **`max_retries = 0`.** El SDK de OpenAI reintenta 429 a 0.5 / 1 / 2 s; esas peticiones son reales, el limitador y el presupuesto no las ven, y echan más leña al bloqueo.
- **`LLMClient.complete` es el único embudo.** Todas las llamadas pasan por un `RateLimiter` de ventana deslizante con un **espaciado mínimo** `60 / max_rpm` (evita ráfagas, que NIM también penaliza).
- Ante un `RateLimitError`, `RateLimiter.cooldown()` duerma `cooldown_429_s × 2ⁿ` y **limpia la ventana** para no soltar otra ráfaga al terminar. Si se agotan los reintentos, `RateLimitPenalizado` aborta con un mensaje accionable.

**El piso de tiempo** lo impone el espaciado, no la API:

```
piso_segundos = llamadas × 60 / max_rpm
```

Pero `max_segundos` mide **reloj de pared**, así que también cuenta los sleeps del limitador y los cooldowns por 429. Un config con el techo justo aborta un run sano a mitad de camino (`max_segundos excedido`, sin que haya ningún error real). Por eso `Settings._tiempo_alcanza` **rechaza el config al cargarlo** si no cumple:

```
minimo = piso + cooldown_429_s × (2^max_reintentos_429 − 1) + 180 s
```

(el peor cooldown de un 429 más 3 min de holgura para latencia). El mensaje de error dice el número exacto a poner. La GUI muestra el mismo cálculo en vivo bajo `max_segundos` y lo corrige automáticamente al cambiar `max_rpm`.

| Config | `max_rpm` | Piso (205 llamadas) | Mínimo exigido | `max_segundos` | Margen |
|---|---|---|---|---|---|
| `configs/smoke-001.toml` | 12 | 1025 s | 1385 s | 3600 s | ✅ |
| Defaults de la GUI | 12 | 1025 s | 1385 s | 3600 s | ✅ |
| `configs/gui/prod-temp0-casino-001.toml` | 7 | ≈1757 s | ≈2117 s | 3600 s | ✅ |

> Historia real: con 7 rpm y `max_segundos=1800` el run abortó tras **82 de 205 llamadas** (1 generación completada). El piso ya era 1757 s; un par de cooldowns de 60/120 s y la latencia se comieron los 43 s de margen. Con 3600 no se repite, y si vuelves a bajar `max_rpm` el validador te lo impide **antes** de perder media hora.

Si necesitas que el run termine antes, la otra palanca es **subir `max_rpm`** (a 12 el piso baja a 1025 s), a costa de más riesgo de 429. No hay forma de acelerar el espaciado sin pagar ese peaje.

**Si una sola petición ya devuelve 429**, la key está en penalty box y ningún ajuste de cliente sirve: deja de llamar 30–60 min y comprueba con `uv run python scripts/sonda_429.py` (una sola petición, `max_retries=0`). Recuerda que esa sonda también reinicia el timer: córrela solo cuando estés listo para empezar a contar la espera.

---

## 8. Salidas de un run

```
runs/<run_id>/
  eventos.jsonl            # cola inmutable, un JSON por línea, con `seq`
  metadata.json            # run_id, modelo, started_at + los 3 hashes
  config.resuelta.toml     # copia exacta del config usado
  prompts/system_*.md      # copia del system prompt usado
  api_cruda/*.json         # respuesta cruda por llamada (si guardar_api_cruda)
  video.mp4                # solo tras exportar desde la GUI
```

Tipos de evento: `run_start`, `generation_start`, `decision`, `tool_rejected`, `turno_fallido`, `spin`, `phrase`, `note`, `generation_end`, `run_end`, `run_abort`. El `run_end` siempre se escribe (incluso si el run aborta) gracias al `finally` del orquestador.

---

## 9. Análisis y tests

```bash
uv run pytest            # 84 tests
uv run ruff check src tests scripts
```

> `ruff` reporta avisos preexistentes no relacionados con la lógica (`EXE002`/`ASYNC220`/`BLE001` en `gui/server.py` y un `import uuid` sin usar en `core/llm.py`).

Cobertura destacada: determinismo del motor entre procesos, independencia de la apuesta respecto a los rodillos, paytable y RTP analítico, invariantes de los schemas, contrato del `LLMClient` (tools, `latency_ms`, reasoning leaks, inmutabilidad de `messages`), guardas del loop, y los nuevos tests de `RateLimiter` / cooldown 429.

Para el modelo analítico exacto:

```python
from casino.slots.analysis import analytic_model
print(analytic_model().rtp)   # Fraction(919, 1000)
```

---

## 10. Limitaciones y notas

- El script de consola `casino` está roto (ver sección 5); usar `python -m casino`.
- `visibilidad_tabla = "incompleta"` requiere crear `prompts/system_incompleta.md`; la GUI lista solo las visibilidades cuyo archivo exista.
- El limitador gobierna **este proceso**. Si compartes la API key con un agente de coding u otra herramienta, ese tráfico también consume el cupo de NIM: baja `max_rpm` en consecuencia.
- Sospecha abierta: además de RPM, NIM parece limitar **tokens por minuto**. El historial `mensajes` crece turno a turno (system + turno + tool call + observación + frase), así que los prompts se inflan con la generación. No se ha medido ni recortado.
- `max_rpm` tiene default `40` en `GuardasConfig` únicamente para que los TOMLs antiguos sin el campo sigan validando; los configs nuevos deben fijarlo explícitamente.
- `repomix-output.xml` es un volcado generado del código, no fuente.
