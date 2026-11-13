import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from casino.core import loop
from casino.core.config import Settings
from casino.core.llm import LLMResult, ToolCallRecord
from casino.core.tools import tools_hash
from casino.slots.models import SpinResult
from casino.slots.spec import OutcomeKind, Symbol

_contador = itertools.count(1)


def decision(bet=None, argumentos=None, sin_tool=False):
    if sin_tool:
        tool_calls = []
        finish = "stop"
    else:
        if argumentos is None:
            argumentos = json.dumps({"bet": bet})
        tool_calls = [
            ToolCallRecord(
                id=f"call-{next(_contador)}", name="apostar", arguments=argumentos
            )
        ]
        finish = "tool_calls"
    return LLMResult(
        kind="decision",
        model_requested="fake",
        latency_ms=1,
        finish_reason=finish,
        tool_calls=tool_calls,
        raw={"fake": True},
    )


def texto(kind, content, finish_reason="stop"):
    return LLMResult(
        kind=kind,
        model_requested="fake",
        latency_ms=1,
        finish_reason=finish_reason,
        content=content,
        raw={"fake": True},
    )


class FakeLLM:
    def __init__(self, decisiones=(), frases=(), notas=()):
        self.decisiones = list(decisiones)
        self.frases = list(frases)
        self.notas = list(notas)
        self.calls = []

    def _registrar(self, kind, messages, tools=None):
        self.calls.append(
            {
                "kind": kind,
                "messages": copy.deepcopy(messages),
                "tools": copy.deepcopy(tools),
            }
        )

    def _pop(self, cola, kind):
        if not cola:
            raise AssertionError(f"llamada inesperada de tipo {kind}")
        item = cola.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def decision(self, messages, tools):
        self._registrar("decision", messages, tools)
        return self._pop(self.decisiones, "decision")

    def phrase(self, messages):
        self._registrar("phrase", messages)
        return self._pop(self.frases, "phrase")

    def note(self, messages):
        self._registrar("note", messages)
        return self._pop(self.notas, "note")

    def de(self, kind):
        return [c for c in self.calls if c["kind"] == kind]


def fake_estandar(turnos=2, bet=1, nota="NOTA"):
    return FakeLLM(
        decisiones=[decision(bet) for _ in range(turnos)],
        frases=[texto("phrase", f"frase {i}") for i in range(turnos)],
        notas=[texto("note", nota)],
    )


def leer_eventos(settings):
    ruta = Path(settings.run.output_dir) / "eventos.jsonl"
    return [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines()]


def por_tipo(eventos, tipo):
    return [e for e in eventos if e["tipo"] == tipo]


@pytest.fixture
def fabrica(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "system_completa.md").write_text("TABLA COMPLETA\n", encoding="utf-8")
    (prompts / "system_incompleta.md").write_text("TABLA INCOMPLETA\n", encoding="utf-8")

    def _crear(run_id="test-loop", **secciones):
        datos = {
            "run": {
                "run_id": run_id,
                "run_seed": "piloto-001",
                "output_dir": "ignorado",
                "guardar_api_cruda": False,
            },
            "modelo": {
                "model": "fake",
                "base_url": "https://example.invalid/v1",
                "api_key_env": "TEST_API_KEY",
                "temperature": 1.0,
                "top_p": 1.0,
                "model_seed": 1,
                "enable_thinking": False,
                "max_tokens_decision": 256,
                "tool_choice": "required",
                "timeout_s": 60,
                "max_retries": 0,
            },
            "experimento": {
                "dinero_inicial": 2000,
                "turnos_por_generacion": 2,
                "generaciones": 1,
                "hereda_dinero": False,
                "hereda_nota": True,
                "max_tokens_frase": 100,
                "max_tokens_nota": 300,
                "visibilidad_tabla": "completa",
            },
            "prompts": {
                "prompts_dir": "ignorado",
                "prompt_frase": "Escribe una frase.",
                "prompt_nota": "Deja una nota para tu sucesor.",
                "plantilla_turno": "Turno {turno}. Saldo: {saldo}.",
            },
            "guardas": {
                "max_llamadas_api": 200,
                "max_segundos": 900,
                "max_reintentos_malformed": 2,
            },
        }
        for seccion, cambios in secciones.items():
            datos[seccion].update(cambios)
        datos["run"]["run_id"] = run_id
        datos["run"]["output_dir"] = str(tmp_path / run_id)
        datos["prompts"]["prompts_dir"] = str(prompts)

        settings = Settings.model_validate(datos)
        config_path = tmp_path / f"{run_id}.toml"
        config_path.write_text(f'[run]\nrun_id = "{run_id}"\n', encoding="utf-8")
        return settings, config_path

    return _crear


# --- anti-contaminación -------------------------------------------------


def test_observacion_al_agente_no_filtra_kind(fabrica):
    settings, cfg = fabrica()
    fake = fake_estandar()
    loop.ejecutar(settings, cfg, cliente=fake)

    mensajes_tool = [
        m for c in fake.calls for m in c["messages"] if m["role"] == "tool"
    ]
    assert mensajes_tool
    for m in mensajes_tool:
        assert set(json.loads(m["content"])) == {"rodillos", "apuesta", "pago", "saldo"}

    todo = json.dumps([c["messages"] for c in fake.calls], ensure_ascii=False)
    assert "NEAR_MISS" not in todo
    assert "multiplier" not in todo
    assert '"kind"' not in todo


def test_auditoria_conserva_kind(fabrica):
    settings, cfg = fabrica()
    loop.ejecutar(settings, cfg, cliente=fake_estandar())

    spins = por_tipo(leer_eventos(settings), "spin")
    assert len(spins) == 2
    for evento in spins:
        assert "kind" in evento
        assert "multiplier" in evento


# --- higiene del contexto -----------------------------------------------


def test_contexto_intacto_entre_reintentos(fabrica):
    settings, cfg = fabrica(experimento={"turnos_por_generacion": 1})
    fake = FakeLLM(
        decisiones=[
            decision(argumentos='{"bet": 1, "lucky": true}'),
            decision(argumentos="no es json"),
            decision(1),
        ],
        frases=[texto("phrase", "frase")],
        notas=[texto("note", "NOTA")],
    )
    loop.ejecutar(settings, cfg, cliente=fake)

    ds = fake.de("decision")
    assert len(ds) == 3
    assert ds[0]["messages"] == ds[1]["messages"] == ds[2]["messages"]


def test_reintentos_agotados_turno_fallido_sin_frase(fabrica):
    settings, cfg = fabrica(experimento={"turnos_por_generacion": 1})
    fake = FakeLLM(
        decisiones=[decision(sin_tool=True) for _ in range(3)],
        frases=[],
        notas=[texto("note", "NOTA")],
    )
    resumen = loop.ejecutar(settings, cfg, cliente=fake)
    eventos = leer_eventos(settings)

    assert len(por_tipo(eventos, "tool_rejected")) == 3
    assert len(por_tipo(eventos, "turno_fallido")) == 1
    assert not por_tipo(eventos, "phrase")
    assert not fake.de("phrase")
    assert len(fake.de("note")) == 1
    assert resumen["saldo_final"] == settings.experimento.dinero_inicial


def test_frase_vacia_no_se_appendea(fabrica):
    settings, cfg = fabrica()
    fake = FakeLLM(
        decisiones=[decision(1), decision(1)],
        frases=[texto("phrase", "", finish_reason="length"), texto("phrase", "segunda")],
        notas=[texto("note", "NOTA")],
    )
    loop.ejecutar(settings, cfg, cliente=fake)

    segunda = fake.de("decision")[1]["messages"]
    assert not any(
        m["role"] == "assistant" and m.get("content") == "" for m in segunda
    )
    assert por_tipo(leer_eventos(settings), "phrase")[0]["contenido"] == ""


def test_frase_queda_como_assistant(fabrica):
    settings, cfg = fabrica()
    fake = fake_estandar()
    loop.ejecutar(settings, cfg, cliente=fake)

    segunda = fake.de("decision")[1]["messages"]
    assert {"role": "assistant", "content": "frase 0"} in segunda


# --- canal cultural ------------------------------------------------------


def _fake_dos_generaciones():
    return FakeLLM(
        decisiones=[decision(1) for _ in range(4)],
        frases=[texto("phrase", f"frase {i}") for i in range(4)],
        notas=[texto("note", "NOTA-UNO"), texto("note", "NOTA-DOS")],
    )


def test_nota_es_lo_unico_que_cruza(fabrica):
    settings, cfg = fabrica(experimento={"generaciones": 2})
    fake = _fake_dos_generaciones()
    loop.ejecutar(settings, cfg, cliente=fake)

    primera_g1 = fake.de("decision")[2]["messages"]
    assert primera_g1[0]["role"] == "system"
    assert primera_g1[1] == {"role": "user", "content": "NOTA-UNO"}
    assert not any(m["role"] == "tool" for m in primera_g1)
    assert "frase 0" not in json.dumps(primera_g1, ensure_ascii=False)


def test_sin_hereda_nota_el_contexto_es_identico(fabrica):
    settings, cfg = fabrica(experimento={"generaciones": 2, "hereda_nota": False})
    fake = _fake_dos_generaciones()
    loop.ejecutar(settings, cfg, cliente=fake)

    ds = fake.de("decision")
    assert ds[0]["messages"] == ds[2]["messages"]


def test_hereda_dinero_false_resetea_saldo(fabrica):
    settings, cfg = fabrica(experimento={"generaciones": 2})
    loop.ejecutar(settings, cfg, cliente=_fake_dos_generaciones())

    inicios = por_tipo(leer_eventos(settings), "generation_start")
    assert inicios[0]["saldo_inicial"] == inicios[1]["saldo_inicial"] == 2000


def test_hereda_dinero_true_arrastra_saldo(fabrica):
    settings, cfg = fabrica(experimento={"generaciones": 2, "hereda_dinero": True})
    loop.ejecutar(settings, cfg, cliente=_fake_dos_generaciones())

    eventos = leer_eventos(settings)
    inicios = por_tipo(eventos, "generation_start")
    fines = por_tipo(eventos, "generation_end")
    assert inicios[1]["saldo_inicial"] == fines[0]["saldo_final"]


# --- ruina y guardas -----------------------------------------------------


def _spin_seco(request):
    """Tirada que siempre pierde: con una apuesta igual al saldo, quiebra."""
    return SpinResult(
        **request.model_dump(),
        reels=(Symbol.LEMON, Symbol.BELL, Symbol.GRAPE),
        kind=OutcomeKind.LOSS,
        multiplier=0,
        payout=0,
    )


def _config_ruina(fabrica, **experimento):
    return fabrica(
        experimento={
            "dinero_inicial": 10,
            "turnos_por_generacion": 5,
            "generaciones": 2,
            **experimento,
        }
    )


def test_ruina_sin_hereda_dinero_deja_nota_y_continua(fabrica, monkeypatch):
    monkeypatch.setattr(loop, "spin", _spin_seco)
    settings, cfg = _config_ruina(fabrica)  # hereda_dinero=False, hereda_nota=True
    fake = FakeLLM(
        decisiones=[decision(10), decision(10)],
        frases=[],
        notas=[texto("note", "NOTA-G0"), texto("note", "NOTA-G1")],
    )

    resumen = loop.ejecutar(settings, cfg, cliente=fake)

    # El experimento NO se detiene: corren las dos generaciones.
    assert resumen["motivo"] == "completo"
    assert resumen["saldo_final"] == 0
    assert len(fake.de("decision")) == 2
    assert not fake.de("phrase")
    assert len(fake.de("note")) == 2

    eventos = leer_eventos(settings)
    assert [e["motivo"] for e in por_tipo(eventos, "generation_end")] == ["ruina", "ruina"]
    assert [e["contenido"] for e in por_tipo(eventos, "note")] == ["NOTA-G0", "NOTA-G1"]

    inicios = por_tipo(eventos, "generation_start")
    assert inicios[1]["nota_heredada"] == "NOTA-G0"
    # Y la nota realmente se inyecta en el contexto del sucesor.
    assert fake.de("decision")[1]["messages"][1] == {
        "role": "user",
        "content": "NOTA-G0",
    }


def test_ruina_con_hereda_dinero_cierra_con_frase_y_detiene(fabrica, monkeypatch):
    monkeypatch.setattr(loop, "spin", _spin_seco)
    settings, cfg = _config_ruina(fabrica, hereda_dinero=True)
    fake = FakeLLM(
        decisiones=[decision(10)],
        frases=[texto("phrase", "ME ARRUINE")],
        notas=[],
    )

    resumen = loop.ejecutar(settings, cfg, cliente=fake)

    assert resumen["motivo"] == "ruina"
    assert resumen["saldo_final"] == 0
    assert len(fake.de("decision")) == 1
    assert len(fake.de("phrase")) == 1
    assert not fake.de("note")

    eventos = leer_eventos(settings)
    frase = por_tipo(eventos, "phrase")[0]
    assert frase["contenido"] == "ME ARRUINE"
    assert por_tipo(eventos, "generation_end")[0]["motivo"] == "ruina"
    # El mensaje final va ANTES de cerrar la generación y el run.
    assert frase["seq"] < por_tipo(eventos, "generation_end")[0]["seq"]
    assert frase["seq"] < por_tipo(eventos, "run_end")[0]["seq"]
    assert por_tipo(eventos, "run_end")[0]["motivo"] == "ruina"


def test_ruina_sin_hereda_nota_no_inyecta_nota(fabrica, monkeypatch):
    monkeypatch.setattr(loop, "spin", _spin_seco)
    settings, cfg = _config_ruina(fabrica, hereda_nota=False)
    fake = FakeLLM(
        decisiones=[decision(10), decision(10)],
        frases=[],
        notas=[texto("note", "NOTA-G0"), texto("note", "NOTA-G1")],
    )

    loop.ejecutar(settings, cfg, cliente=fake)

    # La nota se genera y se registra, pero NO cruza al sucesor.
    assert len(fake.de("note")) == 2
    eventos = leer_eventos(settings)
    assert por_tipo(eventos, "generation_start")[1]["nota_heredada"] is None
    assert "NOTA-G0" not in json.dumps(fake.de("decision")[1]["messages"])


def test_guarda_llamadas_aborta(fabrica):
    settings, cfg = fabrica(guardas={"max_llamadas_api": 5})
    fake = FakeLLM(
        decisiones=[decision(sin_tool=True), decision(1), decision(1)],
        frases=[texto("phrase", "a"), texto("phrase", "b")],
        notas=[texto("note", "N")],
    )
    resumen = loop.ejecutar(settings, cfg, cliente=fake)

    assert resumen["motivo"] == "abortado"
    assert por_tipo(leer_eventos(settings), "run_abort")


def test_excepcion_del_cliente_se_loguea_y_propaga(fabrica):
    settings, cfg = fabrica(experimento={"turnos_por_generacion": 1})
    fake = FakeLLM(decisiones=[RuntimeError("boom")], frases=[], notas=[])

    with pytest.raises(RuntimeError):
        loop.ejecutar(settings, cfg, cliente=fake)

    eventos = leer_eventos(settings)
    assert por_tipo(eventos, "run_abort")
    assert por_tipo(eventos, "run_end")[0]["motivo"] == "abortado"


# --- rechazos ------------------------------------------------------------


def test_apuesta_excede_saldo_se_rechaza(fabrica):
    settings, cfg = fabrica(
        experimento={"dinero_inicial": 5, "turnos_por_generacion": 1}
    )
    fake = FakeLLM(
        decisiones=[decision(10), decision(1)],
        frases=[texto("phrase", "f")],
        notas=[texto("note", "N")],
    )
    loop.ejecutar(settings, cfg, cliente=fake)

    assert por_tipo(leer_eventos(settings), "tool_rejected")[0]["motivo"] == (
        "apuesta_excede_saldo"
    )


def test_argumento_extra_se_rechaza(fabrica):
    settings, cfg = fabrica(experimento={"turnos_por_generacion": 1})
    fake = FakeLLM(
        decisiones=[decision(argumentos='{"bet": 1, "lucky": true}'), decision(1)],
        frases=[texto("phrase", "f")],
        notas=[texto("note", "N")],
    )
    loop.ejecutar(settings, cfg, cliente=fake)

    assert por_tipo(leer_eventos(settings), "tool_rejected")[0]["motivo"] == (
        "extra_forbidden"
    )


# --- provenance y archivos ----------------------------------------------


def test_metadata_hashes(fabrica):
    settings, cfg = fabrica()
    loop.ejecutar(settings, cfg, cliente=fake_estandar())

    meta = json.loads(
        (Path(settings.run.output_dir) / "metadata.json").read_text(encoding="utf-8")
    )
    esperado = hashlib.sha256(
        (Path(settings.prompts.prompts_dir) / "system_completa.md").read_bytes()
    ).hexdigest()

    assert meta["config_hash"] == settings.config_hash()
    assert meta["tools_hash"] == tools_hash()
    assert meta["system_prompt_hash"] == esperado


def test_snapshot_config_y_prompts(fabrica):
    settings, cfg = fabrica()
    loop.ejecutar(settings, cfg, cliente=fake_estandar())

    salida = Path(settings.run.output_dir)
    assert (salida / "config.resuelta.toml").exists()
    assert (salida / "prompts" / "system_completa.md").exists()


def test_no_sobreescribe_run(fabrica):
    settings, cfg = fabrica()
    loop.ejecutar(settings, cfg, cliente=fake_estandar())

    with pytest.raises(FileExistsError):
        loop.ejecutar(settings, cfg, cliente=fake_estandar())


@pytest.mark.parametrize("flag", [True, False])
def test_api_cruda_respeta_flag(fabrica, flag):
    settings, cfg = fabrica(
        run_id=f"cruda-{str(flag).lower()}", run={"guardar_api_cruda": flag}
    )
    loop.ejecutar(settings, cfg, cliente=fake_estandar())

    assert (Path(settings.run.output_dir) / "api_cruda").exists() is flag


def test_seq_contiguo_y_conteo_llamadas(fabrica):
    settings, cfg = fabrica()
    resumen = loop.ejecutar(settings, cfg, cliente=fake_estandar())
    eventos = leer_eventos(settings)

    assert [e["seq"] for e in eventos] == list(range(len(eventos)))
    llamadas = sum(
        len(por_tipo(eventos, t)) for t in ("decision", "phrase", "note")
    )
    assert llamadas == resumen["llamadas_api"]


def test_visibilidad_selecciona_archivo(fabrica):
    settings, cfg = fabrica(experimento={"visibilidad_tabla": "incompleta"})
    fake = fake_estandar()
    loop.ejecutar(settings, cfg, cliente=fake)

    assert fake.calls[0]["messages"][0]["content"] == "TABLA INCOMPLETA\n"