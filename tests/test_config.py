import pytest
from pydantic import ValidationError

from casino.core.config import Settings


def _datos_base() -> dict:
    """Config válida de 5 gen × 20 turnos = 205 llamadas."""
    return {
        "run": {
            "run_id": "test-config",
            "run_seed": "0",
            "output_dir": "runs/test-config",
            "guardar_api_cruda": True,
        },
        "modelo": {
            "model": "test-model",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "TEST_API_KEY",
            "temperature": 0.0,
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
            "turnos_por_generacion": 20,
            "generaciones": 5,
            "hereda_dinero": False,
            "hereda_nota": True,
            "max_tokens_frase": 100,
            "max_tokens_nota": 300,
            "visibilidad_tabla": "completa",
        },
        "prompts": {
            "prompts_dir": "prompts",
            "prompt_frase": "Escribe una frase.",
            "prompt_nota": "Deja una nota para tu sucesor.",
            "plantilla_turno": "Turno {turno}. Saldo: {saldo}.",
        },
        "guardas": {
            "max_llamadas_api": 300,
            "max_segundos": 3600,
            "max_reintentos_malformed": 2,
            "max_rpm": 7,
            "cooldown_429_s": 60.0,
            "max_reintentos_429": 2,
        },
    }


def test_rechaza_max_segundos_por_debajo_del_piso():
    datos = _datos_base()
    # 205 × 60 / 7 = 1757s de piso; 1800 solo deja 43s y no cubre cooldowns.
    datos["guardas"]["max_segundos"] = 1800

    with pytest.raises(ValidationError, match="mínimo"):
        Settings.model_validate(datos)


def test_acepta_max_segundos_con_holgura():
    settings = Settings.model_validate(_datos_base())

    assert settings.guardas.max_segundos == 3600
    assert settings.guardas.max_rpm == 7


def test_subir_max_rpm_baja_el_minimo_exigido():
    datos = _datos_base()
    datos["guardas"].update({"max_rpm": 12, "max_segundos": 1800})

    # 205 × 60 / 12 = 1025s de piso → 1800s sí alcanza.
    assert Settings.model_validate(datos).guardas.max_segundos == 1800
