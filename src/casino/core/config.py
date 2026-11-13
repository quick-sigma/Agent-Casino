import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9\-]*$")
    run_seed: str = Field(min_length=1, max_length=64)
    output_dir: Path
    guardar_api_cruda: bool


class ModeloConfig(BaseModel):
    # protected_namespaces vacío: sin esto, model_seed choca con el namespace
    # reservado de Pydantic y emite un warning que filterwarnings=error vuelve fatal.
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float = Field(gt=0.0, le=1.0)
    model_seed: int = Field(ge=0)
    enable_thinking: bool
    max_tokens_decision: int = Field(gt=0)
    tool_choice: Literal["auto", "required", "none"]
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class ExperimentoConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dinero_inicial: int = Field(gt=0)
    turnos_por_generacion: int = Field(gt=0)
    generaciones: int = Field(gt=0)
    hereda_dinero: bool
    hereda_nota: bool
    max_tokens_frase: int = Field(gt=0)
    max_tokens_nota: int = Field(gt=0)
    visibilidad_tabla: Literal["completa", "incompleta"]


class PromptsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompts_dir: Path
    prompt_frase: str = Field(min_length=1)
    prompt_nota: str = Field(min_length=1)
    plantilla_turno: str = Field(min_length=1)

    @model_validator(mode="after")
    def _plantilla_completa(self) -> "PromptsConfig":
        if "{turno}" not in self.plantilla_turno or "{saldo}" not in self.plantilla_turno:
            raise ValueError("plantilla_turno debe contener {turno} y {saldo}")
        return self


class GuardasConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_llamadas_api: int = Field(gt=0)
    max_segundos: float = Field(gt=0)
    max_reintentos_malformed: int = Field(ge=0)
    # Default 40 para que los TOMLs viejos sigan validando. En la práctica el
    # free tier de NIM aplica un "penalty box" muy por debajo de 40 RPM, así que
    # los configs nuevos usan 12 con espaciado mínimo entre llamadas.
    max_rpm: int = Field(default=40, gt=0)
    # NIM no manda Retry-After y cada petición durante el 429 reinicia el bloqueo.
    cooldown_429_s: float = Field(default=60.0, gt=0)
    max_reintentos_429: int = Field(default=2, ge=0)


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run: RunConfig
    modelo: ModeloConfig
    experimento: ExperimentoConfig
    prompts: PromptsConfig
    guardas: GuardasConfig

    @model_validator(mode="after")
    def _presupuesto_alcanza(self) -> "Settings":
        # 2 llamadas por turno (apuesta + frase) más 1 de nota por generación.
        necesarias = self.experimento.generaciones * (
            2 * self.experimento.turnos_por_generacion + 1
        )
        if self.guardas.max_llamadas_api < necesarias:
            raise ValueError(
                f"max_llamadas_api={self.guardas.max_llamadas_api} "
                f"no alcanza para {necesarias} llamadas del diseño"
            )
        return self

    @model_validator(mode="after")
    def _tiempo_alcanza(self) -> "Settings":
        # La guarda mide reloj de pared, así que incluye los sleeps del limitador
        # y los cooldowns 429. Sin holgura aborta un run perfectamente sano.
        necesarias = self.experimento.generaciones * (
            2 * self.experimento.turnos_por_generacion + 1
        )
        piso = necesarias * 60.0 / self.guardas.max_rpm
        peor_cooldown = self.guardas.cooldown_429_s * (
            2**self.guardas.max_reintentos_429 - 1
        )
        minimo = piso + peor_cooldown + 180.0  # +3 min de holgura de latencia
        if self.guardas.max_segundos < minimo:
            raise ValueError(
                f"max_segundos={self.guardas.max_segundos:g}s es menor que el "
                f"mínimo {minimo:.0f}s (piso {piso:.0f}s = {necesarias} llamadas "
                f"a {self.guardas.max_rpm} rpm + {peor_cooldown:.0f}s de cooldowns "
                "+ 180s de holgura). Sube max_segundos o sube max_rpm."
            )
        return self

    def config_hash(self) -> str:
        canonico = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def load_config(path: Path) -> Settings:
    path = Path(path)
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    settings = Settings.model_validate(data)
    if settings.run.run_id != path.stem:
        raise ValueError(
            f"run_id {settings.run.run_id!r} no coincide con el archivo {path.name!r}"
        )
    return settings


def resolve_api_key(settings: Settings) -> SecretStr:
    # La key nunca entra a Settings: así el dump y el hash jamás la contienen.
    value = os.environ.get(settings.modelo.api_key_env)
    if not value:
        raise RuntimeError(
            f"falta la variable de entorno {settings.modelo.api_key_env!r}"
        )
    return SecretStr(value)