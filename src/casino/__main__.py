import argparse
from pathlib import Path

from dotenv import load_dotenv

from casino.core.config import load_config, resolve_api_key
from casino.core.loop import ejecutar

def main() -> None:
    parser = argparse.ArgumentParser(prog="casino")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()


    load_dotenv()
    settings = load_config(args.config)
    resolve_api_key(settings)

    resumen = ejecutar(settings, args.config)
    for clave, valor in resumen.items():
        print(f"{clave}={valor}")


if __name__ == "__main__":
    main()