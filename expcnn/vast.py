"""El freno de los experimentos que alquilan, para Telegram: `/use exp-vast`.

    python3 -m expcnn.vast estado            todos los libros de Vast del repo + lo vivo
    python3 -m expcnn.vast apagar <prefijo>  para sus unidades y destruye sus maquinas

Existe en el MISMO commit que el modo `trabajo` (regla 4 de escritura, R11): quien puede
encender maquinas desde aqui tiene que poder apagarlas desde el movil.

Lee los libros de `*/resultados/vast/*/` -- el directorio donde los deja `nn/vast.sh` de
cada experimento -- y delega todo lo demas en el lanzador (`vast_instance.py trabajo
--estado/--apagar`): el conocimiento de la API de Vast vive en un sitio.

⚠ `apagar` solo acepta prefijos `expc-`: desde este ejecutor no se puede destruir una
maquina que no sea de un experimento de este repo.
"""

from __future__ import annotations

import subprocess
import sys

from .entorno import exigir_lanzador
from .registro import raiz


def _v(*args: str) -> int:
    lanzador = exigir_lanzador()
    return subprocess.run([sys.executable, str(lanzador / "scripts" / "vast_instance.py"),
                           *args]).returncode


def estado() -> int:
    libros = sorted(p for p in raiz().glob("*/resultados/vast/*") if p.is_dir())
    if not libros:
        print("Ningun experimento ha alquilado nada en Vast desde este repo (no hay libros).",
              flush=True)
    for p in libros:
        print(f"\n== {p.relative_to(raiz())}", flush=True)
        _v("trabajo", "--estado", "--libro", str(p))
    print("\n== lo vivo en la cuenta de Vast, ahora:", flush=True)
    return _v("list")


def apagar(prefijo: str) -> int:
    if not prefijo.startswith("expc-") or len(prefijo) < 6:
        print("✗ desde aqui solo se apagan maquinas de experimentos: el prefijo tiene que "
              "empezar por `expc-` (p. ej. `expc-bork-`). Lo demas, con el lanzador.")
        return 2
    return _v("trabajo", "--apagar", prefijo)


def main(argv: list[str]) -> int:
    """Sale con 0 SIEMPRE: el coordinador lee cualquier otro codigo como «el ejecutor
    fallo» y entonces no corre los encargados, y el texto no llegaria a Telegram. El
    resultado va en el TEXTO (el mismo criterio que `cerrable.mjs --exit0`)."""
    if not argv or argv[0] in ("estado", "--estado"):
        estado()
    elif argv[0] == "apagar" and len(argv) == 2:
        apagar(argv[1])
    else:
        print("uso: estado | apagar <prefijo-expc->")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
