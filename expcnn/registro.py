"""Quién es quién: id -> carpeta, leyendo los `experimento.json`.

⚠ El nombre de la carpeta NO es la identidad (R16 de las reglas de diseño). La
identidad es el `id` que cada experimento declara en su `experimento.json`, y
por eso se puede renombrar, agrupar o mudar una carpeta sin romper nada: el
registro la vuelve a encontrar escaneando.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ESTADOS = ("abierto", "corriendo", "cerrado", "detenido")
# `gasta` decide qué obligaciones hereda el experimento (ver CLAUDE.md § freno):
#   no            -> nada que apagar
#   entrena-local -> tarda: el freno tiene que verlo, y por eso el script se
#                    llama `entrenar_local.py` (contrato con cerrable.mjs:137)
#   alquila       -> además cuesta dinero: ejecutor de Telegram en el MISMO commit
GASTOS = ("no", "entrena-local", "alquila")

MANIFIESTO = "experimento.json"


def raiz() -> Path:
    """La raíz del repo. Se deduce de la ubicación de este fichero, que es el
    único sitio donde eso es legítimo: este módulo *es* del repo."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Experimento:
    id: str
    carpeta: Path
    datos: dict

    @property
    def titulo(self) -> str:
        return self.datos.get("titulo", "(sin título)")

    @property
    def estado(self) -> str:
        return self.datos.get("estado", "abierto")

    @property
    def gasta(self) -> str:
        return self.datos.get("gasta", "no")

    @property
    def usa_fv(self) -> bool:
        return bool(self.datos.get("usa_fv", False))

    @property
    def entrada(self) -> str | None:
        return self.datos.get("entrada")

    def rel(self) -> str:
        return str(self.carpeta.relative_to(raiz()))


def _manifiestos(base: Path | None = None) -> list[Path]:
    base = base or raiz()
    # A cualquier profundidad: agrupar por tema o por trimestre tiene que seguir
    # siendo un `git mv` y nada más.
    return sorted(p for p in base.rglob(MANIFIESTO) if ".git" not in p.parts)


def experimentos(base: Path | None = None) -> list[Experimento]:
    """Todos los experimentos del repo. Un manifiesto ilegible NO tumba la lista:
    se salta con aviso, como hace el registry del coordinador con un JSON roto."""
    salida = []
    for m in _manifiestos(base):
        try:
            datos = json.loads(m.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - un roto no puede esconder a los sanos
            print(f"⚠ {m}: no se puede leer ({e}); lo salto")
            continue
        ident = datos.get("id")
        if not ident:
            print(f"⚠ {m}: sin `id`; lo salto")
            continue
        salida.append(Experimento(id=ident, carpeta=m.parent, datos=datos))
    return salida


def por_id(ident: str, base: Path | None = None) -> Experimento:
    for e in experimentos(base):
        if e.id == ident:
            return e
    conocidos = ", ".join(sorted(x.id for x in experimentos(base))) or "(ninguno)"
    raise KeyError(f"no hay ningún experimento con id '{ident}'. Hay: {conocidos}")
