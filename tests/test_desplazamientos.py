"""La sección `## Desplazamientos` de REGLAS.md es obligatoria desde el 2026-10-09 (regla del dueño, CLAUDE.md).

    python3 -m unittest tests/test_desplazamientos.py

Stdlib pura, como `comprobar.py`. Los experimentos de prueba se crean DENTRO del repo (en un directorio temporal que se
borra) porque `Experimento.rel()` los ubica respecto de la raíz.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
import comprobar  # noqa: E402
from expcnn import experimentos, raiz  # noqa: E402

CINCO = "\n".join(comprobar.SECCIONES_REGLAS) + "\n"


def _exp(base: Path, nombre: str, creado: str, reglas: str):
    d = base / nombre
    d.mkdir()
    (d / "experimento.json").write_text(json.dumps({"id": nombre, "titulo": "t", "pregunta": "p", "creado": creado}))
    (d / "REGLAS.md").write_text(reglas)


class SeccionDesplazamientos(unittest.TestCase):
    def _problemas(self, creado: str, reglas: str) -> list[str]:
        with tempfile.TemporaryDirectory(dir=raiz()) as tmp:
            _exp(Path(tmp), f"{creado}-prueba", creado, reglas)
            return comprobar._problemas_de_reglas(experimentos(Path(tmp)))

    def test_nuevo_sin_seccion_falla(self):
        p = self._problemas("2026-10-09", CINCO)
        self.assertTrue(any(comprobar.SECCION_DESPLAZAMIENTOS in m for m in p), p)

    def test_nuevo_con_seccion_pasa(self):
        self.assertEqual(self._problemas("2026-10-10", CINCO + comprobar.SECCION_DESPLAZAMIENTOS + "\nno reconoce nada\n"), [])

    def test_anterior_no_se_le_exige(self):
        self.assertEqual(self._problemas("2026-10-08", CINCO), [])

    def test_seccion_de_la_plantilla_sin_rellenar_falla(self):
        txt = (RAIZ / comprobar.PLANTILLA_REGLAS).read_text(encoding="utf-8")
        self.assertIn(comprobar.SECCION_DESPLAZAMIENTOS, txt, "la plantilla tiene que traer la sección")
        ini = txt.index(comprobar.SECCION_DESPLAZAMIENTOS); fin = txt.index("## Scripts")
        p = self._problemas("2026-10-09", CINCO + txt[ini:fin])
        self.assertTrue(any(comprobar.SIN_RELLENAR in m for m in p), p)


if __name__ == "__main__":
    unittest.main()
