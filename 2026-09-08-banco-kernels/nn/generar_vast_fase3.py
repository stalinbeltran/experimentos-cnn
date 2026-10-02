"""Genera banco-k/nn/vast-fase3.json: una maquina por k evalua los kernels de ese k
con `identidad` y el aleatorio de su k corridos EN ESA MAQUINA (la regla de deriva de la
fase 1: |Δ IoU eval| = 0,0012 > 0,001). Uso: python3 generar_fase3.py <prefijos...>"""
import json, sys
from pathlib import Path
EXP = Path.home() / "src/experimentos-cnn/2026-09-08-banco-kernels"
KS = (3, 5, 7, 9, 11, 13, 15, 17, 19)
prefijos = sys.argv[1:] or ["bork", "borae", "borpca"]
INSTALL = json.loads((EXP / "nn/vast-fase1.json").read_text())["install"]
trabajos = []
for k in KS:
    kernels = sorted(p.stem for pre in prefijos for p in EXP.glob(f"kernels/{pre}-k{k:02d}*.npy"))
    if not kernels:
        continue
    d_al = "aleatorio" if k == 9 else f"aleatorio-k{k}"
    run = ["cd ..",
           "lscpu | grep -E '^(Model name|CPU\\(s\\))'",
           "export OMP_NUM_THREADS=${TRABAJO_VCPU:-4}",
           "echo \"vCPU pagadas: ${TRABAJO_VCPU:-?} · OMP_NUM_THREADS=$OMP_NUM_THREADS\"",
           "fallos=0",
           "# 1. identidad, 10 semillas, EN ESTA MAQUINA (la calibracion la corrio en el dev)",
           "../.venv/bin/python -u -c \"import sys; sys.path.insert(0, 'nn'); from calibrar import _grupo; _grupo('identidad', lambda s: None)\" || fallos=$((fallos + 1))",
           "# 2. los kernels; el primero crea el aleatorio de este k, tambien en esta maquina",
           f"for K in {' '.join('kernels/' + n + '.npy' for n in kernels)}; do",
           "  ../.venv/bin/python -u nn/evaluar_kernel.py --kernel \"$K\" || fallos=$((fallos + 1))",
           "done",
           "echo \"grupos con fallo: $fallos\"",
           "exit $fallos"]
    trae = []
    for n in kernels + ["identidad", d_al]:
        trae.append(f"../resultados/{n}")
        trae += [f"../resultados/{n}-s{s}" for s in range(10)]
    trae += [f"../kernels/{d_al}-r{s}.npy" for s in range(10)]
    trabajos.append({"id": f"k{k:02d}", "run": "\n".join(run), "trae": trae})
d = {
    "experimento": "banco-k",
    "descripcion": ("Fase 3 del plan de kernels: evaluar en el banco los kernels de bor-k, bor-ae y bor-pca, UNA "
                    "maquina por k, con `identidad` y el aleatorio de su k corridos en ESA maquina (regla de deriva "
                    "de la fase 1, escrita antes de medir: |Δ IoU eval| 0,0012 > 0,001)."),
    "maquina": {"cpus": 4, "max_cpus": 8, "min_ram": 8, "cpu": "", "disk_gb": 24, "max_price": 0.12},
    "envia": [
        {"origen": "../..", "destino": "experimentos-cnn",
         "excluye": [".git", ".venv", "datos", "__pycache__", "experimentos_cnn.egg-info",
                     "resultados-r20260908-descartado", "muestras", "resultados", "pesos"]},
        {"origen": "$EXPCNN_DATOS/experimentos-cnn/parrafos1000-584px-r4-r20260908b",
         "destino": "foveal-vision-data/experimentos-cnn/parrafos1000-584px-r4-r20260908b"},
    ],
    "entorno": {"EXPCNN_DATOS": "/root/trabajo/foveal-vision-data", "PYTHONUNBUFFERED": "1"},
    "install": INSTALL,
    "trabajos": trabajos,
    "notas": [
        "SIN ningun `resultados/` en el payload (`excluye`): asi `identidad` y el aleatorio de cada k se recalculan en la maquina que evalua sus kernels. Con los del dev, la comparacion mezclaria kernel con maquina.",
        "Lo traido que choca con lo local (identidad*, y aleatorio* de k=7,9,11) NO pisa nada: queda en el libro, resultados/vast/fase3/<k>/traido/. Lo nuevo (cada kernel y los aleatorios de k=3,5,13,15,17,19) va a su sitio.",
        "Los hilos: OMP_NUM_THREADS = TRABAJO_VCPU (las vCPU que se pagan). nproc cuenta las del host.",
        "`run` y `trae` son relativos a ESTE fichero en la maquina (`nn/`): por eso `cd ..` y `../resultados`.",
    ],
}
out = EXP / "nn/vast-fase3.json"
out.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{out}: {len(trabajos)} trabajos · kernels por k: " + ", ".join(
    f"{t['id']}={sum(1 for x in t['trae'] if x.count('/') == 2 and '-s' not in x.split('/')[-1][-3:] and not x.endswith('.npy')) - 2}" for t in trabajos))
