"""Executa kernels novos, salva saídas e falha no primeiro erro real."""
import argparse
import os
from pathlib import Path
import sys
import time
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import json

RAIZ = Path(__file__).resolve().parents[1]


def executar(nomes=None):
    # Tudo que é configuração transitória fica dentro da pasta ignorada.
    temporarios = RAIZ / ".execucao"
    spec = temporarios / "kernels/python3"
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python — trabalho", "language": "python",
    }))
    os.environ["MPLCONFIGDIR"] = str(temporarios / "matplotlib")
    os.environ["IPYTHONDIR"] = str(temporarios / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(temporarios / "runtime")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    arquivos = [RAIZ / n for n in nomes] if nomes else sorted(RAIZ.glob("conjunto_*/*.ipynb"))
    registros = []
    for arquivo in arquivos:
        print(f"Executando {arquivo.relative_to(RAIZ)}", flush=True)
        inicio = time.monotonic()
        nb = nbformat.read(arquivo, as_version=4)
        cliente = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(arquivo.parent)}})
        cliente.km = cliente.create_kernel_manager()
        cliente.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(spec.parent)])
        try:
            cliente.execute()
        finally:
            nbformat.write(nb, arquivo)
        duracao = round(time.monotonic() - inicio, 2)
        registros.append({"notebook": str(arquivo.relative_to(RAIZ)), "segundos": duracao, "estado": "executado_sem_erros"})
        print(f"Concluído em {duracao}s", flush=True)
    (RAIZ / "resultados/execucao_notebooks.json").write_text(json.dumps(registros, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*")
    args = parser.parse_args()
    executar(args.notebooks or None)
