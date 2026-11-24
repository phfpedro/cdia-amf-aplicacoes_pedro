"""Confere artefatos executados e invariantes que sustentam a avaliação."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
import pandas as pd
import nbformat

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from metodo import carregar_dados, construir_serie, custo_classificacao, custo_quantidade


def verificar():
    verificacoes = []
    for arquivo in sorted(RAIZ.glob("conjunto_*/*.ipynb")):
        nb = nbformat.read(arquivo, as_version=4)
        nbformat.validate(nb)
        assert arquivo.with_suffix(".md").exists()
        assert "Cinco decisões" in nb.cells[0].source
        assert "R$ 400" in nb.cells[0].source and "R$ 1.200" in nb.cells[0].source
        for i, celula in enumerate(nb.cells):
            if celula.cell_type != "code":
                continue
            assert celula.execution_count is not None, (arquivo.name, "Célula não executada", i)
            assert i > 0 and nb.cells[i - 1].cell_type == "markdown"
            assert i + 1 < len(nb.cells) and nb.cells[i + 1].cell_type == "markdown"
            assert not any(o.output_type == "error" for o in celula.outputs)
        assert "A conclusão será consolidada" not in "\n".join(c.source for c in nb.cells)
        assert "A recomendação será consolidada" not in "\n".join(c.source for c in nb.cells)
        verificacoes.append(f"Notebook executado, válido e acompanhado de prompt: {arquivo.relative_to(RAIZ)}")

    bruto, dados, filtros = carregar_dados()
    serie = construir_serie(dados)
    a = pd.read_csv(RAIZ / "resultados/classificacao_teste_eventos.csv", parse_dates=["data_chegada", "data_rotulo"])
    b = pd.read_csv(RAIZ / "resultados/previsoes_b_teste.csv", parse_dates=["origem", "data", "fim_treino"])
    cal = pd.read_csv(RAIZ / "resultados/previsoes_b_calibracao.csv", parse_dates=["data", "origem", "fim_treino"])
    val = pd.read_csv(RAIZ / "resultados/previsoes_b_validacao.csv", parse_dates=["data", "origem", "fim_treino"])
    resumo_a = json.loads((RAIZ / "resultados/resumo_a.json").read_text())
    resumo_b = json.loads((RAIZ / "resultados/resumo_b.json").read_text())
    assert a.id_linha.is_unique
    assert a.id_linha.isin(dados.id_linha).all()
    assert np.allclose(a.custo, custo_classificacao(a.is_canceled, a.revender))
    for fase in [cal, val, b]:
        assert ((fase.origem - fase.fim_treino).dt.days == 16).all()
        assert ((fase.data - fase.origem).dt.days == fase.horizonte - 1).all()
        assert fase.groupby(["modelo", "origem"]).size().eq(7).all()
        assert not fase.duplicated(["modelo", "data"]).any()
        for nome, g in fase.groupby("modelo"):
            assert set(g.data) == set(fase.loc[fase.modelo.eq("Ingênuo sazonal"), "data"])
    assert cal.data.max() + pd.Timedelta(days=15) <= val.origem.min()
    assert val.data.max() + pd.Timedelta(days=15) <= b.origem.min()
    assert a.data_chegada.min() == b.data.min() and a.data_chegada.max() == b.data.max()
    esperado = a.groupby("data_chegada").is_canceled.sum().reindex(sorted(b.data.unique()), fill_value=0)
    for modelo, grupo in b.groupby("modelo"):
        por_dia = grupo.set_index("data").sort_index()
        assert np.array_equal(por_dia.real.to_numpy(), esperado.to_numpy())
        assert np.allclose(grupo.custo_q25, custo_quantidade(grupo.real, grupo.q25))
        assert grupo.q25.ge(0).all()
        assert grupo.inferior80.le(grupo.q25).all()
        assert grupo.q25.le(grupo.superior80).all()
        assert grupo.q25.eq(np.floor(grupo.q25)).all()
        assert np.isfinite(grupo[["ponto", "inferior80", "superior80"]].to_numpy()).all()
    assert resumo_a["positivos_teste"] == resumo_b["real_total"]
    assert resumo_a["baseline_teste"] == resumo_b["custo_zero"]
    # Uma única reserva deve recuperar exatamente a mesma matriz de perdas em A e B.
    y = np.array([0, 0, 1, 1])
    q = np.array([0, 1, 0, 1])
    assert np.array_equal(custo_classificacao(y, q), custo_quantidade(y, q))
    verificacoes.extend([
        "Mesma população e mesmo alvo diário em A e B, inclusive no teste",
        "Perdas de A equivalentes às de B para uma reserva; valores exportados recalculados",
        "Embargo de 15 dias e cronologia da calibração/seleção/teste respeitados",
        "Sem sobreposição de alvos entre origens; mesmas datas para todos os modelos",
        "Quantis inteiros, não negativos, ordenados e com previsões finitas",
    ])
    fontes = json.loads((RAIZ / "dados/proveniencia.json").read_text())
    for documento in fontes["documentos_preservados"]:
        assert hashlib.sha256((RAIZ / documento["arquivo"]).read_bytes()).hexdigest() == documento["sha256"]
    verificacoes.append("Hashes da base e dos documentos preservados conferidos")
    resultado = {"estado": "aprovado", "verificacoes": verificacoes}
    (RAIZ / "resultados/verificacao_entrega.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2) + "\n")
    for v in verificacoes:
        print("OK:", v)


if __name__ == "__main__":
    verificar()
