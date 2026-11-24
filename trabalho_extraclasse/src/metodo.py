"""Rotinas auditáveis compartilhadas pelos quatro notebooks.

As decisões e interpretações estão nos notebooks. Aqui ficam operações repetidas
de leitura, avaliação e previsão, para preservar o mesmo método entre conjuntos.
"""

from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, confusion_matrix,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning


RAIZ = Path(__file__).resolve().parents[1]
CUSTO_FP = 1200.0
CUSTO_FN = 400.0
LIMIAR = CUSTO_FP / (CUSTO_FP + CUSTO_FN)
QUANTIL = CUSTO_FN / (CUSTO_FP + CUSTO_FN)
ATRASO = 15
HORIZONTE = 7
INICIO_TESTE = pd.Timestamp("2017-03-01")
FIM_TESTE = pd.Timestamp("2017-08-29")

NUMERICAS = [
    "lead_time", "stays_in_week_nights", "stays_in_weekend_nights",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "dia_semana", "mes",
]
CATEGORICAS = ["market_segment", "distribution_channel", "reserved_room_type"]
VARIAVEIS = NUMERICAS + CATEGORICAS
PROIBIDAS = [
    "is_canceled", "reservation_status", "reservation_status_date",
    "data_rotulo", "assigned_room_type", "booking_changes", "country",
    "adr", "deposit_type", "days_in_waiting_list", "agent", "company",
    "adults", "children", "babies", "meal", "customer_type",
    "required_car_parking_spaces", "total_of_special_requests",
]


def carregar_dados():
    """Verifica a cópia fixa e registra todas as reduções da população."""
    caminho = RAIZ / "dados/hotels.csv"
    meta = json.loads((RAIZ / "dados/proveniencia.json").read_text())
    assert hashlib.sha256(caminho.read_bytes()).hexdigest() == meta["sha256"]
    bruto = pd.read_csv(caminho, keep_default_na=False, low_memory=False)
    bruto["id_linha"] = np.arange(len(bruto))
    meses = {m: i + 1 for i, m in enumerate([
        "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December",
    ])}
    bruto["data_chegada"] = pd.to_datetime(pd.DataFrame({
        "year": bruto.arrival_date_year,
        "month": bruto.arrival_date_month.map(meses),
        "day": bruto.arrival_date_day_of_month,
    }))
    bruto["data_status"] = pd.to_datetime(bruto.reservation_status_date)
    bruto["data_criacao_aproximada"] = (
        bruto.data_chegada - pd.to_timedelta(bruto.lead_time, unit="D")
    )
    bruto["noites"] = bruto.stays_in_week_nights + bruto.stays_in_weekend_nights
    bruto["dia_semana"] = bruto.data_chegada.dt.dayofweek
    bruto["mes"] = bruto.data_chegada.dt.month
    bruto["data_rotulo"] = bruto.data_status + pd.Timedelta(days=1)
    registros = [{"etapa": "Arquivo integral", "antes": len(bruto), "depois": len(bruto)}]
    dados = bruto.copy()
    filtros = [
        ("Somente City Hotel", lambda d: d.hotel.eq("City Hotel")),
        ("Criada antes do dia da chegada", lambda d: d.lead_time.ge(1)),
        ("Estadia planejada entre 1 e 14 noites", lambda d: d.noites.between(1, 14)),
        ("Ativa no fim da véspera", lambda d:
         d.is_canceled.eq(0) | d.data_status.ge(d.data_chegada)),
    ]
    for nome, filtro in filtros:
        antes = len(dados)
        dados = dados.loc[filtro(dados)].copy()
        registros.append({"etapa": nome, "antes": antes, "depois": len(dados)})
    registros = pd.DataFrame(registros)
    registros["excluidas"] = registros.antes - registros.depois
    # O embargo só é válido se a auditoria comprovar a maturação máxima.
    atraso = (dados.data_rotulo - dados.data_chegada).dt.days
    assert atraso.between(1, ATRASO).all(), "Maturação excede o protocolo: redesenhar antes de modelar."
    assert dados.loc[dados.is_canceled.eq(1), "data_status"].eq(
        dados.loc[dados.is_canceled.eq(1), "data_chegada"]
    ).all()
    assert not set(VARIAVEIS) & set(PROIBIDAS)
    assert dados[VARIAVEIS].notna().all().all()
    return bruto, dados.sort_values(["data_chegada", "id_linha"]), registros


def construir_serie(dados):
    calendario = pd.date_range("2015-07-01", "2017-08-31", freq="D")
    serie = dados.groupby("data_chegada").agg(
        cancelamentos=("is_canceled", "sum"), exposicao=("id_linha", "size"),
        ultimo_rotulo=("data_rotulo", "max"),
    ).reindex(calendario)
    serie.index.name = "data"
    serie[["cancelamentos", "exposicao"]] = serie[
        ["cancelamentos", "exposicao"]
    ].fillna(0).astype(int)
    assert len(serie) >= 100 and len(serie) >= 2 * 7
    assert serie.cancelamentos.sum() == dados.is_canceled.sum()
    return serie


def particoes_a(dados):
    treino = dados.loc[(dados.data_chegada < "2016-11-01") &
                       (dados.data_rotulo <= "2016-11-01")].copy()
    calibracao = dados.loc[dados.data_chegada.between("2016-11-01", "2016-12-17")].copy()
    validacao = dados.loc[dados.data_chegada.between("2017-01-01", "2017-02-14")].copy()
    teste = dados.loc[dados.data_chegada.between(INICIO_TESTE, FIM_TESTE)].copy()
    assert calibracao.data_rotulo.max() <= pd.Timestamp("2017-01-01")
    assert validacao.data_rotulo.max() <= INICIO_TESTE
    conjuntos = [treino, calibracao, validacao, teste]
    assert len(set().union(*(set(d.id_linha) for d in conjuntos))) == sum(map(len, conjuntos))
    return dict(zip(["treino", "calibracao", "validacao", "teste"], conjuntos))


def criar_pipeline(tipo):
    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("padronizar", StandardScaler()),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preparo = ColumnTransformer([
        ("numerico", numerico, NUMERICAS), ("categorico", categorico, CATEGORICAS),
    ])
    if tipo == "Logística":
        estimador = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    else:
        estimador = HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, min_samples_leaf=80,
            l2_regularization=5.0, early_stopping=False, random_state=42,
        )
    return Pipeline([("preparo", preparo), ("estimador", estimador)])


def custo_classificacao(y, acao, custo_fp=CUSTO_FP, custo_fn=CUSTO_FN):
    y, acao = np.asarray(y), np.asarray(acao)
    return custo_fp * ((acao == 1) & (y == 0)) + custo_fn * ((acao == 0) & (y == 1))


def resumo_classificacao(y, p, limiar=LIMIAR):
    y, p = np.asarray(y), np.asarray(p)
    acao = p >= limiar
    tn, fp, fn, tp = confusion_matrix(y, acao, labels=[0, 1]).ravel()
    return {
        "n": len(y), "eventos_positivos": int(y.sum()),
        "ROC_AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "PR_AUC_AP": average_precision_score(y, p) if y.sum() else np.nan,
        "Brier": brier_score_loss(y, p),
        "precisao": precision_score(y, acao, zero_division=0),
        "recall": recall_score(y, acao, zero_division=0),
        "revendas": int(acao.sum()), "FP": int(fp), "FN": int(fn),
        "TP": int(tp), "TN": int(tn),
        "custo_total": float(custo_classificacao(y, acao).sum()),
        "custo_por_reserva": float(custo_classificacao(y, acao).mean()),
    }


def ajustar_classificadores(partes):
    treino, cal = partes["treino"], partes["calibracao"]
    resultados, previsoes, objetos = [], {}, {}
    for tipo in ["Logística", "Gradient boosting"]:
        modelo = criar_pipeline(tipo)
        modelo.fit(treino[VARIAVEIS], treino.is_canceled)
        p_cal = np.clip(modelo.predict_proba(cal[VARIAVEIS])[:, 1], 1e-6, 1 - 1e-6)
        # Nenhuma validação aleatória: o calibrador vê somente o bloco seguinte.
        calibrador = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
        calibrador.fit(logit(p_cal).reshape(-1, 1), cal.is_canceled)
        objetos[tipo] = (modelo, calibrador)
    prevalencia = treino.is_canceled.mean()
    grupo = treino.lead_time.ge(90)
    riscos = treino.groupby(grupo).is_canceled.mean()
    for parte, d in partes.items():
        previsoes[parte] = {
            "Nunca revender": np.zeros(len(d)),
            "Prevalência treino": np.full(len(d), prevalencia),
            "Regra lead ≥90": d.lead_time.ge(90).map(riscos).to_numpy(),
        }
        for tipo, (modelo, calibrador) in objetos.items():
            p = np.clip(modelo.predict_proba(d[VARIAVEIS])[:, 1], 1e-6, 1 - 1e-6)
            previsoes[parte][tipo + " sem calibração"] = p
            previsoes[parte][tipo + " calibrado"] = calibrador.predict_proba(
                logit(p).reshape(-1, 1)
            )[:, 1]
        for nome, p in previsoes[parte].items():
            resultados.append({"parte": parte, "modelo": nome,
                               **resumo_classificacao(d.is_canceled, p)})
    tabela = pd.DataFrame(resultados)
    # Seleção só no bloco de validação; Brier desempata políticas de custo igual.
    elegiveis = tabela.loc[tabela.parte.eq("validacao")].sort_values(
        ["custo_por_reserva", "Brier", "modelo"]
    )
    vencedor = elegiveis.iloc[0].modelo
    return tabela, previsoes, objetos, vencedor


MODELOS_B = ["Zero", "Ingênuo sazonal", "Média 28 dias", "SARIMA"]


def prever_ponto(treino, passos, modelo):
    valores = np.asarray(treino, dtype=float)
    if modelo == "Zero":
        return np.zeros(passos), True
    if modelo == "Ingênuo sazonal":
        return np.resize(valores[-7:], passos), True
    if modelo == "Média 28 dias":
        return np.full(passos, valores[-28:].mean()), True
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        ajuste = SARIMAX(
            treino.astype(float), order=(1, 0, 1), seasonal_order=(1, 0, 0, 7),
            trend="c", enforce_stationarity=True, enforce_invertibility=True,
        ).fit(disp=False, maxiter=150)
    convergiu = bool(ajuste.mle_retvals.get("converged", False))
    return np.maximum(0, np.asarray(ajuste.forecast(passos))), convergiu


def origens_b():
    return {
        "calibracao": pd.date_range("2016-01-01", "2016-08-10", freq="7D"),
        "validacao": pd.date_range("2016-09-01", "2017-02-08", freq="7D"),
        "teste": pd.date_range(INICIO_TESTE, FIM_TESTE - pd.Timedelta(days=6), freq="7D"),
    }


def backtest_b(serie, origens, modelos=MODELOS_B):
    linhas = []
    for origem in origens:
        limite = origem - pd.Timedelta(days=ATRASO)
        treino = serie.loc[serie.index < limite, "cancelamentos"]
        assert len(treino) >= 100
        conhecidos = serie.loc[treino.index, "ultimo_rotulo"].dropna()
        assert conhecidos.le(origem).all()
        assert treino.index[-1] == origem - pd.Timedelta(days=ATRASO + 1)
        escala = np.abs(treino.to_numpy()[7:] - treino.to_numpy()[:-7]).mean()
        assert escala > 0
        datas = pd.date_range(origem, periods=HORIZONTE, freq="D")
        for modelo in modelos:
            todos, convergiu = prever_ponto(treino, ATRASO + HORIZONTE, modelo)
            previsao = todos[-HORIZONTE:]
            assert np.isfinite(previsao).all()
            for h, (data, p) in enumerate(zip(datas, previsao), 1):
                linhas.append({
                    "origem": origem, "data": data, "horizonte": h,
                    "modelo": modelo, "real": int(serie.loc[data, "cancelamentos"]),
                    "ponto": float(p), "escala_MASE": escala,
                    "fim_treino": treino.index[-1], "n_treino": len(treino),
                    "convergiu": convergiu,
                })
    resultado = pd.DataFrame(linhas)
    resultado["erro"] = resultado.real - resultado.ponto
    return resultado


def custo_quantidade(real, quantidade, custo_falta=CUSTO_FN, custo_excesso=CUSTO_FP):
    real, quantidade = np.asarray(real), np.asarray(quantidade)
    return (custo_falta * np.maximum(real - quantidade, 0) +
            custo_excesso * np.maximum(quantidade - real, 0))


def distribuir_previsoes(previsoes, erros_calibracao):
    """Distribuição empírica discreta: ponto + erros genuínos fora da amostra.

    O pool de erros é congelado antes da seleção e do teste. É uma aproximação
    marginal, sem garantia de cobertura sob mudança de regime.
    """
    saida = previsoes.copy()
    for nome, grupo in saida.groupby("modelo"):
        erros = erros_calibracao.loc[erros_calibracao.modelo.eq(nome), "erro"].to_numpy()
        assert len(erros) >= 100
        amostras = np.floor(np.maximum(0, grupo.ponto.to_numpy()[:, None] + erros[None, :]) + 0.5)
        qs = np.quantile(amostras, [0.1, 0.25, 0.5, 0.9], axis=1, method="inverted_cdf")
        for coluna, valores in zip(["inferior80", "q25", "mediana", "superior80"], qs):
            saida.loc[grupo.index, coluna] = valores.astype(int)
        for quantil, coluna in [(0.25, "q25")]:
            # Verifica a decisão inteira contra suas duas vizinhas na distribuição prevista.
            q = saida.loc[grupo.index, coluna].to_numpy()
            esperado = np.array([
                custo_quantidade(amostras, np.maximum(0, q + deslocamento)[:, None]).mean(axis=1)
                for deslocamento in [-1, 0, 1]
            ])
            assert np.all(esperado[1] <= esperado.min(axis=0) + 1e-8)
    for coluna in ["inferior80", "q25", "mediana", "superior80"]:
        saida[coluna] = saida[coluna].astype(int)
    saida["cobriu80"] = saida.real.between(saida.inferior80, saida.superior80)
    saida["largura80"] = saida.superior80 - saida.inferior80
    saida["custo_q25"] = custo_quantidade(saida.real, saida.q25)
    saida["custo_mediana"] = custo_quantidade(saida.real, saida.mediana)
    saida["custo_zero"] = custo_quantidade(saida.real, 0)
    saida["interval_score80"] = (
        saida.largura80 + 10 * np.maximum(saida.inferior80 - saida.real, 0) +
        10 * np.maximum(saida.real - saida.superior80, 0)
    )
    assert (saida.inferior80 <= saida.superior80).all()
    return saida


def metricas_b(previsoes):
    linhas = []
    for (modelo, origem), d in previsoes.groupby(["modelo", "origem"]):
        erro = d.real - d.ponto
        linha = {
            "modelo": modelo, "origem": origem, "MAE": erro.abs().mean(),
            "RMSE": np.sqrt(np.mean(erro**2)),
            "MASE_treino": erro.abs().mean() / d.escala_MASE.iloc[0],
        }
        for coluna in ["cobriu80", "largura80", "interval_score80", "custo_q25", "custo_mediana", "custo_zero"]:
            if coluna in d:
                linha[coluna] = d[coluna].mean()
        linhas.append(linha)
    por_origem = pd.DataFrame(linhas)
    medidas = [c for c in por_origem.columns if c not in ["modelo", "origem"]]
    media = por_origem.groupby("modelo")[medidas].mean()
    media["desvio_MAE"] = por_origem.groupby("modelo").MAE.std()
    media["desvio_RMSE"] = por_origem.groupby("modelo").RMSE.std()
    media["desvio_MASE"] = por_origem.groupby("modelo").MASE_treino.std()
    media["razao_MAE_sazonal_aula"] = media.MAE / media.loc["Ingênuo sazonal", "MAE"]
    return media.sort_values("MAE"), por_origem


def salvar_json(nome, objeto):
    pasta = RAIZ / "resultados"
    pasta.mkdir(exist_ok=True)
    (pasta / nome).write_text(json.dumps(objeto, ensure_ascii=False, indent=2, default=str) + "\n")


def moeda(valor):
    return (f"R$ {valor:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
