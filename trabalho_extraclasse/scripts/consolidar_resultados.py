"""Insere conclusões em células de texto usando apenas evidências executadas."""
from pathlib import Path
import json
import sys
import nbformat
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from metodo import moeda


def numero(n, casas=0):
    return f"{n:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def consolidar():
    auditoria = json.loads((RAIZ / "resultados/auditoria.json").read_text())
    a = json.loads((RAIZ / "resultados/resumo_a.json").read_text())
    b = json.loads((RAIZ / "resultados/resumo_b.json").read_text())
    am = pd.read_csv(RAIZ / "resultados/classificacao_mensal.csv")
    bm = pd.read_csv(RAIZ / "resultados/serie_mensal.csv")
    bt = pd.DataFrame(b["teste"]).set_index("modelo")
    bv = pd.DataFrame(b["validacao"]).set_index("modelo")
    at = pd.DataFrame(a["teste"]).set_index("modelo")
    ganho_mae = 1 - bt.loc[b["vencedor"], "razao_MAE_sazonal_aula"]
    custo_extra = b["custo_q25"] - b["custo_zero"]
    poupanca_mediana = b["custo_mediana"] - b["custo_q25"]
    previsoes = pd.read_csv(RAIZ / "resultados/previsoes_b_teste.csv")
    escolhido = previsoes.loc[previsoes.modelo.eq(b["vencedor"])]
    excesso = int((escolhido.q25 - escolhido.real).clip(lower=0).sum())
    falta = int((escolhido.real - escolhido.q25).clip(lower=0).sum())
    assert b["custo_q25"] == excesso * 1200 + falta * 400
    conclusoes = {
        "RESULTADOS_A1": f"""## Conclusão da auditoria

A cópia contém **{numero(119390)} reservas**. Depois dos filtros explícitos,
restam **{numero(auditoria['n'])} reservas** e **{numero(auditoria['positivos'])}
cancelamentos tardios/no-shows**, taxa de **{numero(auditoria['taxa']*100,2)}%**.
A série tem **{auditoria['dias']} dias**, com {auditoria['dias_sem_exposicao']} dias
sem reserva elegível, e satisfaz os requisitos de tamanho e sazonalidade semanal.

O atraso máximo de conhecimento do rótulo é 15 dias na população auditada.
O recorte reduz a amostra, mas evita usar cancelamentos já conhecidos como
se fossem riscos futuros. A escolha da base está aprovada **para este estudo
retrospectivo sob as hipóteses de reconstrução descritas**, sem afirmar que
dispomos de snapshots operacionais perfeitos.

Há 44 positivos na validação de A: poucas observações para estabelecer a
qualidade de probabilidades extremas. Por isso, qualquer conclusão positiva
de automação exigiria evidência adicional. O próximo notebook mede a perda
das políticas sem alterar esse desenho.
""",
        "RESULTADOS_A2": f"""## Recomendação final

A seleção por custo, com Brier como desempate, escolheu **{a['vencedor']}**.
Todas as alternativas tiveram a mesma perda na validação no limiar de 0,75.
No teste de **{numero(a['n_teste'])} reservas**, com **{a['positivos_teste']}**
cancelamentos tardios/no-shows, a política escolhida fez **{a['revendas']} revendas**.
Sua perda simulada foi **{moeda(a['custo_teste'])}**, idêntica à de nunca revender:
**economia incremental de {moeda(a['economia'])}**. Precisão e recall da ação são
zero, pois nenhum caso cruza o limiar. Isso é uma decisão conservadora, não
identificação bem-sucedida dos faltantes.

Os rankings dos modelos flexíveis foram fracos no período: ROC-AUC de
{numero(at.loc['Logística sem calibração','ROC_AUC'],3)} na logística e
{numero(at.loc['Gradient boosting sem calibração','ROC_AUC'],3)} no boosting.
O desvio entre os seis meses do custo por reserva foi
{moeda(a['desvio_custo_mensal_por_reserva'])}; é uma descrição da variabilidade,
não um intervalo de confiança. Agosto tem apenas 29 dias no teste comum a A e B.

**Para a gestão:** mantenha a política sem revenda automática nesta população.
Não há economia demonstrada que sustente adicionar o modelo à decisão.
Registre o custo de oportunidades perdidas e os casos de não comparecimento.
Antes de outro piloto, obtenha snapshots históricos, demanda de reposição,
inventário e custos reais. Preserve o limiar como função dos custos; não o reduza
até conseguir uma recomendação positiva.

O recorte é deliberadamente estreito: os cancelamentos anteriores à véspera
podem ser tratados por uma operação de inventário conhecida, mas estão fora
do risco aqui modelado. Este resultado não demonstra que todo problema de
cancelamento de hotéis seja imprevisível.
""",
        "RESULTADOS_B1": f"""## Conclusão do diagnóstico

Somar a população de A preserva exatamente **{auditoria['positivos']} eventos
positivos em {auditoria['dias']} dias**. A série contém muitos zeros e exposição
variável; contagens grandes isoladas não podem ser interpretadas apenas como
mudança no risco individual. O período inicial tem seis dias sem evento elegível,
preservados com zero. Não há recorte silencioso das bordas para melhorar métricas.

A avaliação tem 32 origens para calibrar os erros, 23 para selecionar modelos
e **26 semanas não sobrepostas no teste**, totalizando 182 dias. Em cada origem,
o modelo prevê 22 passos desde o fim do treino: 15 de embargo e 7 de operação.
Não há observações reais do trecho intermediário usadas na previsão.

O diagnóstico justifica experimentar estrutura semanal; ele não demonstra que
SARIMA será superior. Essa afirmação depende do próximo notebook.
""",
        "RESULTADOS_B2": f"""## Recomendação final: previsão melhor não bastou

**{b['vencedor']}** foi selecionada antes do teste: empatou com a política zero
no custo da validação (**{moeda(bv.loc[b['vencedor'],'custo_q25'])} por dia**) e
venceu o desempate por MAE. A regra de desempate estava definida antes de abrir
o teste e foi preservada, mesmo com o resultado desfavorável posterior.

Nas **{b['n_origens_teste']} origens e {b['n_dias_teste']} dias de teste**, o MAE
foi **{numero(bt.loc[b['vencedor'],'MAE'],3)} reservas/dia**, contra
{numero(bt.loc['Ingênuo sazonal','MAE'],3)} do ingênuo sazonal: redução descritiva
de **{numero(ganho_mae*100,1)}%**. O MASE com escala no treino foi
{numero(bt.loc[b['vencedor'],'MASE_treino'],3)}; a razão de MAEs usada na aula foi
{numero(bt.loc[b['vencedor'],'razao_MAE_sazonal_aula'],3)}. O desvio semanal do MAE
foi {numero(bt.loc[b['vencedor'],'desvio_MAE'],3)}; uma média menor não demonstra,
sozinha, superioridade estatística. SARIMA teve MAE menor no teste, mas não foi
escolhido na validação e não é promovido retrospectivamente.

A banda nominal de 80% cobriu **{numero(b['cobertura80']*100,1)}%** dos dias,
com largura média de **{numero(b['largura80'],2)} reservas**. Isso é sobrecobertura
global, não calibração perfeita: em abril, a cobertura foi apenas 70,0%.
A discretização, o pool fixo de erros e a mudança de regime limitam a leitura.
Nenhum ajuste SARIMA falhou na convergência das origens executadas.

| Política no teste | Perda incremental simulada |
|---|---:|
| Nunca revender | {moeda(b['custo_zero'])} |
| Quantil de custo de 25%, modelo congelado | {moeda(b['custo_q25'])} |
| Mediana da mesma distribuição | {moeda(b['custo_mediana'])} |

O quantil de custo economizou **{moeda(poupanca_mediana)} frente à mediana**, mas
**perdeu {moeda(custo_extra)} frente a nunca revender**. As
{b['recomendacoes_total']} vagas recomendadas produziram {excesso} unidades
excedentes e {falta} oportunidades não aproveitadas no cenário simulado.
O resultado é consistente com os mesmos {b['real_total']} positivos do teste de A;
a perda sem ação é idêntica nos dois conjuntos.

**Para a gestão:** não adote a política experimental de revenda com base neste
teste. Preserve o benchmark sem revendas e investigue a estabilidade da distribuição
antes de outro piloto. O modelo selecionado continua registrado como média móvel;
rejeitar sua implantação não é trocar o vencedor depois de olhar o teste.

## O mesmo problema, dois instrumentos

A usa uma probabilidade por reserva na chegada; B planeja uma quantidade por dia
no início da semana, com histórico maduro mais antigo. O limiar 0,75 e o quantil
0,25 derivam dos mesmos R$ 1.200 contra R$ 400. A não demonstrou ganho econômico;
B mostrou que reduzir erro pontual e usar um quantil teoricamente correto não
compensa uma distribuição de previsão inadequada ao novo período.

Não some economias de A e B: são políticas alternativas sob informações distintas.
Não some quantis diários para afirmar um quantil do total. Os resultados se
restringem à primeira noite da população elegível e às hipóteses de custos,
procura de reposição e reconstrução histórica já declaradas.
""",
    }
    for arquivo in sorted(RAIZ.glob("conjunto_*/*.ipynb")):
        nb = nbformat.read(arquivo, as_version=4)
        for c in nb.cells:
            if c.cell_type != "markdown":
                continue
            c.source = c.source.replace(
                "../../fds2_series_temporais/apresentacoes/TRABALHO_extraclasse.pdf",
                "../fontes/TRABALHO_extraclasse.pdf",
            )
            for marcador, conteudo in conclusoes.items():
                if f"<!-- {marcador} -->" in c.source:
                    c.source = f"<!-- {marcador} -->\n\n" + conteudo.strip()
        if "02_previsao" in arquivo.name:
            for c in nb.cells:
                if (c.cell_type == "markdown" and c.source.startswith("**Leitura:** os resíduos")
                        and "O ponto bruto usado nas métricas" not in c.source):
                    c.source += "\n\nO ponto bruto usado nas métricas não precisa coincidir com a mediana da distribuição corrigida. Para o modelo Zero, a distribuição resultante é a distribuição histórica empírica do alvo nas origens de calibração; seu quantil de 25% recomenda zero. A comparação com a mediana usa essa mesma distribuição, não o ponto bruto."
        nbformat.write(nb, arquivo)
    print("Conclusões numéricas consolidadas em células Markdown, sem alterar previsões ou escolhas.")


if __name__ == "__main__":
    consolidar()
