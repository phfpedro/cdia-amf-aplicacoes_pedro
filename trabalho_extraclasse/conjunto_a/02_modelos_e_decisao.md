# Registro de instruções para geração assistida por IA

## Pedido original do aluno

“ok inicie o desenvolvimento, não vou estar na frente do pc, mais tarde eu volto”

O contexto anterior autoriza desenvolver os dois conjuntos, documentar o estudo em Markdown e preservar os aprendizados para uso futuro no King, priorizando a entrega acadêmica.

## Especificação de implementação registrada antes da geração

As instruções abaixo foram formuladas pelo assistente nesta execução para orientar a geração. Este documento não é uma transcrição literal de uma conversa separada nem alega que todo o desenvolvimento veio de um único prompt. O histórico de revisões relevantes deve ser acrescentado ao final.

Produza um notebook didático em português: texto antes do código, interpretação depois de números e gráficos, caminhos relativos e nenhum resultado inventado. Preserve os materiais do professor. Dados: Hotel Booking Demand, espelho TidyTuesday de 11/02/2020. A população é City Hotel, estadias planejadas de 1 a 14 noites, lead_time >=1 e reservas ainda ativas no início da chegada; exclua cancelamentos conhecidos até a véspera. O alvo é cancelamento na chegada ou no-show, contado por reserva e, em B, por data de chegada. Não elimine linhas idênticas sem identificador que comprove duplicação. Use R$400 para vaga não revendida que ficaria vazia e R$1200 para revenda indevida; declare hipóteses de primeira noite, procura de reposição e perdas incrementais, sem afirmar lucro real. Limiar 0,75; quantil 0,25. Use somente atributos defensáveis na véspera e audite datas de conhecimento dos rótulos.

Relembre cinco decisões e custos. Implemente baselines constante/majoritário e regra por lead_time, logística e gradient boosting com pipelines ajustados só no treino. Treino anterior a novembro/2016, calibração novembro a 17/dezembro/2016, seleção janeiro a 14/fevereiro/2017 e teste março a agosto/2017; as fronteiras deixam maturação de 15 dias. Ajuste calibração sigmoid apenas no bloco de calibração, escolha por custo de validação e use Brier como desempate. Mostre métricas, curva de calibração, matrizes de confusão, custo por limiar sem escolher limiar pelo teste, sensibilidade ao custo e tabelas mensais. Congele política antes do teste. Explique se não revender for a melhor política.

## Revisões durante a execução — 17/11/2026

- A inspeção local confirmou a população de 44.393 reservas, com 1.155 positivos e 793 dias. O embargo de 15 dias foi validado pelas datas dos rótulos.
- As fronteiras foram fixadas com maturação: em A, calibração até 17/12/2016 e seleção até 14/02/2017; em B, 32 origens de calibração, 23 de seleção e 26 de teste.
- As leituras numéricas finais foram escritas somente depois da execução, a partir dos JSON/CSV exportados, pelo script `consolidar_resultados.py`. As escolhas de modelos e os custos não foram alterados após abrir o teste.
- O teste revelou ausência de ganho econômico em A e perda frente ao baseline sem revendas em B. O resultado negativo foi preservado e explicado.
- Foi esclarecida a diferença entre ponto bruto de previsão, mediana da distribuição corrigida e quantil econômico, assim como a sobrecobertura global e a cobertura insuficiente em abril.
- Foram verificados execução dos quatro notebooks, hashes, maturação, equivalência de custos, mesma população e consistência diária entre os conjuntos.

Os scripts e arquivos de evidência preservam a implementação final. Este registro distingue as instruções iniciais das interpretações e ajustes feitos depois de observar as saídas; não pretende ser uma transcrição integral das ferramentas da sessão.
