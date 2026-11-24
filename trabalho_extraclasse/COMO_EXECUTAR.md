# Executar e revisar o trabalho

Os quatro notebooks são independentes e abrem a mesma cópia de dados. As rotinas compartilhadas ficam em `src/metodo.py`; envie essa pasta junto com os notebooks. O hash em `dados/proveniencia.json` é conferido em toda leitura.

Use Python 3.9 a 3.12. Na raiz deste repositório, execute os comandos abaixo.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r trabalho_extraclasse/requirements.txt
.venv/bin/python trabalho_extraclasse/scripts/executar_notebooks.py
.venv/bin/python trabalho_extraclasse/scripts/consolidar_resultados.py
.venv/bin/python trabalho_extraclasse/scripts/verificar_entrega.py
```

No VS Code, selecione o interpretador `.venv/bin/python` e execute os notebooks na ordem dos conjuntos A e B. Não há dependência de estado entre kernels. Os números e gráficos ficam salvos nos próprios notebooks; não é necessário reexecutar apenas para ler a entrega.

O executor cria kernels novos e configurações transitórias em `.execucao/`, ignorada no Git. Ele para no primeiro erro e não permite saídas de erro como resultado aceito. As verificações metodológicas também interrompem a execução se houver inconsistência.

Os arquivos em `resultados/` preservam previsões, métricas, auditoria e resumos. Os `.md` com o mesmo nome dos notebooks registram as instruções usadas para geração assistida por IA, com a distinção entre o pedido do aluno e a especificação formulada pelo assistente.

Os dados são públicos, usados com atribuição a Antonio, Almeida e Nunes (2019), através do espelho TidyTuesday. Consulte `fontes/` e `dados/proveniencia.json`. Os custos são hipóteses e os resultados financeiros são perdas simuladas, não lucro observado.
