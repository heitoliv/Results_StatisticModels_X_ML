# Baseline refatorado — ETTh1 / ETTh2

## Pipeline

```text
01_collect_inspect.py
        ↓
02_statistical_analysis.py
        ↓
03_data_split.py
        ↓
04_preprocess.py
        ├── MinMax
        └── Standard
        ↓
05_windows.py
        ↓
06_baseline.py
        ├── MLP / LSTM
        └── ARIMA / ARIMAX / SARIMA / SARIMAX
        ↓
07_compare_results.py
```

## Estrutura de saída

```text
data/
├── ETDataset-main/ETT-small/   # dados brutos: ETTh1.csv e ETTh2.csv
├── split/
│   ├── ETTh1/
│   └── ETTh2/
├── processed/
│   ├── minmax/
│   │   ├── ETTh1/
│   │   └── ETTh2/
│   └── standard/
│       ├── ETTh1/
│       └── ETTh2/
├── windows/
│   ├── minmax/
│   └── standard/
├── statistics/
└── baseline/
    ├── <scaler>/<dataset>/     # modelos, previsões e métricas individuais
    ├── baseline_all_results.csv
    ├── baseline_scaler_comparison_test.csv
    ├── preprocessing_comparison_test.csv
    └── plots/
```

## Regras metodológicas

- Split cronológico: 70% / 10% / 20%.
- Scaler ajustado exclusivamente no TRAIN.
- Validation e Test recebem somente `transform`.
- OT permanece em `features` para representar seu histórico como entrada.
- `LOOKBACK=24`, `HORIZON=1`.
- Janelas são construídas na série concatenada e atribuídas ao split pelo timestamp do alvo. Isso permite que a primeira previsão de Validation/Test utilize histórico já observado do período anterior, sem usar futuro.
- Métricas principais são calculadas também na escala original por meio de `inverse_transform` genérico, compatível com MinMaxScaler e StandardScaler.
- TEST não participa do treinamento da LSTM; é reservado para avaliação final.

## Métodos estatísticos incluídos

- estatística descritiva;
- ADF (Dickey-Fuller aumentado);
- KPSS;
- tendência linear;
- ACF e PACF;
- Ljung-Box;
- decomposição STL com sazonalidade diária (`period=24`).

Os testes estatísticos usados para decisões de modelagem são executados sobre TRAIN para evitar que informação do futuro influencie decisões do experimento.

## Organização sugerida

`01_eda.py` consolida a inspeção e os gráficos que antes estavam separados. No material original, a inspeção verificava dimensões, tipos, período, nulos, duplicidades e intervalos temporais; os gráficos percorriam as sete variáveis e salvavam uma imagem por série. O refatoramento mantém essas responsabilidades, mas as move para módulos reutilizáveis.

`04_preprocess.py` executa os dois pipelines de forma idêntica: `minmax` e `standard`. Cada scaler é ajustado separadamente e exclusivamente com TRAIN.

`05_windows.py` usa o timestamp do alvo para atribuir cada janela ao split. Assim, a primeira amostra da validação pode utilizar as 24 horas anteriores, mesmo que parte delas esteja no fim de TRAIN. Isso representa um cenário operacional de previsão e não utiliza qualquer valor futuro.

`06_baseline.py` executa o mesmo protocolo para MLP e LSTM em cada combinação `(dataset, scaler)`, salvando métricas normalizadas e, principalmente, métricas na escala original.

Além das redes, o mesmo script executa ARIMA, ARIMAX, SARIMA e SARIMAX. Os
modelos estatísticos são ajustados uma única vez por dataset, diretamente nos
splits brutos (sem MinMaxScaler ou StandardScaler). Os modelos exógenos usam
as seis variáveis diferentes de `OT` defasadas em uma
hora; assim, a previsão para um instante não consome covariáveis do futuro.
Todos fazem previsão recursiva de um passo à frente e atualizam o estado com o
valor observado anterior. As ordens padrão e o limite de iterações estão em
`src/config.py`; a análise ACF/PACF em `data/statistics/` deve orientar o seu
ajuste final.

A LSTM usa automaticamente a GPU detectada pelo TensorFlow. O script habilita
o crescimento gradual de memória (`memory growth`), para não reservar toda a
VRAM ao iniciar. MLP (scikit-learn) e os modelos ARIMA/SARIMA (statsmodels)
continuam sendo executados em CPU.

O diretório bruto esperado é `data/ETDataset-main/ETT-small/`, onde já estão
`ETTh1.csv` e `ETTh2.csv`.

## Execução

A partir da raiz do projeto refatorado:

```bash
python -m pip install -r requirements.txt
python -m scripts.01_eda
python -m scripts.02_statistical_analysis
python -m scripts.03_data_split
python -m scripts.04_preprocess
python -m scripts.05_windows
python -m scripts.06_baseline
python -m scripts.07_compare_results
```

`01_collect_inspect.py` é uma alternativa textual e rápida a `01_eda.py`; não
é necessário rodar os dois. As etapas 03--07 dependem das anteriores e devem
ser executadas nessa ordem. Para repetir apenas o treinamento depois de mudar
arquitetura ou ordens, basta executar `python -m scripts.06_baseline` e depois
`python -m scripts.07_compare_results` (desde que `data/windows/` já exista).

Após a etapa 06, use `data/baseline/baseline_all_results.csv` como tabela
completa e `data/baseline/plots/` para os gráficos. As métricas com sufixo
`_original` são as apropriadas para a comparação principal.

Após a etapa 07, `data/baseline/baseline_all_predictions.csv` reúne as
previsões de todos os modelos com os valores originais de `OT`. Os gráficos
`data/baseline/plots/predictions/` mostram, para cada modelo e partição,
o valor original versus a previsão ao longo do tempo.
Para TEST, há também arquivos com sufixo `_test_zoom.png`, ampliando os últimos
168 pontos (uma semana); altere `prediction_zoom_points` em `src/config.py`
caso deseje outra janela.

Se preferir executar somente um dataset/scaler, a configuração central em `src/config.py` é o ponto único a ser adaptado.
