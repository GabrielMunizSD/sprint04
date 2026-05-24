#  Zoo Predictor — Korkeasaari · Sprint 4

**Machine Learning · 1TIAPR-2025**  
Grupo: Rafael Tavares (567357) · Gabriel Muniz (568237) · Yuri Quirino (568512) · Leonardo Barros (566788) · Marcelo Augusto (567176)

---

## Sobre o Produto

Aplicação Streamlit que transforma o modelo de previsão de visitantes do Zoológico de Korkeasaari (Helsinki) em um **produto de dados funcional**, com:

-  **Previsão de visitantes** para qualquer data futura
-  **Classificação de demanda** (Alta / Média / Baixa)  
-  **Explicabilidade SHAP** — por que o modelo previu esse valor?
-  **Recomendações operacionais** para gestores de totens e equipes
-  **Avaliação crítica** do modelo (limitações, riscos, melhorias)

---

## Como Executar

### Pré-requisitos

- Python 3.10+
- pip

### Instalação

```bash
# Clone o repositório
git clone https://github.com/RafaelTavaresMA/Sprint-3-Modelagem-Avan-ada-Tuning-e-Interpretabilidade.git
cd Sprint-3-Modelagem-Avan-ada-Tuning-e-Interpretabilidade

# Instale as dependências
pip install -r requirements.txt

# Execute o app
streamlit run app.py
```

O app abrirá automaticamente em `http://localhost:8501`.

---

## Usando Dados Reais

Se você possui o arquivo `zoo_diario.csv` (gerado no Sprint 3), faça o upload na sidebar. O modelo será re-treinado automaticamente com os dados reais.

**Formato esperado do CSV:**

| data       | visitas |
|------------|---------|
| 2018-01-01 | 320     |
| 2018-01-02 | 450     |
| ...        | ...     |

---

## Estrutura do App

```
app.py                  ← Aplicação principal Streamlit
requirements.txt        ← Dependências Python
modelo_zoo.pkl          ← Modelo treinado (gerado na primeira execução)
README.md               ← Este arquivo
```

---

## Abas da Aplicação

| Aba | Conteúdo |
|-----|----------|
|  Previsão & Explicação | Input de data + lags → Previsão + SHAP waterfall + Recomendações |
|  Explorar Dados | Sazonalidade mensal/semanal, série histórica, importância global SHAP |
|  Sobre o Modelo | Pipeline, features, métricas holdout, estratégia de validação |
|  Uso Prático & Limitações | Benefícios para o totem, decisões suportadas, limitações, melhorias |

---

## Modelo

- **Algoritmo:** Gradient Boosting Regressor (sklearn)
- **Hiperparâmetros:** `n_estimators=200, max_depth=5, learning_rate=0.1`
- **Selecionado após:** GridSearchCV + RandomizedSearchCV (Sprint 3)
- **Validação:** Split cronológico 80/20 + K-Fold 5-fold

### Features (19)

| Categoria | Features |
|-----------|----------|
| Lags temporais | `lag_7d`, `lag_30d`, `media_movel_7d` |
| Sazonalidade circular | `mes_sin`, `mes_cos`, `dow_sin`, `dow_cos` |
| Flags de negócio | `fim_de_semana`, `alta_temporada`, `ferias_escolares`, `feriado_natal`, `feriado_fixo` |
| Componentes da data | `dow`, `mes`, `dia`, `dia_ano`, `semana_ano`, `trimestre`, `ano` |

---

## Explicabilidade SHAP

A aplicação usa **SHAP TreeExplainer** para:

1. **Waterfall local** — mostra como cada feature contribuiu para *aquela previsão específica*
2. **Importância global** — média |SHAP| de cada feature no conjunto de treino
3. **Texto em português** — explicação automática em linguagem natural

Exemplo de saída:  
> "↑ Aumentam a previsão: lag_7d + alta_temporada + fim de semana | ↓ Reduzem a previsão: mes_cos"

---

## Uso Prático (Totens de Autoatendimento)

| Cenário | Ação |
|---------|------|
|  Demanda ALTA (≥ 1.500) | Escala máxima, todas as unidades ativas, pré-venda digital |
|  Demanda MÉDIA (700–1.499) | Escala padrão, monitoramento em tempo real |
|  Demanda BAIXA (< 700) | Manutenção preventiva, campanhas de captação |

---

## Limitações Reconhecidas

-  Sem dados de clima (temperatura, chuva, neve)
-  Sem modelagem de eventos especiais (campanhas, exposições)
-  Anos 2020-2021 excluídos por pandemia
-  Lags precisam de entrada manual (sem API de dados históricos)

---

## Próximas Iterações

- Integrar API meteorológica do FMI (Finnish Meteorological Institute)
- Re-treinamento automático mensal
- Intervalo de confiança quantílico
- Notificações push para gestores
- XGBoost / LightGBM como alternativas ao GB

---

## Fonte dos Dados

Helsinki Region Infoshare — [Helsinki Zoo visitor count](https://hri.fi/data/en_GB/dataset/korkeasaaren-kavijamaarat)  
Licença: **CC-BY 4.0**
