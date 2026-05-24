"""
Script de pré-treinamento do modelo.
Execute UMA VEZ antes do app para gerar modelo_zoo.pkl:
    python train_model.py
    
Ou, se tiver o zoo_diario.csv:
    python train_model.py --csv zoo_diario.csv
"""

import argparse, pickle, warnings
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

FEATURES = [
    'ano', 'mes', 'dia', 'dow', 'dia_ano', 'semana_ano', 'trimestre',
    'fim_de_semana', 'alta_temporada', 'ferias_escolares',
    'feriado_natal', 'feriado_fixo',
    'mes_sin', 'mes_cos', 'dow_sin', 'dow_cos',
    'lag_7d', 'lag_30d', 'media_movel_7d',
]

FERIADOS_FIXOS = {(1,1), (1,6), (5,1), (12,6), (12,24), (12,25), (12,26)}


def eh_feriado_fixo(d):
    return int((d.month, d.day) in FERIADOS_FIXOS)


def gerar_dados_sinteticos():
    anos = [2018, 2019, 2022, 2023]
    rng  = np.random.default_rng(SEED)
    sazon = {1:.12, 2:.13, 3:.18, 4:.30, 5:.50, 6:.90, 7:1.00,
             8:.85, 9:.45, 10:.25, 11:.14, 12:.20}
    dow_ef = {0:.60, 1:.55, 2:.55, 3:.60, 4:.75, 5:1.00, 6:.90}
    registros = []
    for ano in anos:
        d = date(ano, 1, 1)
        while d.year == ano:
            feriado = 1.5 if (d.month, d.day) in FERIADOS_FIXOS else 1.0
            natal   = 1.8 if (d.month == 12 and d.day >= 24) else 1.0
            v = 3200 * sazon[d.month] * dow_ef[d.weekday()] * feriado * natal
            v = max(50, v + rng.normal(0, v * 0.12))
            registros.append({"data": pd.Timestamp(d), "visitas": round(v)})
            d += timedelta(days=1)
    return pd.DataFrame(registros)


def gerar_features_serie(df):
    df = df.sort_values("data").reset_index(drop=True)
    df["ano"]         = df["data"].dt.year
    df["mes"]         = df["data"].dt.month
    df["dia"]         = df["data"].dt.day
    df["dow"]         = df["data"].dt.dayofweek
    df["dia_ano"]     = df["data"].dt.dayofyear
    df["semana_ano"]  = df["data"].dt.isocalendar().week.astype(int)
    df["trimestre"]   = df["data"].dt.quarter
    df["fim_de_semana"]    = (df["dow"] >= 5).astype(int)
    df["alta_temporada"]   = df["mes"].isin([6,7,8]).astype(int)
    df["ferias_escolares"] = df["mes"].isin([6,7]).astype(int)
    df["feriado_natal"]    = ((df["mes"]==12) & (df["dia"]>=24)).astype(int)
    df["feriado_fixo"]     = df["data"].apply(lambda d: eh_feriado_fixo(d.date()))
    df["mes_sin"] = np.sin(2*np.pi*df["mes"]/12)
    df["mes_cos"] = np.cos(2*np.pi*df["mes"]/12)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)
    df["lag_7d"]         = df["visitas"].shift(7)
    df["lag_30d"]        = df["visitas"].shift(30)
    df["media_movel_7d"] = df["visitas"].shift(1).rolling(7).mean()
    return df.dropna().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Caminho para zoo_diario.csv")
    args = parser.parse_args()

    if args.csv:
        df_raw = pd.read_csv(args.csv, parse_dates=["data"])
        print(f"✅ CSV carregado: {len(df_raw)} linhas")
    else:
        df_raw = gerar_dados_sinteticos()
        print(f"ℹ️  Usando dados sintéticos: {len(df_raw)} linhas")

    df_fe = gerar_features_serie(df_raw.copy())
    X = df_fe[FEATURES]
    y = df_fe["visitas"]

    split_idx = int(len(df_fe) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print("⚙️  Treinando Gradient Boosting...")
    modelo = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=SEED
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metricas = {
        "MAE":  round(mean_absolute_error(y_test, y_pred), 1),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 1),
        "R2":   round(r2_score(y_test, y_pred), 4),
    }
    print(f"📊 Métricas holdout → MAE: {metricas['MAE']} | RMSE: {metricas['RMSE']} | R²: {metricas['R2']}")

    print("🔍 Calculando SHAP explainer...")
    explainer = shap.TreeExplainer(modelo)

    bundle = {
        "modelo":    modelo,
        "explainer": explainer,
        "metricas":  metricas,
        "X_train":   X_train,
        "y_train":   y_train,
        "df_fe":     df_fe,
    }
    with open("modelo_zoo.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("✅ Modelo salvo em modelo_zoo.pkl")
    print("🚀 Execute: streamlit run app.py")


if __name__ == "__main__":
    main()
