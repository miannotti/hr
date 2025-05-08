import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Cargar modelo y encoder
modelo = joblib.load("modelo_rf.pkl")
encoder = joblib.load("encoder.pkl")

st.title("📊 Ranking de Candidatos por Vacante")

# Cargar archivo de candidatos
archivo = st.file_uploader("Sube el archivo CSV de candidatos", type=["csv"])

if archivo:
    df = pd.read_csv(archivo)

    # Verificamos columnas mínimas
    columnas_esperadas = [
        "vacante_id", "educacion", "experiencia_anios", "experiencia_sector",
        "certificaciones", "puntaje_test", "puntaje_entrevista", "nivel_ingles",
        "referencia_interna"
    ]
    if not all(col in df.columns for col in columnas_esperadas):
        st.error("El archivo no contiene todas las columnas necesarias.")
    else:
        # Codificar variables
        cat_cols = ["educacion", "experiencia_sector", "nivel_ingles"]
        num_cols = ["experiencia_anios", "certificaciones", "puntaje_test", "puntaje_entrevista", "referencia_interna"]

        X_cat = encoder.transform(df[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(cat_cols))
        X = pd.concat([X_cat_df, df[num_cols].reset_index(drop=True)], axis=1)

        # Predecir probabilidades
        df["probabilidad_contratacion"] = modelo.predict_proba(X)[:, 1]

        # Seleccionar vacante
        vacantes = sorted(df["vacante_id"].unique())
        seleccion = st.selectbox("Selecciona una vacante", vacantes)

        df_vacante = df[df["vacante_id"] == seleccion].copy()
        df_vacante = df_vacante.sort_values("probabilidad_contratacion", ascending=False)
        df_vacante["candidato"] = [f"Candidato {i+1}" for i in range(len(df_vacante))]

        # Gráfico interactivo
        fig = px.bar(
            df_vacante,
            x="probabilidad_contratacion",
            y="candidato",
            orientation="h",
            title=f"Ranking de candidatos - {seleccion}",
            labels={"probabilidad_contratacion": "Probabilidad de contratación", "candidato": "Candidato"},
            hover_data=["educacion", "experiencia_anios", "puntaje_test", "puntaje_entrevista"]
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig)

        # Mostrar tabla
        st.subheader("🔎 Detalles de los candidatos")
        st.dataframe(df_vacante.reset_index(drop=True))
