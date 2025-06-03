import streamlit as st
import pandas as pd
import joblib
import os

# Estilo de la app
st.markdown("<h1 style='color: #f965e9;'>Predicción de nuevas indicaciones terapéuticas para fármacos ya comercializados.</h1>", unsafe_allow_html=True)
st.caption("Explora posibles nuevos usos terapéuticos para fármacos mediante modelos de machine learning, aplicando reposicionamiento farmacológico, " \
"que consiste en encontrar indicaciones alternativas para medicamentos ya existentes.")

st.markdown("---")

# Explicación inicial mejorada
st.markdown("### 🔍 ¿Cómo funciona esta aplicación?")
st.write("""
Un **algoritmo de agrupamiento** es una técnica que divide los fármacos en grupos (llamados *clusters*) basándose en sus propiedades químicas y características similares.

Esto nos permite encontrar grupos de fármacos que comparten patrones comunes, lo que ayuda a descubrir posibles nuevas indicaciones terapéuticas para esos grupos.

Por eso, primero debes seleccionar el algoritmo con el que quieres agrupar los fármacos, ya que cada uno organiza los datos de forma diferente y genera distintos clusters.

A continuación, elige la indicación terapéutica que quieres predecir para un fármaco específico y comprueba si podría tener **potencial de reposicionamiento**.
""")

# Configuración de modelos
config = {
    'MeanShift': {
        'Antibacterial': 'modelo_rf_meanshift_cluster0_antibacteriano.pkl',
        'Anti-Inflammatory': 'modelo_rf_meanshift_cluster0_antiinflamatorio.pkl',
        'Antineoplastic': 'modelo_rf_meanshift_cluster0_antineoplasico.pkl',
        'Antihypertensive': 'modelo_rf_meanshift_cluster0_antihipertensivo.pkl'
    },
    'KMeans': {
        'Antibacterial': {
            'Cluster 0': 'modelo_rf_kmeans_cluster0_antibacteriano.pkl',
            'Cluster 1': 'modelo_rf_kmeans_cluster1_antibacteriano.pkl'
        },
        'Antipsychotic': {
            'Cluster 2': 'modelo_rf_kmeans_cluster2_antipsicotico.pkl'
        }
    },
    'GMM': {
        'Antibacterial': {
            'Cluster 0': 'modelo_rf_gmm_cluster0_antibacteriano.pkl',
            'Cluster 1': 'modelo_rf_gmm_cluster1_antibacteriano.pkl'
        }
    }
}

# Cargar dataset único
df = pd.read_excel('farmacos_a_clasificar_normalizados.xlsx')

# Si quieres renombrar la columna aquí, descomenta esta línea:
# df.rename(columns={'Indicacion1': 'Indicacion'}, inplace=True)

# Controles principales

# Primer selectbox: algoritmo
algoritmo = st.selectbox("🧬 Selecciona el criterio de agrupamiento:", [""] + list(config.keys()))

if algoritmo:
    # Segundo selectbox o texto fijo según algoritmo
    if algoritmo == 'GMM':
        indicacion = 'Antibacterial'
        st.markdown("**🩺 Indicación disponible:** Antibacterial")
    else:
        indicaciones_disponibles = list(config[algoritmo].keys())
        indicacion = st.selectbox("🩺 ¿Para qué indicación quieres buscar nuevos candidatos?", [""] + indicaciones_disponibles)

    if algoritmo == 'GMM' or indicacion:
        st.markdown("---")

        # Filtrar dataset: eliminar fármacos que ya tengan esa indicación
        df_filtrado = df[df['Indicacion'] != indicacion].copy()
        df_filtrado['Display'] = df_filtrado['ChEMBL ID'] + ' — ' + df_filtrado['Name'] + ' — ' + df_filtrado['Indicacion']

        display_options = [""] + list(df_filtrado['Display'])
        selected_display = st.selectbox("💊 Selecciona un fármaco:", display_options)

        selected_cluster = None

        if selected_display:
            selected_drug = df_filtrado[df_filtrado['Display'] == selected_display]['ChEMBL ID'].values[0]

            st.markdown("---")
            st.markdown("### 📊 Datos básicos del fármaco seleccionado:")
            drug_data = df_filtrado[df_filtrado['ChEMBL ID'] == selected_drug][['Indicacion', 'Name', 'ChEMBL ID']]
            st.dataframe(drug_data)

            # Selección de características para la predicción
            features = ['Molecular Weight', 'Targets', 'AlogP', 'Polar Surface Area', 'HBA', 'HBD',
                        '#RO5 Violations', '#Rotatable Bonds', 'QED Weighted', 'Aromatic Rings']
            X = df_filtrado[df_filtrado['ChEMBL ID'] == selected_drug][features]

            # Si hay clusters
            if isinstance(config[algoritmo][indicacion], dict):
                if algoritmo == 'GMM' and indicacion == 'Antibacterial':
                    st.info("📊 El algoritmo GMM agrupó los fármacos en varios clusters y, en todos ellos, la indicación más frecuente fue **Antibacterial**. "
                            "Por ello, se entrenaron modelos independientes para **Cluster 0** y **Cluster 1**.")
                elif algoritmo == 'KMeans' and indicacion == 'Antibacterial':
                    st.info("📊 En este caso, KMeans formó varios agrupamientos donde la indicación más común en ambos fue **Antibacterial**. "
                            "Por eso, dispones de modelos separados para ambos.")
                elif algoritmo == 'KMeans' and indicacion == 'Antipsychotic':
                    st.info("📊 Solo se detectó un agrupamiento relevante para antipsicóticos mediante KMeans: **Cluster 2**.")

                cluster_options = list(config[algoritmo][indicacion].keys())
                selected_cluster = st.selectbox("🧩 Selecciona el agrupamiento (cluster):", [""] + cluster_options)

            # Botón para lanzar predicción (solo aparece si ya está todo elegido)
            if (not isinstance(config[algoritmo][indicacion], dict)) or (selected_cluster):
                # Mensaje antes del botón
                st.markdown("### ▶️ Pulsa el botón para ver la predicción")
                if st.button("🚀 Realizar predicción"):
                    if isinstance(config[algoritmo][indicacion], dict):
                        model_file = config[algoritmo][indicacion][selected_cluster]
                    else:
                        model_file = config[algoritmo][indicacion]

                    # Aquí cargamos desde la carpeta 'modelos/'
                    model_path = os.path.join("modelos", model_file)
                    clf = joblib.load(model_path)
                    st.write(f"📦 Modelo cargado: {model_file}")

                    prediction = clf.predict(X)[0]

                    st.markdown("---")
                    drug_name = drug_data['Name'].values[0]
                    if prediction == 1:
                        st.success(f"✅ {drug_name} podría tener actividad **{indicacion}**.")
                    else:
                        st.error(f"❌ {drug_name} NO parece **{indicacion}**.")