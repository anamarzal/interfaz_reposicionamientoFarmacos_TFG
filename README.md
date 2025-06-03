# interfaz_reposicionamientoFarmacos_TFG

# 💊 TFG - Reposicionamiento de Fármacos mediante Machine Learning

Este proyecto corresponde a mi Trabajo de Fin de Grado (TFG) en Ingeniería Biomédica, donde se ha desarrollado una herramienta interactiva para el **reposicionamiento de fármacos** empleando algoritmos de **clustering** y **clasificación**.

La aplicación está implementada en **Python** utilizando **Streamlit**, permitiendo al usuario explorar predicciones sobre posibles nuevas indicaciones terapéuticas para diferentes compuestos.

---

## 📊 Descripción de la App

La interfaz permite al usuario:

- Seleccionar un fármaco de la base de datos.
- Elegir una indicación terapéutica de interés.
- Ejecutar la predicción mediante un modelo de **Machine Learning** entrenado previamente.
- Visualizar si el fármaco podría ser candidato para esa nueva indicación, en función de sus características.

El modelo se ha desarrollado utilizando técnicas de **clustering** para segmentar los datos y posteriormente entrenar clasificadores específicos por clúster, optimizando así la precisión del reposicionamiento.

---

## 🧪 Tecnologías empleadas

- **Python 3.x**
- **Streamlit**
- **scikit-learn == 1.3.2**
- **pandas**
- **numpy**

---
