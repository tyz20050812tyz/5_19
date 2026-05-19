# -*- coding: utf-8 -*-
"""Iris 鸢尾花分类 - Streamlit 版本

运行：
    streamlit run streamlit_app.py
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "iris.csv")
IMG_DIR = os.path.join(HERE, "static", "imgs")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
    y = df["species"].values
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200).fit(X, y),
        "K-Nearest Neighbour": KNeighborsClassifier(n_neighbors=5).fit(X, y),
        "SVM": SVC().fit(X, y),
    }
    return models


def main():
    st.set_page_config(page_title="Iris Predictor", page_icon="🌸", layout="wide")
    st.title("🌸 Iris Species Predictor")
    st.caption("基于 Streamlit 的鸢尾花分类 Web 应用 - 上机内容2")

    df = load_data()
    models = train_models(df)

    with st.sidebar:
        st.header("🔧 输入特征")
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.5, 0.1)
        petal_length = st.slider("Petal Length", 0.0, 7.0, 1.4, 0.1)
        petal_width = st.slider("Petal Width", 0.0, 3.0, 0.2, 0.1)
        model_choice = st.selectbox("选择模型", list(models.keys()))
        do_predict = st.button("🚀 预测", use_container_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 数据预览")
        st.dataframe(df.head(10), use_container_width=True)
        st.write(f"样本数：**{len(df)}**，类别：{', '.join(df['species'].unique())}")

    with col2:
        st.subheader("🔮 预测结果")
        if do_predict:
            sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            model = models[model_choice]
            result = model.predict(sample)[0]

            st.success(f"使用 **{model_choice}** 预测结果：**{result}**")
            st.write("输入特征：", {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            })

            img_map = {
                "setosa": "iris_setosa.jpg",
                "versicolor": "iris_versicolor.jpg",
                "virginica": "iris_virginica.jpg",
            }
            img_path = os.path.join(IMG_DIR, img_map.get(result, ""))
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Iris {result}", width=260)
        else:
            st.info("👈 在左侧调整参数后点击「预测」")


if __name__ == "__main__":
    main()
