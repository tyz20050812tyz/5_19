# -*- coding: utf-8 -*-
"""训练销售预测线性回归模型并通过 pickle 持久化保存。

运行：
    python model_train.py
将在当前目录生成 model.pkl，供 app.py 加载使用。
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "sales.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")


def convert_to_int(word):
    """将英文数字字符串转为整数；非字符串/未知值返回 0。"""
    word_dict = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "zero": 0, 0: 0,
    }
    return word_dict.get(word, 0)


def main():
    dataset = pd.read_csv(CSV_PATH)

    # 缺失值处理：rate 缺失填 0；首月销量缺失填均值
    dataset["rate"] = dataset["rate"].fillna(0)
    dataset["sales_in_first_month"] = dataset["sales_in_first_month"].fillna(
        dataset["sales_in_first_month"].mean()
    )

    X = dataset.iloc[:, :3].copy()
    X["rate"] = X["rate"].apply(convert_to_int)
    y = dataset.iloc[:, -1]

    regressor = LinearRegression()
    regressor.fit(X, y)

    # 代码 1：通过 pickle 将模型持久化保存到硬盘
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(regressor, f)
    print(f"[OK] 模型已保存到 {MODEL_PATH}")

    # 代码 2：通过 pickle 从硬盘加载模型并测试
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    sample = np.array([[4, 300, 500]])
    print("测试输入 [4, 300, 500] -> 预测第三月销量:", model.predict(sample)[0])


if __name__ == "__main__":
    main()
