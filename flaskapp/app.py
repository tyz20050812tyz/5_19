import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# 通过pickle的方式将硬盘中的模型加载进来进行测试
# 代码 2：加载已经训练并持久化保存的线性回归模型
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# 训练时使用的特征列名，预测时同样用 DataFrame 传入以消除 sklearn 的 feature names 警告
FEATURE_NAMES = ['rate', 'sales_in_first_month', 'sales_in_second_month']


def _predict(values):
    """用 DataFrame 包装输入，避免 sklearn UserWarning。"""
    df = pd.DataFrame([values], columns=FEATURE_NAMES)
    return model.predict(df)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 表单可能输入小数（如 rate=0.2），用 float 避免 ValueError
    features = [float(x) for x in request.form.values()]
    prediction = _predict(features)

    output = round(float(prediction[0]), 2)

    return render_template(
        'index.html',
        prediction_text='Sales should be $ {}'.format(output),
    )


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = _predict(list(data.values()))

    output = float(prediction[0])
    return jsonify(output)


if __name__ == "__main__":
    # debug=True 会让 watchdog 监听整个 Python 路径（含 site-packages 和系统库），
    # 频繁重启影响开发体验。此处用 use_reloader=False 关闭自动重载，
    # 仍保留 debugger 错误页（debug=True）。如不需要调试器可同时设 debug=False。
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
