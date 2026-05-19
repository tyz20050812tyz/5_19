# 机器学习上机 2 — 模型训练与 Web 双平台部署

> 课程：机器学习  
> 日期：2026-05-19  
> 工作目录：`d:\徐安然\大三下\机器学习\5_19`

---

## 📌 一、本次上机做了什么

本次上机的核心是 **「机器学习模型的 Web 部署」**，围绕一份"预测第三个月销售额"的线性回归模型展开，并额外练习用 Streamlit 部署 Iris 鸢尾花分类器。完整覆盖以下 5 个环节：

| # | 任务 | 产物 | 状态 |
|---|------|------|------|
| 1 | 用 `LinearRegression` 训练销售预测模型 | `flaskapp/model.pkl` | ✅ |
| 2 | 用 `pickle` 持久化保存 / 加载模型 | `flaskapp/model_train.py` | ✅ |
| 3 | 修改 `app.py` 加载模型，构建 Flask Web 服务 | `flaskapp/app.py` | ✅ |
| 4 | 本地部署测试（含表单页 + JSON API） | http://127.0.0.1:5000 | ✅ |
| 5 | Streamlit 部署 Iris Predictor | `Iris_Predictor/streamlit_app.py` → http://127.0.0.1:8501 | ✅ |
| 6 |PythonAnywhere 云端部署 | 见文末说明 | 📖 教程 |

---

## 📁 二、目录结构总览

```
5_19/
├── Exercise2-20260519.ipynb         # 上机题目 & 两处 pickle 填空
├── Iris_Predictor.zip               # 老师提供的原始资源（保留）
│
├── flaskapp/                        # 任务 1~4：销售预测 Flask 应用
│   ├── app.py                       # ★ Flask 主程序（已完成 pickle 加载）
│   ├── model_train.py               # ★ 训练并保存 model.pkl
│   ├── model.pkl                    # ★ 已训练好的线性回归模型
│   ├── sales.csv                    # 训练数据
│   ├── request.py                   # /results API 测试脚本
│   ├── templates/index.html         # 前端页面
│   └── static/css/style.css         # 前端样式
│
└── Iris_Predictor/                  # 任务 5：Streamlit 鸢尾花分类
    ├── streamlit_app.py             # ★ Streamlit 主程序（新写）
    ├── app.py                       # 老师提供的旧版（依赖已废弃 sklearn.externals）
    ├── data/iris.csv                # 鸢尾花数据集
    ├── static/imgs/                 # 三种鸢尾花图片
    └── templates/                   # Flask 版模板（仅供参考）
```

---

## 🧠 三、核心技术点

### 3.1 数据预处理（`sales.csv`）
```python
dataset['rate'].fillna(0)                                       # 缺失等级填 0
dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean())  # 均值填充
X['rate'] = X['rate'].apply(convert_to_int)                     # 'four' → 4 等英文转整型
```

### 3.2 训练 + pickle 保存（对应 ipynb「代码 1」）
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X, y)
pickle.dump(regressor, open('model.pkl', 'wb'))   # ★ 代码 1
```

### 3.3 加载模型测试（对应 ipynb「代码 2」）
```python
model = pickle.load(open('model.pkl', 'rb'))      # ★ 代码 2
print(model.predict([[4, 300, 500]]))
```

### 3.4 Flask 关键路由（[app.py](file:///d:/徐安然/大三下/机器学习/5_19/flaskapp/app.py)）
| 路由 | 方法 | 用途 |
|------|------|------|
| `/` | GET | 渲染 index.html 表单 |
| `/predict` | POST | 接收表单 → 模型预测 → 返回页面 |
| `/results` | POST | 接收 JSON → 返回预测值（`request.py` 调用） |

#### 🧾 页面三个输入栏说明

打开 http://127.0.0.1:5000 看到的三个输入框，对应模型的 **3 个特征**，预测目标是 **第三个月的销量**。

| # | 输入框（页面） | 字段名（CSV） | 含义 | 示例 |
|---|---|---|---|---|
| 1️⃣ | **rate** | `rate` | 销售员/产品的评级（原数据用英文 `one~twelve`，已转为 0~12 整数） | `4`、`5` |
| 2️⃣ | **sales in first month** | `sales_in_first_month` | 第一个月销量 | `200`、`300` |
| 3️⃣ | **sales in second month** | `sales_in_second_month` | 第二个月销量 | `400`、`500` |

🔮 点击 **Predict sales in third month** 后，`LinearRegression` 输出预测的 `sales_in_third_month`。

**数据来源对照**（[sales.csv](file:///d:/徐安然/大三下/机器学习/5_19/flaskapp/sales.csv)）：

| rate | sales_in_first_month | sales_in_second_month | sales_in_third_month（标签） |
|------|----------------------|------------------------|------------------------------|
| (空→0) | 2 | 500 | 300 |
| (空→0) | 4 | 300 | 650 |
| four=4 | 600 | 200 | 400 |
| nine=9 | 450 | 320 | 650 |
| seven=7 | 600 | 250 | 350 |
| five=5 | 550 | 200 | 700 |

> 训练时前 3 列作为特征 X，第 4 列作为标签 y；页面上让你输入这三列正是为了预测第 4 列。

**试一组示例**：输入 `rate=5` / `sales_in_first_month=200` / `sales_in_second_month=400`，页面返回：`Sales should be $ 523.85` ✅

> 💡 原数据 `rate` 列用的是英文单词（`four/five/...`），[model_train.py](file:///d:/徐安然/大三下/机器学习/5_19/flaskapp/model_train.py) 里用 `convert_to_int` 函数转成了数字。在网页输入时**直接填阿拉伯数字**即可，不要填 `four`。

### 3.5 Streamlit Iris Predictor（[streamlit_app.py](file:///d:/徐安然/大三下/机器学习/5_19/Iris_Predictor/streamlit_app.py)）
- 加载 `iris.csv` 实时训练 3 个模型：Logistic Regression / KNN / SVM
- 侧边栏滑块输入 4 个特征
- 点击「预测」即时显示分类结果 + 鸢尾花图片

---

## 🚀 四、本地启动指南

### 4.1 环境依赖

> Python 版本：3.9（已验证）

```powershell
# 如代理异常导致 pip 装不上，使用清华源
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple `
    flask scikit-learn pandas numpy requests streamlit
```

> ⚠️ Windows 若开启了 IE 代理（127.0.0.1:10808 等），`pip` 会被 WinINET 劫持。  
> 解决方法：临时设置 `$env:NO_PROXY="*"` 后再装。

### 4.2 启动 ① — Flask 销售预测（端口 5000）

```powershell
cd "d:\徐安然\大三下\机器学习\5_19\flaskapp"

# 第一次需先生成 model.pkl（已在）
python model_train.py

# 启动 Flask
python app.py
```

浏览器访问 → **http://127.0.0.1:5000**  
在三个输入框填 `rate / sales_in_first_month / sales_in_second_month`，点击 *Predict sales in third month*。

测试 JSON API：
```powershell
python request.py
# 期望输出：523.8530977224776
```

### 4.3 启动 ② — Streamlit 鸢尾花分类（端口 8501）

```powershell
cd "d:\徐安然\大三下\机器学习\5_19\Iris_Predictor"
streamlit run streamlit_app.py
```

浏览器访问 → **http://127.0.0.1:8501**  
拖动侧边栏滑块 → 选择模型 → 点 🚀 预测，右侧出现物种名 + 对应图片。

### 4.4 一键启动（PowerShell 小脚本，可选）

```powershell
# 起 Flask（后台）
Start-Process powershell -ArgumentList "-NoExit","-Command","cd 'd:\徐安然\大三下\机器学习\5_19\flaskapp'; python app.py"

# 起 Streamlit（后台）
Start-Process powershell -ArgumentList "-NoExit","-Command","cd 'd:\徐安然\大三下\机器学习\5_19\Iris_Predictor'; streamlit run streamlit_app.py"
```

---

## ☁️ 五、PythonAnywhere 云端部署（选做）

参考课件 PPT《4-机器学习Web应用》：

1. 注册并登录 https://www.pythonanywhere.com  
2. **Files** 标签 → 把整个 `flaskapp/` 目录上传（含 `model.pkl`、`templates/`、`static/`）  
3. **Web** 标签 → *Add a new web app* → 选 **Flask** → Python **3.9**  
4. 修改 WSGI 配置文件，指向你的 `app.py`：
   ```python
   import sys
   path = '/home/<你的用户名>/flaskapp'
   if path not in sys.path:
       sys.path.append(path)
   from app import app as application
   ```
5. **Consoles** → Bash 中执行：
   ```bash
   pip install --user flask scikit-learn pandas numpy
   ```
6. 回到 **Web** 页 → 点 **Reload**  
7. 访问 `https://<你的用户名>.pythonanywhere.com` 即可远程使用 🎉

---

## 🐛 六、踩坑记录

| 问题 | 现象 | 解决 |
|------|------|------|
| pip 安装失败 | `Cannot connect to proxy` | Windows IE 代理被劫持，设 `$env:NO_PROXY="*"` 或关闭系统代理 |
| Iris_Predictor/app.py 报错 | `from sklearn.externals import joblib` 在新版 sklearn 已被移除 | 改用 `import joblib` 或直接用本仓库 `streamlit_app.py` |
| `LinearRegression` 警告 feature names | `X does not have valid feature names` | 仅警告，不影响结果。可统一用 `np.array` 输入 |
| Flask debug 模式重启两次 | `* Restarting with watchdog` 属正常 | 无需处理 |

---

## 📄 七、ipynb 两处填空答案速查

```python
# 代码 1：保存模型
pickle.dump(regressor, open('model.pkl', 'wb'))

# 代码 2：加载模型
model = pickle.load(open('model.pkl', 'rb'))
```

---

## ✅ 八、验收清单

- [x] `model.pkl` 已生成
- [x] `python app.py` 本地能访问 `127.0.0.1:5000` 表单
- [x] `python request.py` 返回数值（523.85）
- [x] `streamlit run streamlit_app.py` 能预测三种鸢尾花
- [ ] PythonAnywhere 云端访问正常（按上面第 5 节自行完成）

恭喜你已经完成本次上机全部内容！🎉
