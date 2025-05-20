import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# 读取数据
data = pd.read_csv("C:/Desktop/wendu/静止.txt", delimiter=",", encoding='GBK')
data.columns = ["%time", "field.ad1", "field.ad2", "field.ad3", "field.ad4", "field.ad5", "field.ad6", "field.ad7"]

# 更改列类型
data = data.astype({"%time": "int64", "field.ad1": "int64", "field.ad2": "int64", "field.ad3": "int64"})

# 计算时间秒数
data["秒"] = data["%time"] / 1e9 - 1716468238.0561182

# 删除多余的列
data = data.drop(columns=["%time", "field.ad4", "field.ad5", "field.ad6", "field.ad7"])

# 修正错误数据
data["field.ad1"] = 4096 - data["field.ad1"]
data["field.ad2"] = 4096 - data["field.ad2"]
data["field.ad3"] = 4096 - data["field.ad3"]

# 计算温度和湿度
data["field.ad1"] = -66.875 + 218.75 * (data["field.ad1"] / 4096)
data["field.ad2"] = -12.5 + 125 * (data["field.ad2"] / 4096)
data["field.ad3"] = data["field.ad3"] / 4096 * 5

# 重新排序列
data = data[["秒", "field.ad1", "field.ad2", "field.ad3"]]

# 应用卡尔曼滤波
kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)

# 对每一列应用卡尔曼滤波
for column in ["field.ad1", "field.ad2", "field.ad3"]:
    data[column], _ = kf.smooth(data[column].values)

# 保存处理后的数据
data.to_csv("C:/Desktop/wendu/静止_卡尔曼滤波.csv", index=False, encoding='GBK')
