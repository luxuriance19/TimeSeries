#!/usr/bin/env python
"""
 Created by Dai at 18-9-27.
"""

import pandas as pd
from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import numpy as np


class BaseModel():
    def __init__(self, dataset: dict):
        """
            针对单元　分析
        :param dataset:
        """
        self.dataset = pd.DataFrame(data=dataset, dtype=np.float)
        self.key = self.dataset.keys()[0]
        self.data = self.dataset[self.key]

    def stationarity(self, data=None):
        """
            平稳性检测 ADF检验的p<0.01
        :return: (Ture,p)
        """
        if data is None:
            data = self.data

        pval = adfuller(data.dropna(), autolag='AIC')
        return (True, pval[1]) if pval[1] < 0.01 else (False, pval[1])

    def randomness(self, data=None):
        """
            随机性检测
            默认情况下, acorr_ljungbox只计算LB统计量, 只有当参数boxpierce=True时, 才会输出Q统计量.
            一般如果统计量的P值小于0.05时，则可以拒绝原假设，认为该序列为非白噪声序列，跟Q统计量差不多。
        :return:
        """
        if data is None:
            data = self.data
        lbvalue, pval = acorr_ljungbox(data.dropna(), lags=True)
        return (True, pval[0]) if pval[0] > 0.05 else (False, pval[0])


if __name__ == "__main__":
    data = {
        "黑子": [601, 604, 620, 626, 641, 642, 645, 655, 682, 678, 692, 707,
               736, 753, 763, 775, 775, 783, 794, 813, 823, 826, 829, 831,
               830, 838, 854, 872, 882, 903, 919, 937, 927, 962, 975, 995,
               1001, 1013, 1021, 1028, 1027, 1048, 1070, 1095, 1113, 1143, 1154, 1173,
               1178, 1183, 1205, 1208, 1209, 1223, 1238, 1245, 1258, 1278, 1294, 1314,
               1323, 1336, 1355, 1377, 1416, 1430, 1455, 1480, 1514, 1545, 1589, 1634,
               1669, 1715, 1760, 1812, 1809, 1828, 1871, 1892, 1946, 1983, 2013, 2045,
               2048, 2097, 2140, 2171, 2208, 2272, 2311, 2349, 2362, 2442, 2479, 2528,
               2571, 2634, 2684, 2790, 2890, 2964, 3085, 3159, 3237, 3358, 3489, 3588,
               3624, 3719, 3821, 3934, 4028, 4129, 4205, 4349, 4463, 4598, 4725, 4827,
               4939, 5067, 5231, 5408, 5492, 5653, 5828, 5965,
               ]
    }

    model = BaseModel(dataset=data)

    print(model.stationarity())
    print(model.randomness())