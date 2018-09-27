#!/usr/bin/env python
"""
 Created by Dai at 18-9-27.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA ,ARIMA

from base import BaseModel


class ARIMAModel(BaseModel):

    def __init__(self, **kwargs):
        super(ARIMAModel, self).__init__(**kwargs)
        self.first_values = []

    def gen_stationary_data(self, max_diff=1) -> tuple:
        """
        对数据进行做最优差分 必须满足平稳且非白噪声序列
        :param data_series: 历史数据
        :param diff_value: 最大差分数
        :return:
        """
        D_data = self.data
        for i in range(max_diff+1):
            if self.stationarity(D_data)[0] and self.randomness() is not False:
                return D_data, i
            self.first_values.append(pd.Series([D_data[i]]))
            D_data = D_data.diff(1).dropna()
        return D_data, False

    def restore_data(self, D_data):
        """
            根据d阶差分与记录下的值　对差分进行还原
        :param D_data:
        :return: time_series_restored
        """
        time_series_restored = D_data
        for first in reversed(self.first_values):
            time_series_restored = first.append(time_series_restored).cumsum()
        return time_series_restored

    def _proper_model(self, data, max_ar=5, max_ma=5):
        """
            获取模型最佳 p ,q
        :return: （p,q）
        """
        order = st.arma_order_select_ic(data, max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic', 'hqic'])
        return order.bic_min_order

    def fit_model(self, data, order=None,name ="ARMA"):

        if name == "ARMA":
            if order is None:
                order = self._proper_model(data=data)
            model = ARMA(data, order=order)

        if name == "ARIMA":
            if order is None:
                order = self._proper_model(data=data)
                order = order[0],order[1],len(self.first_values)
            model = ARMA(data, order=order)


        result_arma = model.fit(disp=-1, method='css')
        return result_arma

    def error(self, ori_data, pre_data):
        """
            计算拟合误差
            《均方根值（RMS）+ 均方根误差（RMSE）+标准差（Standard Deviation）》
        :return:
        """
        RMSE = np.sqrt(((pre_data - ori_data) ** 2).sum() / pre_data.size)
        return RMSE


if __name__ == "__main__":
    odata = {
        "黑子": [
            330.45, 330.97, 331.64, 332.87, 333.61, 333.55,
            331.90, 330.05, 328.58, 328.31, 329.41, 330.63,
            331.63, 332.46, 333.36, 334.45, 334.82, 334.32,
            333.05, 330.87, 329.24, 328.87, 330.18, 331.50,
            332.81, 333.23, 334.55, 335.82, 336.44, 335.99,
            334.65, 332.41, 331.32, 330.73, 332.05, 333.53,
            334.66, 335.07, 336.33, 337.39, 337.65, 337.57,
            336.25, 334.39, 332.44, 332.25, 333.59, 334.76,
            335.89, 336.44, 337.63, 338.54, 339.06, 338.95,
            337.41, 335.71, 333.68, 333.69, 335.05, 336.53,
            337.81, 338.16, 339.88, 340.57, 341.19, 340.87,
            339.25, 337.19, 335.49, 336.63, 337.74, 338.36
        ]
    }

    model = ARIMAModel(dataset=odata)

    data, d = model.gen_stationary_data()
    arima = model.fit_model(data=data,name="ARMA")

    pdata = model.restore_data(arima.predict())

    pdata.plot()
    model.data.plot()
    plt.show()
