#!/usr/bin/env python
"""
 Created by Dai at 18-9-27.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st
from statsmodels.tsa.arima_model import ARMA, ARIMA

from base import BaseModel


class ARIMAModel(BaseModel):

    def __init__(self, **kwargs):
        super(ARIMAModel, self).__init__(**kwargs)
        self.first_values = []

    def _diff_data(self, dataset=None, interval=1):
        if dataset is None:
            dataset = self.data
        diff = list()
        ###差分后的前interval个数应为空
        for i in range(interval):
            diff.append(None)

        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)

        ### 在Model计入每一次差分的情况
        res = {
            'interval': interval,
            "first_values": dataset
        }
        self.first_values.append(res)
        return pd.Series(diff)

    def _restore_data(self, D_data, first_values, interval):
        """
            根据d阶差分与记录下的值　对差分进行还原
        :param D_data:
        :return: time_series_restored
        """

        index = D_data.keys()
        # 获取差分对应索引
        try:
            start = index._start
            stop = index._stop
        except:
            start = index[0]
            stop = index[-1] + 1

        ###判断启始值
        for i in range(start, stop):
            if pd.isna(D_data[i]):
                start += 1
        ### 还原差分
        for i in range(start, stop):
            if i - interval < start:
                D_data[i] = D_data[i] + first_values[i - interval]
            else:
                D_data[i] = D_data[i] + D_data[i - interval]
        return D_data

    def gen_stationary_data(self, max_diff=3) -> tuple:
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
            D_data = self._diff_data(D_data)
        return D_data, False

    def gen_restore_data(self, D_data):
        """
            根据d阶差分与记录下的值　对差分进行还原
        :param D_data:
        :return: time_series_restored
        """
        time_series_restored = D_data
        for res in reversed(self.first_values):
            time_series_restored = self._restore_data(D_data=time_series_restored, first_values=res["first_values"],
                                                      interval=res["interval"])
        return time_series_restored

    def _proper_model(self, data, max_ar=5, max_ma=5):
        """
            获取模型最佳 p ,q
        :return: （p,q）
        """
        order = st.arma_order_select_ic(data.dropna(), max_ar=max_ar, max_ma=max_ma, ic=['aic', 'bic', 'hqic'])
        return order.bic_min_order

    def fit_model(self, data, order=None, name="ARMA"):

        if name == "ARMA":
            if order is None:
                order = self._proper_model(data=data)
            model = ARMA(data.dropna(), order=order)

        if name == "ARIMA":
            if order is None:
                order = self._proper_model(data=data.dropna())
                order = order[0], order[1], len(self.first_values)
            model = ARIMA(data.dropna(), order=order)

        self.order = order
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
        "黑子": [601, 604, 620, 626, 641, 642, 645, 655, 682, 678, 692, 707,
               736, 753, 763, 775, 775, 783, 794, 813, 823, 826, 829, 831,
               830, 838, 854, 872, 882, 903, 919, 937, 927, 962, 975, 995
               ]
    }

    model = ARIMAModel(dataset=odata)

    diff, d = model.gen_stationary_data()
    print(d)
    arma = model.fit_model(data=diff,order=(0,1))
    print(arma.summary())
    #
    diff = arma.predict()

    pdata = model.gen_restore_data(D_data=diff)
    model.data.plot()
    pdata.plot()

    plt.show()
