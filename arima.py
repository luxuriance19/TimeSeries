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
        self.O_datas = []

    def gen_stationary_data(self, max_diff=3, interval=1) -> tuple:
        """
        对数据进行做最优差分 必须满足平稳且非白噪声序列
        :param data_series: 历史数据
        :param diff_value: 最大差分数
        :param interval : 差分步长
        :return:
        """
        D_data = self.data
        for i in range(max_diff + 1):
            if self.stationarity(data=D_data)[0] and self.randomness(data=D_data) is not False:
                return D_data, i
            # 用于还原
            res = {
                'interval': interval,
                "O_data": D_data
            }
            D_data = self.diff_data(D_data)
            self.O_datas.append(res)
        return D_data, False

    def gen_restore_data(self, D_data):
        """
            对D阶差分进行数据还原
        :param D_data:
        :return: time_series_restored
        """
        time_series_restored = D_data
        for res in reversed(self.O_datas):
            time_series_restored = self.restore_data(D_data=time_series_restored, O_Data=res['O_data'],
                                                     interval=res['interval'])
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
               830, 838, 854, 872, 882, 903, 919, 937, 927, 962, 975, 995,
               1001, 1013, 1021, 1028, 1027, 1048, 1070, 1095, 1113, 1143, 1154, 1173,
               1178, 1183, 1205, 1208, 1209, 1223, 1238, 1245, 1258, 1278, 1294, 1314
               ]
    }


    model = ARIMAModel(dataset=odata)

    diff, d = model.gen_stationary_data()
    print(d)
    arma = model.fit_model(data=diff, order=(1,1))
    print(arma.summary())
    #
    diff = arma.predict()

    pdata = model.gen_restore_data(D_data=diff)
    model.data.plot()
    pdata.plot()

    plt.show()
