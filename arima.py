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

    def gen_stationary_data(self, max_diff=1, interval=1) -> tuple:
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
            time_series_restored = self.restore_data(D_data=time_series_restored, O_data=res['O_data'],
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
                order = order[0], order[1], len(self.O_datas)
            model = ARIMA(data.dropna(), order=order)

        self.order = order
        print(order)
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
            327247.09, 347478.49, 374911.84, 279318.77, 274205.67, 318741.02, 230855.53,
            260575.12, 322738.7, 285257.64, 335864.08, 426566.69,

            445154.67, 376713.84, 519396.13, 333463.18, 355447.97, 423262.35, 311847.74,
            363631.89, 428433.28, 465468.01, 545247.03, 500616.36,

            815810.83, 715607.76, 747712.06, 497507.13, 507974.84, 600870.61, 483643.6, 572014.1,
            520403.12, 424715.01, 375824.94, 347118.65,

            699998.45, 608926.51, 680231.88, 490350.63, 477458.87, 558411.24, 434173.61,
            496290.86, 638074.49, 536694.26, 541530, 595415.35,

            1029705.92, 1023414.62, 1051051.77, 694068.16, 659232.24, 773098.82, 606358.09,
            666796.64, 774057.73, 709486.25, 631372.56, 581343.73,

            1168697.41, 1021054.65, 1041016.89, 673026.89, 739222.39, 814380.48, 582755.02,
            652611.29, 760037.96, 698872.54, 634313.34, 534320.98,

            1217967.63, 986056.66, 1003707.95, 581083.52, 769340.89, 964701.47, 569202.04,
            596954.49, 804764.51, 683640.3, 636969.78, 531690.9,

            1113023.99, 878617.01, 1071963.89, 625730.85, 845306.68, 1092311.23, 567435.85,
            613000.69, 878620.35, 753745.77, 603894.67, 466470.61,

            1462012.05, 1129919.74, 1243461.6, 536280.29, 700910.07, 947933.11, 551402.73,
            613766.62, 889793.12, 766009.93, 589666.36, 438017.3,

            1620662.94, 938194.46, 1232950.03, 592982.4, 761815.59, 1099848.57, 1105647.29,
            668776.72, 934340.25, 801472.4, 640799.44, 461440.43,

            2586391, 1047581.32, 1601063, 683061.66, 948148.47, 1357147.7, 1280763.16, 1151793.74,
            1077538.26

        ]
    }

    model = ARIMAModel(dataset=odata)
    diff, d = model.gen_stationary_data()
    # diff.plot()
    arma = model.fit_model(data=diff, order=(0, 1), name='ARMA')

    print(arma.summary())
    diff = arma.predict(start=2, end=140)
    pdata = model.gen_restore_data(D_data=diff)
    model.data.plot()
    pdata.plot()

    plt.show()
