import torch
from statsmodels.tsa.stattools import adfuller
import pywt
import numpy as np
import pandas as pd
from sklearn import svm
import statsmodels.api as sm
import copy


class Wavelet:
    def __init__(self, data, level):
        self.data = data
        self.level = level

    def wavelet_smooth(self, wavelet="db4"):
        coeff = pywt.wavedec(self.data, wavelet, self.level)
        for i in range(1, len(coeff)):
            detail_coeff = coeff[i]
            threshold_value = np.std(detail_coeff) * 0.5  # threshold multiply

            for j in range(len(detail_coeff)):
                magnitude = np.abs(detail_coeff[j])

                # Check if magnitude is zero before division
                if magnitude != 0:
                    detail_coeff[j] = (
                        1 - detail_coeff[j] / magnitude
                    ) * threshold_value
                else:
                    detail_coeff[j] = 0

            coeff[i] = detail_coeff
        smoothed_data = pywt.waverec(coeff, wavelet)
        return smoothed_data[-1]

    def wavelet(self, windowsize):
        return self.data.rolling(windowsize).agg(self.wavelet_smooth())
