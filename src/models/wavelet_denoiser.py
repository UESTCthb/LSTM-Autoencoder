import pywt
import numpy as np


class Wavelet:
    def __init__(self, data, level, wavelet_type="db4"):
        self.data = data
        self.level = level
        self.wavelet_type = wavelet_type

    def wavelet_smooth(self, data):
        coeff = pywt.wavedec(data, self.wavelet_type, self.level)
        for i in range(1, len(coeff)):
            detail_coeff = coeff[i]
            threshold_value = np.std(detail_coeff) * 0.5  # threshold multiply

            for j in range(len(detail_coeff)):
                magnitude = np.abs(detail_coeff[j])

                if magnitude != 0:
                    detail_coeff[j] = (
                        1 - detail_coeff[j] / magnitude
                    ) * threshold_value
                else:
                    detail_coeff[j] = 0

            coeff[i] = detail_coeff
            smoothed_data = pywt.waverec(coeff, self.wavelet_type)
        return smoothed_data[-1]

    def wavelet(self, window_size):

        return self.data.rolling(window_size).agg(self.wavelet_smooth)
