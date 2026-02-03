import unittest
import numpy as np
import pyfar as pf

from application import (
    generate_ess,
    generate_mls,
    signal_to_audio_array,
    audio_array_to_signal,
    convert_a_to_b_format
)

class TestIRMeasurementLogic(unittest.TestCase):

    def setUp(self):
        self.fs = 48000
        self.duration = 1.0
        self.order = 10

    def test_ess_generation(self):
        expected_n_samples = int(self.fs * self.duration)
        ess_signal = generate_ess(f1=20, f2=20000, duration=self.duration, fs=self.fs)

        self.assertIsInstance(ess_signal, pf.Signal)
        self.assertEqual(ess_signal.n_samples, expected_n_samples)
        self.assertEqual(ess_signal.sampling_rate, self.fs)

    def test_mls_generation(self):
        expected_n_samples = 2 ** self.order - 1
        mls_signal = generate_mls(order=self.order, fs=self.fs)

        self.assertEqual(mls_signal.n_samples, expected_n_samples)
        unique_values = np.unique(mls_signal.time)
        np.testing.assert_array_almost_equal(np.sort(unique_values), [-1.0, 1.0])

    def test_data_conversions(self):
        data = np.random.rand(2, 100).astype(np.float32)
        signal = pf.Signal(data, self.fs)

        audio_arr = signal_to_audio_array(signal)
        self.assertEqual(audio_arr.shape, (100, 2))

        back_to_signal = audio_array_to_signal(audio_arr, self.fs)
        self.assertEqual(back_to_signal.time.shape, (2, 100))
        np.testing.assert_array_almost_equal(signal.time, back_to_signal.time)

    def test_convert_a_to_b_format(self):
        ones_data = np.ones((4, 10))
        a_format_signal = pf.Signal(ones_data, self.fs)

        b_format = convert_a_to_b_format(a_format_signal)

        np.testing.assert_array_almost_equal(b_format.time[0], 4.0)
        np.testing.assert_array_almost_equal(b_format.time[1:], 0.0)

if __name__ == '__main__':
    unittest.main()