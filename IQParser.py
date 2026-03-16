import os
import re
import time
from datetime import datetime
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

VERSION = '2.0.4'


class IQPostProcessing:
    """
    A collection of static methods for processing IQ data from .mat files.
    """

    @staticmethod
    def hex_to_bit_vector(hex_string):
        """
        Converts a hex string to a bit vector with MSB on the right.

        Parameters:
        - hex_string (str): Hexadecimal string to convert.

        Returns:
        - bit_vector (list[int]): List of integers representing bits.
        """
        bit_vector = []
        n_bytes = len(hex_string) // 2

        for i in range(n_bytes):
            byte_hex = hex_string[2 * i: 2 * i + 2]
            byte_int = int(byte_hex, 16)
            byte_bits = [int(b) for b in f'{byte_int:08b}'][::-1]
            bit_vector.extend(byte_bits)
        return bit_vector

    @staticmethod
    def bits_to_hex_right_msb(bits):
        assert len(bits) % 8 == 0, "Length of bits must be a multiple of 8"
        hex_string = ""
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i + 8]
            byte_bits_reversed = byte_bits[::-1]
            byte_int = int("".join(map(str, byte_bits_reversed)), 2)
            hex_string += f"{byte_int:02X}"
        return hex_string

    @staticmethod
    def burst_detect_by_envelope(signal, fs):
        """
        Detects bursts in a complex-valued signal based on amplitude thresholding.

        Parameters:
        - signal (np.array): Complex signal (IQ samples).
        - fs (float): Sampling frequency in Hz.

        Returns:
        - delta_time_v (np.array): Durations of bursts in seconds.
        - rise_index (np.array): Start indices of bursts.
        - fall_index (np.array): End indices of bursts.
        """
        envelope = np.abs(signal)
        if len(envelope) == 0:
            return np.array([]), np.array([]), np.array([])

        threshold = np.max(envelope) / 10

        binary_input = (envelope >= threshold).astype(int)

        rise_index = np.where((binary_input[0:-1] == 0) & (binary_input[1:] == 1))[0] + 1
        fall_index = np.where((binary_input[0:-1] == 1) & (binary_input[1:] == 0))[0] + 1

        if len(rise_index) > len(fall_index):
            fall_index = np.append(fall_index, len(signal))

        min_length = min(len(rise_index), len(fall_index))
        rise_index = rise_index[:min_length]
        fall_index = fall_index[:min_length]

        delta_time_v = (fall_index - rise_index) / fs

        return delta_time_v, rise_index, fall_index

    @staticmethod
    def estimate_frequency_offset(
        signal,
        rise_index,
        fall_index,
        delta_time_v,
        fs,
        samp_skip=1000,
        symbol_rate=1e-6,
        dF_2MHz=0.42
    ):
        """
        Estimate frequency offset and correct frequency data for bursts in an IQ signal.

        Parameters:
        - signal (np.array): Complex-valued IQ samples.
        - rise_index (np.array): Indices of burst starts.
        - fall_index (np.array): Indices of burst ends.
        - delta_time_v (np.array): Durations of bursts in seconds.
        - fs (float): Sampling frequency in Hz.

        Returns:
        - frequency_offsets (list): List of frequency offsets in kHz.
        - corrected_frequencies (list[np.array]): List of arrays containing corrected frequency data in kHz.
        - delta_time_index (int): Index of the burst used for frequency correction.
        """
        frequency_offsets = []
        corrected_frequencies = []
        delta_time_index = None
        ble_type = None

        true_symbol_rate = 1e-6
        packet_len = 376
        preamble_addres = 'aad6be898e'

        samples_per_us = max(1, round(1e-6 * fs))

        for i in range(len(delta_time_v)):
            if delta_time_v[i] < 100e-6:
                continue

            burst_signal = signal[rise_index[i]:fall_index[i]]
            if len(burst_signal) <= max(samp_skip + 10, samples_per_us * 120):
                continue

            burst_phase = np.unwrap(np.angle(burst_signal)) / (2 * np.pi)

            phase_segment = burst_phase[samp_skip:]
            if len(phase_segment) < 10:
                continue

            time_segment = np.arange(len(phase_segment)) / fs
            coef = np.polyfit(time_segment, phase_segment, 1)

            # Store offset in kHz so downstream f_c = input_center + 1e3 * f_offset still works.
            frequency_offset = coef[0] * 1e-3

            phase_over_usec = burst_phase[samples_per_us:] - burst_phase[:-samples_per_us]

            guard_samps = samples_per_us * 50
            if len(phase_over_usec) <= 2 * guard_samps:
                continue

            # phase_over_usec is cycles per 1usec, numerically MHz
            phase_mid = phase_over_usec[guard_samps:-guard_samps] - frequency_offset * 1e-3

            if np.max(np.abs(phase_mid)) > dF_2MHz:
                true_symbol_rate = 0.5e-6
                packet_len = 480
                preamble_addres = 'aaaad6be898e'
                ble_type = 'BLE5-DBL-EXT'
            elif delta_time_v[i] > 250e-6:
                true_symbol_rate = 1e-6
                packet_len = 376
                preamble_addres = 'aad6be898e'
                ble_type = 'LEGACY'
            else:
                true_symbol_rate = 1e-6
                packet_len = 120
                preamble_addres = 'aad6be898e'
                ble_type = 'BLE5-AUX-PTR'

            samples_per_symbol = max(1, round(true_symbol_rate * fs))
            if len(burst_phase) <= samples_per_symbol:
                continue

            # Convert to kHz so drift/avg_df are consistent.
            burst_freq = (
                (burst_phase[samples_per_symbol:] - burst_phase[0:-samples_per_symbol]) / true_symbol_rate
            ) * 1e-3

            corrected_freq = burst_freq - frequency_offset

            frequency_offsets.append(frequency_offset)
            corrected_frequencies.append(corrected_freq)
            delta_time_index = i

        return frequency_offsets, corrected_frequencies, delta_time_index, true_symbol_rate, packet_len, preamble_addres, ble_type

    @staticmethod
    def detect_preamble(preamble_bits, fs, data_freq, signal, skip_samp=10, symbol_rate=1e-6):
        """
        Detect preamble in frequency-demodulated FSK data using correlation.

        Parameters:
        - preamble_bits (np.array): 1D array of bits representing the preamble.
        - fs (float): Sampling frequency in Hz.
        - data_freq (np.array): Frequency-demodulated data in kHz.
        - signal (np.array): Complex IQ signal for amplitude masking.
        - skip_samp (int): Number of initial samples to skip in the correlation search.

        Returns:
        - corr_output (np.array): Correlation values at each starting point.
        - preamble_index (int): Index of detected preamble (or None if not found).
        """
        sync_preamble_length = len(preamble_bits)
        samples_per_us = int(np.round(symbol_rate * fs))

        max_index = int(len(data_freq) - (1 + samples_per_us * sync_preamble_length))
        if max_index <= 0:
            return np.array([]), None

        corr_output = np.zeros(max_index)

        preamble_symbols = np.array(preamble_bits) - 0.5

        for ind_n in range(skip_samp, max_index):
            curr_input = data_freq[ind_n: ind_n + samples_per_us * sync_preamble_length: samples_per_us]

            if len(curr_input) == sync_preamble_length:
                corr_output[ind_n] = np.abs(np.dot(preamble_symbols, curr_input))

        envelope = np.abs(signal)
        threshold = np.max(envelope) / 10
        use_ind = envelope >= threshold
        corr_output *= use_ind[:len(corr_output)]

        if len(corr_output) == 0:
            return corr_output, None

        preamble_index = np.argmax(corr_output)
        if corr_output[preamble_index] == 0:
            preamble_index = None

        return corr_output, preamble_index

    @staticmethod
    def generate_ble_scrambler(channel=37, seed_length=336):
        """
        Generates a BLE scrambler sequence.
        """
        def de2bi(num, bits):
            return np.array(list(np.binary_repr(num, width=bits)), dtype=int)

        LFSR_FIFO = np.concatenate(([1], de2bi(channel, 6)))
        scrambler_seed = np.zeros(seed_length, dtype=int)

        for j in range(seed_length):
            scrambler_seed[j] = LFSR_FIFO[-1]
            next_stage = np.zeros_like(LFSR_FIFO)

            next_stage[0] = LFSR_FIFO[-1]
            next_stage[1] = LFSR_FIFO[0]
            next_stage[2] = LFSR_FIFO[1]
            next_stage[3] = LFSR_FIFO[2]
            next_stage[4] = LFSR_FIFO[3] ^ LFSR_FIFO[-1]
            next_stage[5] = LFSR_FIFO[4]
            next_stage[6] = LFSR_FIFO[5]

            LFSR_FIFO = next_stage

        return scrambler_seed

    @staticmethod
    def process_interpolated_burst(
        signal,
        preamble_index,
        fs,
        fs_inter,
        frequency_offset,
        num_symbols=376,
        symbol_time=1e-6 ):
        """
        Upsample burst signal, estimate symbol timing adjustment, and demodulate to soft and hard bits.

        Returns:
        - soft_bits: np.array, soft bits in kHz after optimal timing recovery and DC removal.
        - hard_bits: np.array, hard bits (1 or -1) from soft bits.
        - updated_timing: int, optimal symbol period in high-rate samples.
        - est_symbol_clock: float, estimated symbol duration in microseconds.
        """
        up = int(np.round(fs_inter / fs))
        down = 1
        data_inter = resample_poly(np.asarray(signal, dtype=np.complex128), up, down)

        num_sym_hypo = int(np.ceil(10* fs_inter / fs))

        phase_inter = np.unwrap(np.angle(data_inter)) / (2 * np.pi)
        samps_in_symbol_inter = int(np.round(symbol_time * fs_inter))

        if samps_in_symbol_inter <= 0 or len(phase_inter) <= samps_in_symbol_inter:
            return np.array([]), np.array([]), None, None

        # kHz
        freq_inter = ((phase_inter[samps_in_symbol_inter:] - phase_inter[:-samps_in_symbol_inter]) / symbol_time) * 1e-3

        freq_inter -= frequency_offset

        preamble_index_inter = round(preamble_index * fs_inter / fs)

        variances = []
        for sym_change in range(-num_sym_hypo, num_sym_hypo + 1):
            updated_timing = samps_in_symbol_inter + sym_change

            curr_indices = np.arange(
                preamble_index_inter,
                preamble_index_inter + updated_timing * num_symbols,
                updated_timing,
            )

            curr_indices = curr_indices[curr_indices < len(freq_inter)]

            curr_soft_bits = freq_inter[curr_indices.astype(int)]
            variances.append(np.std(curr_soft_bits))

        if len(variances) == 0:
            return np.array([]), np.array([]), None, None

        max_timing_index = np.argmax(variances)
        best_timing_offset = max_timing_index - num_sym_hypo
        updated_timing = samps_in_symbol_inter + best_timing_offset

        if updated_timing <= 0:
            return np.array([]), np.array([]), None, None

        est_symbol_clock = updated_timing / fs_inter * 1e6

        final_indices = np.arange(preamble_index_inter, len(freq_inter), updated_timing)
        final_indices = final_indices[final_indices < len(freq_inter)]
        soft_bits = freq_inter[final_indices.astype(int)]

        soft_bits = soft_bits
        hard_bits = np.where(soft_bits >= np.mean(soft_bits), 1, -1)

        return soft_bits, hard_bits, updated_timing, est_symbol_clock

    @staticmethod
    def calc_drift(soft_bits):
        """
        soft_bits are in kHz.

        Returns:
        - drift_khz_per_symbol: slope of (kHz) vs symbol index  [kHz / symbol]
        - freq_key_khz: intercept at index 0                    [kHz]
        """
        if len(soft_bits) == 0:
            return [0.0, 0.0]
        if len(soft_bits) == 1:
            return [0.0, float(soft_bits[0])]

        idx = np.arange(len(soft_bits))
        [drift_khz_per_symbol, freq_key_khz] = np.polyfit(idx, soft_bits, 1)
        return [drift_khz_per_symbol, freq_key_khz]

    @staticmethod
    def calc_avg_df(soft_bits, symbol_rate=1e-6):
        """
        soft_bits are in kHz.

        Returns:
        - avg_df_kHz: estimated FSK deviation (kHz)
        """
        if len(soft_bits) == 0:
            return None

        mean_val = np.mean(soft_bits)
        ones = soft_bits[soft_bits >= mean_val]
        zeros = soft_bits[soft_bits < mean_val]

        if len(ones) == 0 or len(zeros) == 0:
            return None

        tone_sep_khz = np.mean(ones) - np.mean(zeros)
        avg_df_kHz = tone_sep_khz / 2.0
        return avg_df_kHz

    @staticmethod
    def calc_freq_offset_by_peak(signal, fs, input_center, plot_on=False, ch_freq=None):
        N = len(signal)
        xdelta = 1 / fs
        fft_result = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(N, d=xdelta)

        fft_result_shifted = np.fft.fftshift(fft_result)
        fft_freqs_shifted = np.fft.fftshift(fft_freqs) + input_center

        peak_ind = np.argmax(np.abs(fft_result_shifted))
        peak_freq = fft_freqs_shifted[peak_ind]

        if ch_freq is None:
            freq_offset = input_center - peak_freq
        else:
            freq_offset = ch_freq - peak_freq

        if plot_on:
            plt.figure(figsize=(10, 6))
            plt.plot(peak_freq, 10 * np.log10(np.abs(fft_result_shifted[peak_ind]) + 1e-12), marker='x', markersize=10, color='red')
            plt.plot(fft_freqs_shifted, 10 * np.log10(np.abs(fft_result_shifted) + 1e-12))
            plt.title('IQ Data Plot')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude [dB]')
            plt.show()

        return freq_offset

    @staticmethod
    def return_iq_rec_contant(mat_file_path):
        data = sio.loadmat(mat_file_path)

        iq_data = np.asarray(data['Y']).flatten()
        input_center = float(np.squeeze(data['InputCenter']))
        span = float(np.squeeze(data['Span']))
        xdelta = float(np.squeeze(data['XDelta']))
        fs = 1 / xdelta
        return iq_data, input_center, span, fs

    @staticmethod
    def calc_packet_start_time(mat_file_path: str, start_time):
        """
        Get the creation_time of a record and return the time from start_time.

        If the filename matches the pattern 'iq_<freq>MHz_YYYYMMDD_HHMMSS_microseconds.mat',
        extract the timestamp from the filename. Otherwise, use the file system creation time.
        """
        filename = os.path.basename(mat_file_path)
        pattern = r'iq_\d+MHz_(\d{8})_(\d{6})_(\d+)\.mat'
        match = re.match(pattern, filename)

        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            microseconds_str = match.group(3)

            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            creation_time = dt.timestamp() + int(microseconds_str) / 1_000_000.0
        else:
            creation_time = os.path.getctime(mat_file_path)

        time_from_start = creation_time - start_time
        return float(time_from_start)

    @staticmethod
    def calc_avg_power(signal: np.array):
        if len(signal) != 0:
            p_avg_dbm = 10 * np.log10(np.mean(np.abs(signal)))
        else:
            p_avg_dbm = 999
        return p_avg_dbm

    @staticmethod
    def plot_eye_diagram(freq_data, samples_per_symbol):
        # Slicing per symbol and ploting to visualize eye diagram
        plt.figure(figsize=(10, 6))

        for i in range(len(freq_data) // samples_per_symbol - 2):
            start_idx = i * samples_per_symbol
            end_idx = start_idx + 2 * samples_per_symbol
            time_axis = np.arange(2 * samples_per_symbol) * 1 / (samples_per_symbol * 1e6) * 1e6  # in microseconds
            if end_idx <= len(freq_data):
                symbol = freq_data[start_idx:end_idx]
            plt.plot(time_axis, symbol, alpha=0.5)
       
        plt.title("Eye Diagram of Frequency Data")
        plt.xlabel("Time [usec]")
        plt.ylabel("Frequency [kHz]")
        plt.show()

    @staticmethod
    def plot_soft_bits(soft_bits, f_c, freq_ch, mat_file_path, drift_khz_per_packet, freq_key, save_path, plot_name, packet_len):
        """
        Plots soft bits and drift for an IQ recording.

        NOTE: soft_bits are in kHz.
        drift_khz_per_packet is kHz across the whole packet.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(soft_bits, 'x', markersize=5)
        plt.suptitle(f'IQ Recording Soft Bits, f_offset = {(f_c - freq_ch) * 1e-3} [kHz]')
        plt.ylabel("Soft Symbols [kHz]")
        plt.xlabel("Bit Number")
        plt.title(f'File path: {mat_file_path}', fontdict={'fontsize': 7})

        if len(soft_bits) > 0:
            lin_reg = np.arange(len(soft_bits)) * (drift_khz_per_packet / max(len(soft_bits), 1)) + freq_key
            plt.plot(lin_reg, label=f'Drift: {drift_khz_per_packet:.3f} kHz (N={packet_len})\n')
            plt.legend()

        save_filepath = os.path.join(save_path, f'{plot_name}.jpeg')
        os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
        plt.savefig(save_filepath)
        plt.close()

    @staticmethod
    def build_results_payload(
        time_from_start: float,
        raw_packet: str = None,
        drift=None,
        f_offset: float = None,
        p_avg_dbm: float = None,
        crc: str = None,
        soft_bits: np.array = None,
        avg_df: float = None,
        ble_type: str = None,
        freq_data: np.array = None,
        samples_per_symbol: int = None
    ):
        results = {
            "time_from_start": f'{time_from_start} [sec]',
            'VERSION': VERSION,
            "raw_packet": raw_packet,
            "drift": f'{drift} [kHz]',
            "frequecny_offest": f'{f_offset} [kHz]',
            "avg_power": f'{p_avg_dbm} [dBm]',
            "crc": crc,
            "soft_bits": soft_bits,
            "avg_df": f'{avg_df} [kHz]',
            "ble_type": ble_type,
            "freq_data": f'{freq_data} [kHz]',
            "samples_per_symbol": samples_per_symbol
        }
        return results

    @staticmethod
    def freq_to_ble_channel(freq: int = 2402):
        ch = 37

        if np.abs(freq) > 1e6:
            freq_mhz = freq * 1e-6
        else:
            freq_mhz = float(freq)

        freq_ch = 2402e6

        if np.isclose(freq_mhz, 2402, atol=2):
            ch = 37
            freq_ch = 2402e6
        elif np.isclose(freq_mhz, 2424, atol=2):
            ch = 10
            freq_ch = 2424e6

        return ch, freq_ch

    @staticmethod
    def process_single_iq_file(mat_file_path: str, save_path: str = None,
                               start_time: float = 0, debug_flag: bool = False):
        """
        Returns:
        - results (dict): key=basename, value=payload
          payload fields:
            - drift: kHz per packet
            - frequecny_offest: kHz
            - avg_df: kHz
        """
        plot_name = os.path.basename(mat_file_path)
        time_from_start = IQPostProcessing.calc_packet_start_time(mat_file_path, start_time)

        result = {}

        # Extract IQ data and recording parameters from the .mat file.
        try:
            iq_data, input_center, span, fs = IQPostProcessing.return_iq_rec_contant(mat_file_path=mat_file_path)
        except Exception as e:
            debug_flag and print(f"Cannot load file [{mat_file_path}] due to [{e}]")
            result[plot_name] = IQPostProcessing.build_results_payload(time_from_start=time_from_start)
            return result

        # Padding with zeros if the signal edges
        iq_data = np.concatenate([np.zeros(495, dtype=iq_data.dtype), iq_data])

        # Calculate average power 
        p_avg_dbm = IQPostProcessing.calc_avg_power(iq_data)

        if p_avg_dbm < -80 or p_avg_dbm == 999:
            debug_flag and print("Packet power is below -80dBm")
            return

        # Evelope Detection to find burst segments in the IQ data.
        delta_time, rise_index, fall_index = IQPostProcessing.burst_detect_by_envelope(iq_data, fs)

        # Calculate frequency offset and correct frequency data for the detected burst segments.
        f_offsets, freq_data, delta_time_index, true_symbol_rate, packet_len, preamble_addres, ble_type = IQPostProcessing.estimate_frequency_offset(
            signal=iq_data, rise_index=rise_index, fall_index=fall_index,
            delta_time_v=delta_time, fs=fs, symbol_rate=1e-6
        )

        if not f_offsets or not freq_data or delta_time_index is None:
            debug_flag and print("The packet is too short")
            result[plot_name] = IQPostProcessing.build_results_payload(time_from_start=time_from_start, p_avg_dbm=p_avg_dbm)
            return result

        filtered_iq_data = iq_data[rise_index[delta_time_index]:fall_index[delta_time_index]]

        symbol_rate = true_symbol_rate

        # f_offsets are in kHz, so multiply by 1e3 to get Hz correction.
        f_c = np.round(input_center + 1e3 * f_offsets[0])

        ch, freq_ch = IQPostProcessing.freq_to_ble_channel(f_c)

        preamble_bit_vector = IQPostProcessing.hex_to_bit_vector(preamble_addres)
        sync_preamble_length = len(preamble_bit_vector)
        scram_seed = IQPostProcessing.generate_ble_scrambler(
            channel=ch,
            seed_length=max(packet_len - sync_preamble_length, 0)).astype(int)

        corr_output, preamble_index = IQPostProcessing.detect_preamble(
            preamble_bits=preamble_bit_vector,
            fs=fs,
            data_freq=freq_data[0],
            signal=filtered_iq_data,
            symbol_rate=symbol_rate
        )

        if preamble_index is None:
            debug_flag and print("Preamble was not detected")
            result[plot_name] = IQPostProcessing.build_results_payload(
                time_from_start=time_from_start,
                f_offset=(f_c - freq_ch) * 1e-3,
                p_avg_dbm=p_avg_dbm,
                ble_type=ble_type
            )
            return result

        soft_bits, hard_bits, updated_timing, est_symbol_clock = IQPostProcessing.process_interpolated_burst(
            signal=filtered_iq_data,
            preamble_index=preamble_index,
            frequency_offset=f_offsets[0],
            fs=fs,
            fs_inter=200 * fs,
            num_symbols=packet_len,
            symbol_time=symbol_rate
        )

        if len(soft_bits) == 0 or len(hard_bits) == 0:
            debug_flag and print("Interpolated burst processing failed")
            result[plot_name] = IQPostProcessing.build_results_payload(
                time_from_start=time_from_start,
                f_offset=(f_c - freq_ch) * 1e-3,
                p_avg_dbm=p_avg_dbm,
                ble_type=ble_type
            )
            return result

        drift_khz_per_symbol, freq_key = IQPostProcessing.calc_drift(soft_bits)
        drift_khz_per_packet = drift_khz_per_symbol * packet_len

        avg_df_kHz = IQPostProcessing.calc_avg_df(soft_bits=soft_bits, symbol_rate=symbol_rate)

        # Slicing the frequency data for the eye diagram plot, using the estimated symbol clock to determine the number of samples per symbol.
        samples_per_symbol = int(round(1e-6 * est_symbol_clock * fs))
        freq_data_sliced = freq_data[0][preamble_index: preamble_index + packet_len * samples_per_symbol]
        IQPostProcessing.plot_eye_diagram(freq_data=freq_data_sliced,  samples_per_symbol = samples_per_symbol)
        
        # Converting soft bits to be centered around the real frequecy
        harder_bits = ((hard_bits + 1) * 0.5).astype(int)
        if len(hard_bits) >= packet_len:
            packet = np.bitwise_xor(
                harder_bits[sync_preamble_length:packet_len],
                scram_seed[:packet_len - sync_preamble_length]
            )
            packet_hex = IQPostProcessing.bits_to_hex_right_msb(packet)
        else:
            result[plot_name] = IQPostProcessing.build_results_payload(
                time_from_start=time_from_start,
                drift=drift_khz_per_packet,
                f_offset=(f_c - freq_ch) * 1e-3,
                p_avg_dbm=p_avg_dbm,
                avg_df=avg_df_kHz,
                soft_bits=soft_bits.tolist(),
                ble_type=ble_type,
                freq_data=freq_data_sliced.tolist(),
                samples_per_symbol=samples_per_symbol
            )
            return result

        debug_flag and print(f"{os.path.basename(mat_file_path)}: \n{packet_hex}\n")

        if save_path is not None:
            IQPostProcessing.plot_soft_bits(
                freq_key=freq_key,
                f_c=f_c,
                freq_ch=freq_ch,
                mat_file_path=mat_file_path,
                drift_khz_per_packet=drift_khz_per_packet,
                save_path=save_path,
                plot_name=plot_name,
                soft_bits=soft_bits,
                packet_len=packet_len
            )

        raw_packet = packet_hex
        crc = None
        if len(packet_hex) >= 6:
            raw_packet = packet_hex[:-6] + '000000'
            crc = packet_hex[-6:]

        result[plot_name] = IQPostProcessing.build_results_payload(
            time_from_start=time_from_start,
            raw_packet=raw_packet,
            drift=drift_khz_per_packet,
            f_offset=(f_c - freq_ch) * 1e-3,
            p_avg_dbm=p_avg_dbm,
            avg_df=avg_df_kHz,
            crc=crc,
            soft_bits=soft_bits.tolist(),
            ble_type=ble_type,
            freq_data=freq_data_sliced.tolist(),
            samples_per_symbol=samples_per_symbol
        )

        return result

    @staticmethod
    def process_directory_iq_files(
        directory_path: str,
        start_time: float = 0,
        save_path: str = None,
        debug_flag: bool = False
        ):
        """
        Processes all IQ recording .mat files in a given directory using the IQPostProcessing class.

        Parameters:
        - directory_path (str): Path to the directory containing .mat files.
        - save_path (str, optional): Path to save plots if needed (default: None).

        Returns:
        - results (dict): Dictionary containing the processing results for each file.
        """

        results = {}

        mat_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.mat') or f.endswith('.MAT')])

        for mat_file in mat_files:
            mat_file_path = os.path.join(directory_path, mat_file)
            debug_flag and print(f"Processing file: {mat_file_path}")

            file_results = IQPostProcessing.process_single_iq_file(
                mat_file_path=mat_file_path,
                save_path=save_path,
                start_time=start_time,
                debug_flag=debug_flag
                )

            if file_results:
                results.update(file_results)

        return results


if __name__ == "__main__":
     IQPost = IQPostProcessing()
     results = IQPost.process_directory_iq_files(
       r"mat_files",
       save_path=r"mat_files",
       debug_flag=True
    )