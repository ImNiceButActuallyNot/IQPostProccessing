import os
import re
import time
from datetime import datetime
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

VERSION = '2.0.0'

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
        n_bytes = len(hex_string)//2
        
        for i in range(n_bytes):
            byte_hex = hex_string[2 * i: 2 * i + 2]
            byte_int = int(byte_hex, 16)
            # Convert to 8-bit binary string, then map to integers (right-msb, big-endian)
            byte_bits = [int(b) for b in f'{byte_int:08b}'][::-1]  # Reverse the bits
            bit_vector.extend(byte_bits)
        return bit_vector

    @staticmethod
    def bits_to_hex_right_msb(bits):
        assert len(bits) % 8 == 0, "Length of bits must be a multiple of 8"
        hex_string = ""
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            byte_bits_reversed = byte_bits[::-1]  # Flip LSB to MSB within byte!
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
        threshold = np.max(envelope) / 10

        # Binary detection of bursts
        binary_input = (envelope >= threshold).astype(int)

        # Find rising edges (0 → 1) and falling edges (1 → 0)
        rise_index = np.where((binary_input[0:-1] == 0) & (binary_input[1:] == 1))[0] 
        fall_index = np.where((binary_input[0:-1] == 1) & (binary_input[1:] == 0))[0] 

        # Handle case where the burst stays "on" until the end of the signal
        if len(rise_index) > len(fall_index):
            fall_index = np.append(fall_index, len(signal))

        # Trim to same length (in case of any mismatch)
        min_length = min(len(rise_index), len(fall_index))
        rise_index = rise_index[:min_length]
        fall_index = fall_index[:min_length]

        # Calculate burst durations in seconds
        delta_time_v = (fall_index - rise_index) / fs

        return delta_time_v, rise_index, fall_index

    @staticmethod
    def estimate_frequency_offset(signal, rise_index, fall_index, delta_time_v, fs , samp_skip = 999,symbol_rate = 1e-6, dF_2MHz = 0.42):
        """
        Estimate frequency offset and correct frequency data for bursts in an IQ signal.

        Parameters:
        - signal (np.array): Complex-valued IQ samples.
        - rise_index (np.array): Indices of burst starts.
        - fall_index (np.array): Indices of burst ends.
        - delta_time_v (np.array): Durations of bursts in seconds.
        - fs (float): Sampling frequency in Hz.

        Returns:
        - frequency_offsets (list): List of frequency offsets (in MHz) for each burst.
        - corrected_frequencies (list[np.array]): List of arrays containing corrected frequency data (in MHz) for each burst.
        - delta_time_index (int): Index of the burst used for frequency correction.
        """
        frequency_offsets = []
        corrected_frequencies = []
        delta_time_index = None
        ble_type = None

        samples_per_us = round(symbol_rate * fs)
        true_symbol_rate = 1e-6
        packet_len = 376
        preamble_addres = 'aad6be898e'

        for i in range(len(delta_time_v)):
            if delta_time_v[i] < 100e-6:  # Skip bursts shorter than 250 μs
                continue
            # Extract the current burst
            burst_signal = signal[rise_index[i]:fall_index[i]]

            # Phase extraction
            burst_phase = np.unwrap(np.angle(burst_signal)) / (2 * np.pi)

            # Frequency estimation over 1 μs windows
            burst_freq = burst_phase[samples_per_us:] - burst_phase[0:-samples_per_us]

            # Rough frequency offset estimation using linear phase fitting
            phase_segment = burst_phase[samp_skip:]  # Skip first samples for LO to stabilize
            time_segment = np.arange(len(phase_segment)) / fs
            coef = np.polyfit(time_segment, phase_segment, 1)

            frequency_offset = coef[0] * 1e-3  # Convert from kHz to MHz
            
            #Estimate frequecny deviation 
            phase_over_usec = phase_segment[samples_per_us:] - phase_segment[:-samples_per_us]
            phase_mid = phase_over_usec[samples_per_us*30:-samples_per_us*30] - frequency_offset*1e-3
            
            if(np.max(np.abs(phase_mid)) > dF_2MHz):
                true_symbol_rate = 0.5e-6
                packet_len = 480
                preamble_addres = 'aaaad6be898e'
                ble_type = 'BLE5-DBL-EXT'
            elif (delta_time_v[i] > 250e-6):
                true_symbol_rate = 1e-6
                packet_len = 376
                preamble_addres = 'aad6be898e'
                ble_type = 'LEGACY'
            else:
                true_symbol_rate = 1e-6
                packet_len = 120
                preamble_addres = 'aad6be898e'
                ble_type = 'BLE5-AUX-PTR'

            # Correct frequency data
            corrected_freq = burst_freq - frequency_offset * 1e-3

            # Store results
            frequency_offsets.append(frequency_offset)
            corrected_frequencies.append(corrected_freq)
            delta_time_index = i

        return frequency_offsets, corrected_frequencies, delta_time_index, true_symbol_rate, packet_len, preamble_addres , ble_type

    @staticmethod
    def detect_preamble(preamble_bits, fs, data_freq, skip_samp=10, symbol_rate = 1e-6):
        """
        Detect preamble in frequency-demodulated FSK data using correlation.

        Parameters:
        - preamble_bits (np.array): 1D array of bits representing the preamble.
        - fs (float): Sampling frequency in Hz.
        - data_freq (np.array): Frequency-demodulated data.
        - skip_samp (int): Number of initial samples to skip in the correlation search.

        Returns:
        - corr_output (np.array): Correlation values at each starting point.
        - preamble_index (int): Index of detected preamble (or None if not found). 
        """
        sync_preamble_length = len(preamble_bits)
        samples_per_us = int(np.round(symbol_rate * fs))

        # Allocate correlation output vector
        max_index = int(len(data_freq) - (samples_per_us * sync_preamble_length))
        if max_index <= 0:
            raise ValueError("The frequency data is too short for preamble detection.")
        corr_output = np.zeros(max_index)

        # Normalize preamble to match the MATLAB behavior
        preamble_symbols = np.array(preamble_bits) - 0.5

        # Perform correlation search
        for ind_n in range(skip_samp, max_index):
            # Extract samples corresponding to the preamble 
            curr_input = data_freq[ind_n : ind_n + samples_per_us * sync_preamble_length : samples_per_us]

            if len(curr_input) == sync_preamble_length:
                # Dot product correlation
                corr_output[ind_n] = np.abs(np.dot(preamble_symbols, curr_input))

        # Find index of maximum correlation
        preamble_index = np.argmax(corr_output)
        if corr_output[preamble_index] == 0:
            preamble_index = None

        return corr_output, preamble_index

    @staticmethod
    def generate_ble_scrambler(channel = 37, seed_length = 336):
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
            next_stage[4] = LFSR_FIFO[3] ^ LFSR_FIFO[-1]  # XOR
            next_stage[5] = LFSR_FIFO[4]
            next_stage[6] = LFSR_FIFO[5]

            LFSR_FIFO = next_stage

        return scrambler_seed
    
    @staticmethod
    def process_interpolated_burst(signal, preamble_index, fs, fs_inter, 
                                frequency_offset, num_symbols=376,symbol_time = 1e-6):
        """
        Upsample burst signal, estimate symbol timing adjustment, and demodulate to soft and hard bits.

        Parameters:
        - signal: np.array, complex-valued burst signal at low sampling rate.
        - preamble_index: int, index of preamble in low-rate samples.
        - fs: float, original sampling frequency.
        - fs_inter: float, high (interpolated) sampling frequency.
        - frequency_offset: float, frequency offset in MHz.
        - num_sym_hypo: int, range of timing hypotheses to test (default: 1).
        - num_symbols: int, number of symbols to extract (default: 376).

        Returns:
        - soft_bits: np.array, soft bits (frequency deviations in MHz) after optimal timing recovery.
        - hard_bits: np.array, hard bits (1 or -1) from soft bits.
        - updated_timing: int, optimal symbol period in high-rate samples.
        - est_symbol_clock: float, estimated symbol clock in MHz.
        """

        # Resample burst signal to high rate (upsampling)
        up = int(np.round(fs_inter / fs))
        down = 1
        data_inter = resample_poly(signal, up, down)

        num_sym_hypo= int(np.ceil(1e-12*fs_inter))
        # Extract phase and frequency at high sampling rate
        phase_inter = np.unwrap(np.angle(data_inter)) / (2 * np.pi)
        samps_in_symbol_inter = int(np.round(symbol_time * fs_inter))

        freq_inter = (phase_inter[samps_in_symbol_inter:] - phase_inter[:-samps_in_symbol_inter])
        freq_inter -= frequency_offset * 1e-3  # MHz correction

        # Map preamble index to high-rate samples
        preamble_index_inter = round((preamble_index - 1) * fs_inter / fs)

        # Optimize timing offset by maximizing soft symbol variance
        variances = []
        for sym_change in range(-num_sym_hypo, num_sym_hypo + 1):
            updated_timing = samps_in_symbol_inter + sym_change

            # Sample symbols with this timing hypothesis
            curr_indices = np.arange(
                preamble_index_inter,
                preamble_index_inter + updated_timing * num_symbols,
                updated_timing,
            )

            # Remove indices beyond available data
            curr_indices = curr_indices[curr_indices < len(freq_inter)]

            # Extract soft symbols
            curr_soft_bits = freq_inter[curr_indices.astype(int)]

            # Store variance of extracted soft symbols
            variances.append(np.std(curr_soft_bits))

        # Select the best timing offset
        max_timing_index = np.argmax(variances)
        best_timing_offset = max_timing_index - num_sym_hypo
        updated_timing = samps_in_symbol_inter + best_timing_offset

        # Estimate symbol clock in MHz
        est_symbol_clock = updated_timing / fs_inter * 1e6  # Symbol duration in us

        # Extract final soft bits using the best timing
        final_indices = np.arange(preamble_index_inter, len(freq_inter), updated_timing)
        soft_bits = freq_inter[final_indices.astype(int)]

        # Hard decision based on soft bits
        hard_bits = np.where(soft_bits >= np.average(soft_bits), 1, -1)

        return soft_bits, hard_bits, updated_timing, est_symbol_clock
    
    @staticmethod
    def calc_drift(soft_bits):
        soft_ones = soft_bits[soft_bits>np.average(soft_bits)]
        soft_ones_index = np.where(soft_bits>np.average(soft_bits))[0]
        [drift , freq_key] = np.polyfit(soft_ones_index,soft_ones,1)
    
        return [drift , freq_key]
    
    @staticmethod
    def calc_avg_df(soft_bits,symbol_rate = 1e-6):
        # Intuition: calculating the "distance" between the linear fit of the soft "ones" and "zeros" relative to their mean
        soft_ones = soft_bits[soft_bits>np.average(soft_bits)]
        soft_ones_index = np.where(soft_bits>np.average(soft_bits))[0]
        [drift_ones , freq_key_ones] = np.polyfit(soft_ones_index,soft_ones,1)

        soft_zeros = soft_bits[soft_bits<np.average(soft_bits)]
        soft_zeros_index = np.where(soft_bits<np.average(soft_bits))[0]
        [drift_zeros , freq_key_zeros] = np.polyfit(soft_zeros_index,soft_zeros,1)

        # the distance between the two lines of f(t) = drift*t + freq_key -> df = f_1(t) - f_0(t) ~ freq_key_ones - freq_key_zeros (assuming the drift is similiar) 
        avg_df_cycles = freq_key_ones - freq_key_zeros
        
         # Convert to Hz: Δf = Δφ / T_sym , the chnage from '0'(f) to '1'(f) is twice the deviation from carrier
        avg_df_Hz = avg_df_cycles / (2*symbol_rate)

        # Return frequecny deviation in MHz
        avg_df_MHz = avg_df_Hz / 1e6
    
        return avg_df_MHz
    
    @staticmethod
    def calc_freq_offset_by_peak(signal, fs, input_center, plot_on = False, ch_freq = None):
        N = len(signal)
        xdelta = 1/fs
        fft_result = np.fft.fft(signal)
        fft_freqs = np.fft.fftfreq(N, d= xdelta)

        # Shift the FFT output
        fft_result_shifted = np.fft.fftshift(fft_result)
        fft_freqs_shifted = np.fft.fftshift(fft_freqs) + input_center
        
        peak_ind = np.argmax(fft_result_shifted)
        peak_freq = fft_freqs_shifted[peak_ind]
        
        # Calculating the frequecny offset relative to ch_freq, if ch_freq is None than by input_center
        if(ch_freq is None):
            freq_offset = input_center - peak_freq
        else:
            freq_offset = ch_freq - peak_freq
        
        if(plot_on):   
            # Plot the magnitude spectrum
            plt.figure(figsize=(10, 6))
            plt.plot(peak_freq, 10*np.log(fft_result_shifted[peak_ind]), marker='x',markersize=10, color = 'red')
            plt.plot(fft_freqs_shifted, 10*np.log(np.abs(fft_result_shifted)))
            plt.title('IQ Data Plot')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude [dB]')
            plt.show()
        
        return freq_offset
        
  
    @staticmethod
    def return_iq_rec_contant(mat_file_path):
        try:
            # Load the IQ file data and unfold it
            data = sio.loadmat(mat_file_path)

            iq_data = data['Y'].flatten()
            input_center = data['InputCenter'].item()
            span = data['Span'].item()
            xdelta = data['XDelta'].item()
            fs = 1 / xdelta
            return iq_data , input_center, span, fs
        except Exception as e:
            return e

    @staticmethod
    def calc_packet_start_time(mat_file_path: str, start_time):
        """
        Get the creation_time of a record and return the time from start_time.
        
        If the filename matches the pattern 'iq_<freq>MHz_YYYYMMDD_HHMMSS_microseconds.mat',
        extract the timestamp from the filename. Otherwise, use the file system creation time.
        """
        # Try to extract timestamp from filename
        # Pattern: iq_<freq>MHz_YYYYMMDD_HHMMSS_microseconds.mat
        filename = os.path.basename(mat_file_path)
        pattern = r'iq_\d+MHz_(\d{8})_(\d{6})_(\d+)\.mat'
        match = re.match(pattern, filename)
        
        if match:
            date_str = match.group(1)  # YYYYMMDD
            time_str = match.group(2)  # HHMMSS
            microseconds_str = match.group(3)  # microseconds
            
            # Parse the datetime
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            
            # Add microseconds and convert to timestamp
            creation_time = dt.timestamp() + int(microseconds_str) / 1_000_000.0
        else:
            # Fall back to file system creation time
            creation_time = os.path.getctime(mat_file_path)
        
        time_from_start = creation_time - start_time
        return float(time_from_start)

    @staticmethod
    def calc_avg_power (signal:np.array):
        if(len(signal) != 0):
            p_avg_dbm = 10*np.log(sum(np.abs(signal))/len(signal)) + 30
        else:
            p_avg_dbm = 999
        return p_avg_dbm
    
    @staticmethod
    def plot_soft_bits(soft_bits, f_offsets, mat_file_path, drift, freq_key, save_path, plot_name):
        """
        Plots soft bits and drift for an IQ recording.

        Parameters:
        - soft_bits: Array of soft symbols.
        - f_offsets: Frequency offsets (list or array).
        - mat_file_path: Path to the .mat file.
        - drift: Frequency drift per symbol.
        - freq_key: Initial frequency key for the linear drift.
        - save_path: Directory where the plot should be saved.
        - plot_name: Name of the output plot file.
        """
        plt.plot(soft_bits, 'x', markersize=5)
        plt.suptitle(f'IQ Recording Soft Bits, f_offset = {f_offsets[0]} KHz')
        plt.ylabel("Soft Symbols [MHz]")
        plt.xlabel("Bit Number")
        plt.title(f'File path: {mat_file_path}', fontdict={'fontsize': 7})

        # Plot drift
        lin_reg = np.arange(len(soft_bits)) * drift + freq_key
        plt.plot(lin_reg, label=f'Drift: {drift * 1e3} kHz/Symbol \n')
        plt.legend()
        
        # Save plot
        save_filepath = os.path.join(save_path, f'{plot_name}.jpeg')
        plt.savefig(save_filepath)
        plt.clf()

    @staticmethod
    def build_results_payload(
         time_from_start: float, 
         raw_packet: str = None,
         drift = None, 
         f_offset: float = None,
         p_avg_dbm: float = None, 
         crc: str = None,
         soft_bits:np.array=None,
         avg_df: float = None, 
         ble_type: str = None):
        results = {
                "time_from_start" : time_from_start,
                'VERSION':VERSION,
                "raw_packet": raw_packet, 
                "drift": drift,
                "frequecny_offest":f_offset, 
                "avg_power":p_avg_dbm,
                "crc": crc,
                "soft_bits": soft_bits,
                "avg_df": avg_df,
                "ble_type": ble_type}
        
        return results
        
    @staticmethod
    def freq_to_ble_channel(freq:int = 2402):
        ch = 37
        freq = freq*1e-6 
        freq_ch = freq
        if (freq > 2400 and freq < 2404):
            ch = 37
            freq_ch = 2402e6
        if (freq < 2426 and freq > 2422):
            ch = 10
            freq_ch = 2426e6
        return ch , freq_ch
       
    @staticmethod
    def process_single_iq_file(mat_file_path: str, save_path: str = None,
                            start_time: float = 0 , debug_flag: bool = False):
        """
        Processes an IQ recording from a .mat file using the IQPostProcessing class.

        Parameters:
        - mat_file_path (str): Path to the .mat file.
        - save_path (str) : Path to save plots if needed
        - preamble_addres (str) : Preamble adress (default aad6be898e)
        - start_time (float) : Staring time relative to records (UTC)

        Returns:
        - results (dict) : 
            * Contains the recording basename as a key
            * Values are packet_hex (descramable), freq_offset and drift (in kHz & kHz/Symbol)
        """
               
        # Set name and time_from_start
        plot_name = os.path.basename(mat_file_path)
        time_from_start = IQPostProcessing.calc_packet_start_time(mat_file_path,start_time)
        
        # Define output 
        result = {}

        # Extract IQ data and metadata
        try:
            iq_data , input_center, span, fs = IQPostProcessing.return_iq_rec_contant(mat_file_path=mat_file_path)
        except Exception as e:
            debug_flag and print(f"Cannot load file [{mat_file_path}] due to [{e}]")
            result[plot_name] =  IQPostProcessing.build_results_payload(time_from_start=time_from_start)
            return result
        # Caculate avrage power in dBm 
        p_avg_dbm = IQPostProcessing.calc_avg_power(iq_data)

        # Skip if the packet power is near\below the noise floor (here ~ -80dBm)
        if p_avg_dbm < -80 or p_avg_dbm == 999:
            debug_flag and print(f"Packet power is below -80dBm")
            return 
        
        # Zero padding the IQ data to avoid missing envelope detection
        iq_data = np.concatenate([np.zeros(495, dtype=iq_data.dtype), iq_data])
        
        # Detect bursts in the signal
        delta_time, rise_index, fall_index = IQPostProcessing.burst_detect_by_envelope(iq_data, fs)

        # Estimate frequency offset
        f_offsets, freq_data, delta_time_index, true_symbol_rate, packet_len, preamble_addres , ble_type = IQPostProcessing.estimate_frequency_offset(
            signal=iq_data, rise_index=rise_index, fall_index=fall_index,
            delta_time_v=delta_time, fs=fs, symbol_rate= 1e-6
        )

        if not f_offsets or not freq_data:
            debug_flag and print("The packet is too short")
            result[plot_name] = IQPostProcessing.build_results_payload(time_from_start=time_from_start, p_avg_dbm =p_avg_dbm)
            return result
        
        # Update symbol time for preamble detection
        symbol_rate = true_symbol_rate
        packet_len = packet_len
        f_c = np.round(input_center + 1e3*f_offsets[0]) 
        
        # Calculate BLE channel
        ch , freq_ch = IQPostProcessing.freq_to_ble_channel(f_c)
        
        # Generate preamble and scrambler seed
        preamble_bit_vector = IQPostProcessing.hex_to_bit_vector(preamble_addres)
        sync_preamble_length = len(preamble_bit_vector)
        scram_seed = IQPostProcessing.generate_ble_scrambler(channel=ch, seed_length=packet_len-sync_preamble_length).astype(int)
        
        # Detect preamble
        corr_output, preamble_index = IQPostProcessing.detect_preamble(preamble_bits=preamble_bit_vector,
                                                                       fs=fs, data_freq=freq_data[0],symbol_rate=symbol_rate)
        
        # Extract relevant IQ data
        filtered_iq_data = iq_data[rise_index[delta_time_index]:fall_index[delta_time_index]]

        # Process IQ data with symbol interpolation
        soft_bits, hard_bits, updated_timing, est_symbol_clock = IQPostProcessing.process_interpolated_burst(
            signal=filtered_iq_data, preamble_index=preamble_index,frequency_offset=f_offsets[0],
            fs=fs, fs_inter=200 * fs,num_symbols=packet_len, symbol_time=symbol_rate
        )

        #calculating drift
        drift, freq_key = IQPostProcessing.calc_drift(soft_bits)
        drift = drift * packet_len
        
        #calculating DF
        avg_df = IQPostProcessing.calc_avg_df(soft_bits=soft_bits,symbol_rate=symbol_rate)

        #
        
        # Descramble the data
        harder_bits = ((hard_bits + 1) * 0.5).astype(int)
        if (len(hard_bits) >= packet_len):
            packet = np.bitwise_xor(harder_bits[sync_preamble_length:packet_len], scram_seed)
            packet_hex = IQPostProcessing.bits_to_hex_right_msb(packet)
        else: #packet is not full
            result[plot_name] = IQPostProcessing.build_results_payload(time_from_start=time_from_start, drift= drift * 1e3,
                                 f_offset=(f_c - freq_ch)*1e-3, p_avg_dbm=p_avg_dbm, avg_df=avg_df ,ble_type= ble_type)
            return result
            
        debug_flag and print(f"{os.path.basename(mat_file_path)}: \n{packet_hex}\n")

        
           
        if save_path is not None:
            # Plot soft bits
            IQPostProcessing.plot_soft_bits(freq_key=freq_key, f_offsets=f_offsets,
                                            mat_file_path=mat_file_path, drift=drift,
                                            save_path= save_path, plot_name=plot_name,
                                            soft_bits= soft_bits)
            
        result[plot_name] = IQPostProcessing.build_results_payload(
                time_from_start=time_from_start, 
                raw_packet=(packet_hex[:-6] + '000000'), 
                drift= drift * 1e3, 
                f_offset=(f_c - freq_ch)*1e-3,
                p_avg_dbm=p_avg_dbm, 
                avg_df=avg_df, 
                crc=packet_hex[-6:], 
                soft_bits=soft_bits.tolist(),
                ble_type = ble_type
            )
        
        return result
        

    @staticmethod
    def process_directory_iq_files(
        directory_path: str,
        start_time: float=0,
        save_path: str = None,
        debug_flag: bool =False
        ):
        """
        Processes all IQ recording .mat files in a given directory using the IQPostProcessing class.

        Parameters:
        - directory_path (str): Path to the directory containing .mat files.
        - save_path (str, optional): Path to save plots if needed (default: None).
        - preamble_addres (str, optional): Preamble address (default: 'P').

        Returns:
        - results (dict): Dictionary containing the processing results for each file.
        """

        results = {}

        # Get a list of all .mat files in the directory
        mat_files = [f for f in os.listdir(directory_path) if f.endswith('.mat') or f.endswith('.MAT')]

        # Process each .mat file
        for mat_file in mat_files:
            mat_file_path = os.path.join(directory_path, mat_file)
            debug_flag and print(f"Processing file: {mat_file_path}")

            # Process the file using the existing function
            file_results = IQPostProcessing.process_single_iq_file(
                mat_file_path=mat_file_path,
                save_path=save_path, 
                start_time=start_time,
                debug_flag= debug_flag
                )

            # Store the result in the dictionary
            if file_results:
                results.update(file_results)

        return results            
     

if __name__ == "__main__":
     IQPost = IQPostProcessing()
    results = IQPost.process_directory_iq_files(
       r"C:\Users\boaz.arnon\Downloads\matfiles",
       save_path=r"C:\Users\boaz.arnon\Downloads\matfiles",
       debug_flag=True
    )