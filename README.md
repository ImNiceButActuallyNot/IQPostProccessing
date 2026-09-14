# IQPostProccessing
BLE4/BLE5  IQ data mat file parser from Tektronix RSA_API/SignalVu

## IQParser.py 

`process_single_iq_file` pipeline:

**1. Load IQ (`return_iq_rec_contant`)**
```
Y = I(t) + jQ(t)      fs = 1/XDelta      fc = InputCenter
```
Zero-pad 495 samples at head.

**2. Average Power Gate (`calc_avg_power`)**
```
P_dBm = 10*log10( mean(|I+jQ|) )
```
Drop file if `P_dBm < -80`.

**3. Burst Detection (`burst_detect_by_envelope`)**
```
E(t) = |I+jQ|      threshold = max(E)/10
```
Rising/falling crossings → `rise_index`, `fall_index`, `delta_time_v`.

**4. Frequency Offset + Packet-Type Classify (`estimate_frequency_offset`)**
```
theta(t) = unwrap(angle(I+jQ))      f_offset = polyfit(theta, degree=1)
```
1µs-differenced phase mid-burst decides type:

| max\|Δf/1µs\| > 0.42 MHz | burst > 250µs | type | symbol rate | packet_len | preamble |
|---|---|---|---|---|---|
| yes | – | BLE5-DBL-EXT | 0.5µs | 480 | aaaad6be898e |
| no | yes | LEGACY | 1µs | 376 | aad6be898e |
| no | no | BLE5-AUX-PTR | 1µs | 120 | aad6be898e |

**5. Preamble → Bits (`hex_to_bit_vector`)**
Hex → LSB-first bit vector.

**6. Preamble Detection (`detect_preamble`)**
```
C(n) = | sum_i (P_bits(i) - 0.5) * F(n + i*Tsym) |
```
`preamble_index = argmax(C)`, masked to envelope region.

**7. Scrambler (`generate_ble_scrambler`)**
BLE 7-bit LFSR seed for the detected channel.

**8. Interpolate + Symbol Timing (`process_interpolated_burst`)**
Upsample to `fs_inter = 200*fs`, recompute phase-derivative frequency track. For each timing hypothesis `Ts`:
```
Ts* = argmax over Ts of Var(S(Ts))
```
(best eye opening — note: master doc says "minimizes", code/formula both maximize).
→ `soft_bits` (kHz), `hard_bits` (±1, mean threshold).

**9. Drift & Deviation**
```
Drift[kHz/sym] = polyfit(soft_bits, degree=1)
avg_df = ( mean(S >= mean(S)) - mean(S < mean(S)) ) / 2
```

**10. Bits → Hex Packet**
Payload bits (preamble stripped) XOR scrambler → `bits_to_hex_right_msb`. Last 3 bytes = CRC, zeroed in `raw_packet`.

---

### Output per file
`time_from_start, raw_packet, drift[kHz], frequecny_offest[kHz], avg_power[dBm], crc, soft_bits, avg_df[kHz], ble_type, freq_data, samples_per_symbol` — filled progressively; a file that fails earlier keeps only the fields computed so far.

