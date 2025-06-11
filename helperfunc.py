#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Chencheng

Course1 2025-06-11
"""

import random
import numpy as np
import pylab as pl
import scipy .signal as signal
from scipy import fftpack
import math

_rrc_filter_cache = {}

# 生成卷积码
def conv_encode(data_bits, G):
    """
@desc:	对输入比特序列进行卷积编码。
@para:	data_bits (list): 输入的位序列。
@para:	G (numpy.ndarray): 生成多项式矩阵，形状为 (n, L)，其中 n 是输出比特数，L 是约束长度。G[i, j] 表示第 i 个输出比特是否受 j 延迟的输入比特影响。
@rtrn:	list: 编码后的比特序列。
    """
    n, L = G.shape
    state = [0] * (L - 1)
    encoded_bits = []

    for bit in data_bits:
        current_state = [bit] + state
        output_bits_frame = []
        for i in range(n):
            output_bit = 0
            for j in range(L):
                output_bit = (output_bit + G[i, j] * current_state[j]) % 2
            output_bits_frame.append(output_bit)
        encoded_bits.extend(output_bits_frame)
        state = current_state[:-1]

    for _ in range(L - 1):
        bit = 0
        current_state = [bit] + state
        output_bits_frame = []
        for i in range(n):
            output_bit = 0
            for j in range(L):
                output_bit = (output_bit + G[i, j] * current_state[j]) % 2
            output_bits_frame.append(output_bit)
        encoded_bits.extend(output_bits_frame)
        state = current_state[:-1]

    return encoded_bits



# 将音频数据转换为位流
def int16_to_bit_list_fast(int16_array):
    """
@desc:	使用更加底层的办法，优化转换为序列的速度
@para:	int16_array (np.ndarray): 输入的 int16 NumPy 数组。
@rtrn:	list: 包含所有比特的列表 (0 或 1)。
    """
    if not isinstance(int16_array, np.ndarray):
        int16_array = np.array(int16_array, dtype=np.int16)

    uint16_array = int16_array.astype(np.uint16)
    bits_2d = ((uint16_array[:, np.newaxis] >> np.arange(16)) & 1)[:, ::-1]

    return bits_2d.flatten().tolist()



#交织
def interleave(data_bits, depth, width):
    """
@desc:	对输入比特序列进行块交织。
@para:	data_bits (list): 输入的比特序列 (0 或 1)。
@para:	depth (int): 交织器的行数。
@para:	width (int): 交织器的列数。
@rtrn:	list: 交织后的比特序列。
    """
    interleaver_capacity = depth * width

    padding_needed = (interleaver_capacity - (len(data_bits) % interleaver_capacity)) % interleaver_capacity
    padded_data_bits = data_bits + [0] * padding_needed

    if len(padded_data_bits) < interleaver_capacity:
        padded_data_bits.extend([0] * (interleaver_capacity - len(padded_data_bits)))
    elif len(padded_data_bits) % interleaver_capacity != 0:
        padded_data_bits.extend([0] * (interleaver_capacity - (len(padded_data_bits) % interleaver_capacity)))

    interleaved_bits = []

    for i in range(0, len(padded_data_bits), interleaver_capacity):
        current_block = padded_data_bits[i : i + interleaver_capacity]
        
        if len(current_block) < interleaver_capacity:
            current_block.extend([0] * (interleaver_capacity - len(current_block)))
        elif len(current_block) > interleaver_capacity:
            current_block = current_block[:interleaver_capacity]

        matrix = []
        for r in range(depth):
            row = []
            for c in range(width):
                row.append(current_block[r * width + c])
            matrix.append(row)

        for c in range(width):
            for r in range(depth):
                interleaved_bits.append(matrix[r][c])
    
    return interleaved_bits



# 64QAM星座映射
def qam64_mapper(bit_stream):
    """
@desc:	将比特流映射为64-QAM的I路和Q路PAM电平，使用NumPy优化。
        每6个比特映射为一个符号。如果比特流长度不是6的倍数，则进行零填充。
@para:	bit_stream (list or np.ndarray): 输入的比特序列 (0 或 1)。
@rtrn:	tuple: (I_levels, Q_levels)，均为np.ndarray，分别代表I路和Q路的PAM电平。
    """
    if not isinstance(bit_stream, np.ndarray):
        bit_stream = np.array(bit_stream, dtype=np.int8)

    # 确保比特流长度是6的倍数，不足则补零
    padding_needed = (6 - (len(bit_stream) % 6)) % 6
    if padding_needed > 0:
        padded_bit_stream = np.pad(bit_stream, (0, padding_needed), 'constant', constant_values=0)
    else:
        padded_bit_stream = bit_stream

    num_symbols = len(padded_bit_stream) // 6
    bit_groups = padded_bit_stream.reshape(num_symbols, 6)

    # 定义8-PAM格雷码映射表
    pam8_map = np.array([
        -7, -5, -1, -3, +7, +5, +1, +3
    ], dtype=np.int8)

    # 提取I分量对应的3比特 (b5 b4 b3)
    weights_3bit = np.array([4, 2, 1], dtype=np.int8)
    i_bit_integers = np.dot(bit_groups[:, :3], weights_3bit)

    # 提取Q分量对应的3比特 (b2 b1 b0)
    q_bit_integers = np.dot(bit_groups[:, 3:], weights_3bit)

    # 映射整数到PAM电平
    I_levels = pam8_map[i_bit_integers]
    Q_levels = pam8_map[q_bit_integers]

    # 返回I路和Q路电平
    return I_levels, Q_levels

import numpy as np
import math



# RCC滤波器
def raised_cosine_filter(input_symbols, Rs, Fs_system, alpha, num_taps=48):
    """
    @desc:	对输入的基带符号序列进行升余弦脉冲成形滤波。
            将低速率符号序列插值到Fs_system采样率，并应用升余弦滤波器。
    @para:	input_symbols (np.ndarray): 输入的基带符号序列 (实数)。
    @para:	Rs (float): 符号速率 (Hz)。
    @para:	Fs_system (float): 系统采样率 (Hz)。
    @para:	alpha (float): 滚降系数 (0 <= alpha <= 1)。
    @para:	num_taps (int, optional): 滤波器的抽头数。默认为48。
    @rtrn:	np.ndarray: 经过脉冲成形后的高采样率数字基带信号。
    """
    if not isinstance(input_symbols, np.ndarray):
        input_symbols = np.array(input_symbols, dtype=float)

    # 计算每个符号的采样点数 (Samples per Symbol, Sps)
    samples_per_symbol = Fs_system / Rs
    if not np.isclose(samples_per_symbol, round(samples_per_symbol)):
        print(f"警告: Fs_system ({Fs_system}) / Rs ({Rs}) = {samples_per_symbol} 不是整数。 "
              f"将四舍五入为 {int(round(samples_per_symbol))}")
    samples_per_symbol = int(round(samples_per_symbol))

    # num_taps 必须是奇数，以确保滤波器是中心对称的
    if num_taps % 2 == 0:
        num_taps += 1 # 确保滤波器长度为奇数

    # 生成RRC滤波器的时间轴
    half_taps = (num_taps - 1) // 2
    t = np.arange(-half_taps, half_taps + 1) / Fs_system # 以秒为单位的时间轴

    # 计算RRC滤波器系数 (h_rrc)
    h_rrc = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if ti == 0:
            h_rrc[i] = 1 - alpha + 4 * alpha / np.pi
        elif np.isclose(np.abs(ti), 1 / (4 * alpha * Rs)) and alpha != 0:
            # 特殊点：当 t = +/- 1/(4*alpha*Rs) 时
            h_rrc[i] = alpha / np.sqrt(2) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha))
            )
        else:
            # 通用公式
            denominator = np.pi * Rs * ti * (1 - (4 * alpha * Rs * ti)**2)
            if denominator == 0:
                # 理论上，当分母为0时，ti 应该就是特殊点，但为了鲁棒性，这里处理一下
                h_rrc[i] = 0
            else:
                numerator = np.sin(np.pi * Rs * ti * (1 - alpha)) + \
                            4 * alpha * Rs * ti * np.cos(np.pi * Rs * ti * (1 + alpha))
                h_rrc[i] = numerator / denominator

    # 归一化滤波器
    if np.sum(h_rrc) != 0:
        h_rrc = h_rrc / np.sum(h_rrc)
    else:
        print("警告: 滤波器系数和为零，可能导致归一化问题。")

    # 脉冲成形
    upsampled_symbols = np.zeros(len(input_symbols) * samples_per_symbol)
    upsampled_symbols[::samples_per_symbol] = input_symbols
    shaped_signal = np.convolve(upsampled_symbols, h_rrc, mode='full')

    return shaped_signal



# 载波调制
def qam_modulate(I_signal, Q_signal, fc, Fs_system_oversampled):
    """
@desc:	对QAM基带I路和Q路信号进行载波调制，生成实数形式的射频信号。
        此函数使用一个指定的过采样系统采样率 Fs_system_oversampled 来生成时间轴和载波。
@para:	I_signal (np.ndarray): 经过脉冲成形后的I路基带信号，其采样率应与 Fs_system_oversampled 匹配。
@para:	Q_signal (np.ndarray): 经过脉冲成形后的Q路基带信号，其采样率应与 Fs_system_oversampled 匹配。
@para:	fc (float): 载波频率 (Hz)。
@para:	Fs_system_oversampled (float): 过采样后的系统采样率 (Hz)。
@rtrn:	np.ndarray: 实数形式的调制信号 (I * cos(2*pi*fc*t) - Q * sin(2*pi*fc*t))。
    """
    # 确保I_signal和Q_signal长度相同
    if len(I_signal) != len(Q_signal):
        raise ValueError("I_signal and Q_signal must have the same length.")

    # 根据信号长度和新的过采样率生成时间轴
    # 假设 I_signal 和 Q_signal 已经是 Fs_system_oversampled 采样率下的信号
    num_samples = len(I_signal)
    t = np.arange(0, num_samples) / Fs_system_oversampled

    # 生成载波
    carrier_cos = np.cos(2 * np.pi * fc * t)
    carrier_sin = np.sin(2 * np.pi * fc * t)

    # QAM调制公式：s(t) = I(t) * cos(2*pi*fc*t) - Q(t) * sin(2*pi*fc*t)
    modulated_signal = I_signal * carrier_cos - Q_signal * carrier_sin
    return modulated_signal