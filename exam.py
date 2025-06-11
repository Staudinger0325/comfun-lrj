import numpy as np
import matplotlib.pyplot as plt
import math
import comtheory as ct
from scipy.io import wavfile
import pygame
import os
import random

#==========1. 信源
#导入音频
import_audio_rate, import_audio_data = wavfile.read('天雷滚滚.wav')
audio_data1 = import_audio_data[import_audio_rate:6*import_audio_rate,0]
audio_data2 = import_audio_data[import_audio_rate:6*import_audio_rate,1]


#导入数据
if not os.path.exists('bitstream.bin'):  # 检查文件是否存在
    print("文件不存在，请先生成比特序列并保存到本地文件")
else:  # 从本地文件读取比特流  
    with open('bitstream.bin', 'rb') as f:
        imported_bitstream = f.read()
bit_data = []
for byte in imported_bitstream:
    binary_str = bin(byte)[2:].zfill(8)   #将每个字节转换为二进制字符串，然后去掉前缀 '0b'，并填充到8位
    bit_data.extend([int(bit) for bit in binary_str])    #将二进制字符串转换为列表，并将字符转换为整数

#定义时间轴t和频率轴f
Fs = import_audio_rate
dt = 1/Fs
T_obs = len(audio_data1)*dt  #5s
t=np.arange(0,T_obs,dt)
N_obs = len(t)
df =1/T_obs
f=np.arange(-N_obs/2*df,N_obs/2*df,df)


#需要传输的原始音频数据：audio_data1，audio_data2
#需要传输的比特数据：bit_data


#==========2. 发送模块
#-----发送端各模块实现





#-----发送到信道上的已调信号统一命名为s_t
s_t = np.zeros(len(t)) #这里装已调信号
ff,S_f = ct.FourierTransfrom(t,s_t)


#==========3. 信道
#理想限带信道的冲激响应h_channel_t和传递函数H_channel_f
BPF_low_freq = 900
BPF_high_freq = 1100
H_channel_f = 0.01*np.where((np.abs(f) >= BPF_low_freq) & (np.abs(f) <= BPF_high_freq), 1, 0)
tt,h_channel_t = ct.RFourierTransfrom(f,H_channel_f)

#宽带高斯噪声
N0 = 1e-10 #噪声单边功率谱密度
sigma = np.sqrt(N0*Fs)
nw_t = np.random.normal(loc=0, scale=sigma, size=len(t))

#信号s(t)通过信道
Y_f = S_f*H_channel_f
tt,y_t = ct.RFourierTransfrom(f,Y_f)
y_t = y_t+nw_t


#==========4. 接收模块
#-------接收端各模块实现（接收信号是y_t）




#==========5. 信宿
#-------有效性、可靠性分析计算


#-------音频播放代码参考
# audio_data_out=np.column_stack((audio_data1, audio_data2))
# pygame.mixer.init(frequency=import_audio_rate)  #创建一个pygame的音频对象，初始化mixer，设置采样率
# sound = pygame.sndarray.make_sound(audio_data_out)  #创建声音对象
# sound.play()  #播放声音