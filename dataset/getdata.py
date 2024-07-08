from sklearn.linear_model import LinearRegression
import librosa
import pandas as pd
import numpy as np
import os

# 设置工作目录
project_root = "D:/pml/machine-learning"
os.chdir(project_root)

# 确认当前工作目录
print("Current working directory:", os.getcwd())

# 读取音频文件
audio_path = 'dataset/humbugdb_neurips_2021_1/53.wav'
y, sr = librosa.load(audio_path, sr=None)




# 加载CSV文件
# csv_file = 'labels.csv'
# labels_df = pd.read_csv(csv_file)

# 查看CSV文件内容
# print(labels_df.head())

# 提取所有音频文件的特征
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # 加载音频文件
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 提取MFCC特征
    mfccs_mean = np.mean(mfccs, axis=1)  # 计算MFCC的均值
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))  # 计算零交叉率
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # 计算频谱质心
    return np.hstack((mfccs_mean, zcr, spectral_centroid))


print(extract_features(audio_path))
