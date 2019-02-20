import pickle
from scipy import signal

import numpy as np
from scipy import stats
from libs import detect_peaks

nLabel, nTrial, nUser, nChannel, nTime  = 4, 40, 32, 40, 8064

# 4features
# [ -2.79506181   1.65321851 180.49499424  85.03794643]
# [1.50774537e+01 1.69504305e+00 5.23609747e+04 3.46866120e+02]

# [ 4.49693675e+03,  4.55792115e+03,  8.89524389e+03, -1.84215991e+02,
#   2.39550614e+03,  6.17605211e+07,  9.07945988e+03, -1.07401505e-01,
#  -4.94559417e-01]
# [2.08400737e+09, 2.12768022e+09, 2.42667178e+09, 1.97765917e+09,
#  5.60220714e+07, 9.43489621e+17, 6.75454093e+08, 1.96134527e-01,
#  1.14776963e+00]
def convertData(adr = "E:/DataSet/data/features_clear_less.dat" ):
    print("Program started"+"\n")
    fout_data = open(adr,'w')
    fout_labels0 = open("E:\DataSet\data\labels_0.dat",'w')
    fout_labels1 = open("E:\DataSet\data\labels_1.dat",'w')
    fout_labels2 = open("E:\DataSet\data\labels_2.dat",'w')
    fout_labels3 = open("E:\DataSet\data\labels_3.dat",'w')

    # xq2_sum = np.zeros(9)
    # x_sum = np.zeros(9)
    # EX =np.zeros(9)
    # val = np.zeros(9)
    # total_num = 8064/384*32*40
    # cnt = 0
    # for i in range(32):  #nUser #4, 40, 32, 40, 8064
    #     if(i%1 == 0):
    #         if i < 10:
    #             name = '%0*d' % (2,i+1)
    #         else:
    #             name = i+1
    #     fname = "E:\DataSet\data_preprocessed_python\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
    #     f = open(fname, 'rb')
    #     x = pickle.load(f, encoding='latin1')
    #     print(fname)
    #
    #     for tr in range(nTrial):
    #         start = 0
    #         if(tr%1 == 0):
    #             for dat in range(nTime):
    #                  if dat != 0:
    #                     if(dat%384 == 0 or dat == 8063):
    #                     # if(dat%807 == 0 or dat == 8063):
    #                         features = extract_data(x['data'][tr][38][start:dat])
    #                         start = dat
    #                         cnt = cnt + 1
    #                         for i,fea in enumerate(features):
    #                             EX[i] = EX[i] + fea/total_num
    #                             # xq2_sum[i] = xq2_sum[i] + fea**2/total_num
    #                             # x_sum[i] = xq2_sum[i] + fea/total_num
    #
    # for i in range(32):  #nUser #4, 40, 32, 40, 8064
    #     if(i%1 == 0):
    #         if i < 10:
    #             name = '%0*d' % (2,i+1)
    #         else:
    #             name = i+1
    #     fname = "E:\DataSet\data_preprocessed_python\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
    #     f = open(fname, 'rb')
    #     x = pickle.load(f, encoding='latin1')
    #     print(fname)
    #
    #     for tr in range(nTrial):
    #         start = 0
    #         if(tr%1 == 0):
    #             for dat in range(nTime):
    #                  if dat != 0:
    #                     if(dat%384 == 0 or dat == 8063):
    #                     # if(dat%807 == 0 or dat == 8063):
    #                         features = extract_data(x['data'][tr][38][start:dat])
    #                         start = dat
    #                         cnt = cnt + 1
    #                         for i,fea in enumerate(features):
    #                             val[i] = (fea - EX[i])**2 / total_num + val[i]
    #
    # # print(cnt)
    # print(EX)
    # print(val)
    EX = np.array([ 4.49693675e+03,  4.55792115e+03,  8.89524389e+03, -1.84215991e+02,
              2.39550614e+03,  6.17605211e+07,  9.07945988e+03, -1.07401505e-01,
             -4.94559417e-01])
    val = np.array([2.08400737e+09, 2.12768022e+09, 2.42667178e+09, 1.97765917e+09,
             5.60220714e+07, 9.43489621e+17, 6.75454093e+08, 1.96134527e-01,
             1.14776963e+00])

    std = val ** 0.5

    print("\n"+"Print Successful")

    for i in range(32):  #nUser #4, 40, 32, 40, 8064
        if(i%1 == 0):
            if i < 10:
                name = '%0*d' % (2,i+1)
            else:
                name = i+1
        fname = "E:\DataSet\data_preprocessed_python\s"+str(name)+".dat"     #C:/Users/lumsys/AnacondaProjects/Emo/
        f = open(fname, 'rb')
        x = pickle.load(f, encoding='latin1')
        print(fname)
        for tr in range(nTrial):
            start = 0
            if(tr%1 == 0):
                for dat in range(nTime):
                     if dat != 0:
                        if(dat%384 == 0 or dat == 8063):
                        # if(dat%807 == 0 or dat == 8063):
                            for ch in range(32):
                                features = extract_data(x['data'][tr][ch][start:dat])
                                # re_feas = rescale(features, EX, std)
                                if(tr == 1 and i == 1):
                                    print(features)
                                for fea in features:
                                    fout_data.write(str(fea)+ " ")
                            start = dat

                            fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                            fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                            fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                            fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                            fout_data.write("\n")

                #个性化特征
                # fout_data.write(str(tr)+ " ")
                # fout_data.write(str(i)+ " ")
                # 总
                # print(x['data'][tr][39][:].shape)
                # for data in datas:
                #     fout_data.write(str(data)+ " ")
                # fout_labels0.write(str(x['labels'][tr][0]) + "\n")
                # fout_labels1.write(str(x['labels'][tr][1]) + "\n")
                # fout_labels2.write(str(x['labels'][tr][2]) + "\n")
                # fout_labels3.write(str(x['labels'][tr][3]) + "\n")
                # fout_data.write("\n")#40个特征换行
    fout_labels0.close()
    fout_labels1.close()
    fout_labels2.close()
    fout_labels3.close()
    fout_data.close()
    print("\n"+"Print Successful")

def extract_data(target_data, a = 0):
    target_mean = target_data.mean(axis=a)
    target_median = np.median(target_data, axis=a)
    target_maximum = np.max(target_data, axis=a)
    target_minimum = np.min(target_data, axis=a)
    target_std = np.std(target_data, axis=a)
    target_var = np.var(target_data, axis=a)
    target_range = np.ptp(target_data, axis=a)
    target_skew = stats.skew(target_data, axis=a)
    target_kurtosis = stats.kurtosis(target_data, axis=a)

    features = [target_mean, target_median, target_maximum, target_minimum, target_std, target_var, target_range, target_skew, target_kurtosis]
    # features_rescale = rescale(features)
    return features

def extractFeatures(PPGdata):
    dataWindow = signal.medfilt(PPGdata, 3)

    # Power spectral density (self.sampleWindow/4 reading window, self.sampleWindow/(4*2) sequential window overlap)
    f, pxx = signal.welch(dataWindow, fs = 128, nperseg = 384/4)
    pxx = [10*np.log10(x) for x in pxx]

    # Mean amplitude of HF components (20 to 200Hz range)
    m = np.mean(pxx[20:200])

    # LF/HF component magnitude ratio
    lfhfratio = (pxx[1] + pxx[2]) / (pxx[4] + pxx[5])

    # Apply a 2nd order bandpass butterworth filter to attenuate heart rate components
    # other than heart signal's RR peaks.
    b, a = signal.butter(2, [1/100, 1/10], 'bandpass')
    dataWindow = signal.filtfilt(b, a, dataWindow)
    # Find the segment peaks' indices for peak occurrence variance feature.
    indices = detect_peaks.detect_peaks(dataWindow,mph = -5000, mpd = 50,edge='rising',show=False)
    peakVariance = np.finfo(np.float64).max
    if len(indices) > 1:
        peakVariance = np.var(np.diff(indices))

    # Calculate the heart rate number from number of peaks and start:end timeframe
    timeDifferenceInMinutes =  384 / 128 / 60

    # print(timeDifferenceInMinutes)
    heartRate = float(len(indices)) / timeDifferenceInMinutes

    # features = [m, lfhfratio, peakVariance, heartRate]
    features = [m, lfhfratio, peakVariance, heartRate]


    return features

def rescale(features, EX, std):
    re_feas = np.zeros(9)
    for i,fea in enumerate(features):
        re_feas[i] = (fea - EX[i])/std[i]

    return re_feas

if __name__ == '__main__':
    convertData(adr = "E:/DataSet/data/features_22680.dat");


