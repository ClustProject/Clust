import numpy as np
from scipy import io
from scipy.stats import ranksums

from entropy.MCRDE.MCRDE import MCRDE

from utils.plot_Entropy import plot_Entropy
from utils.write_txt import write_txt


RRIs_CHF_path = './sample_data/RRIs_CHF_1000'                        # CHF(= Congestive Heart Failure) -> 울혈성 심부전 피험자 14명의 RRIs data
RRIs_HEALTHY_path = './sample_data/RRIs_HEALTHY_1000'                # HEALTHY                         -> 건강한 피험자 14명 RRIs data

avg_mcrde_chf_path = './results/avg_mcrde_chf.txt'      # path of CHF subjects MCRDE avg data
avg_mcrde_healthy_path = './results/avg_mcrde_healthy.txt'  # path of HEALTHY subjects MCRDE avg data
std_mcrde_chf_path = './results/std_mcrde_chf.txt'      # path of CHF subjects MCRDE std data
std_mcrde_healthy_path = './results/std_mcrde_healthy.txt'  # path of HEALTHY subjects MCRDE std data

# MCRDE parameters
N = 1000   # RRIs data length
m = 3   # embeding dimension
c = 6   # number of class
tau = 1    # delay factor
scale = 25    # scale factor

# plot parameters
show_fig = True                                            # plot draw flag
subject = np.array(['CHF', 'HEALTHY'])                     # 비교군: CHF, Healthy subject
plt_color = np.array(['red','blue'])                          # CHF MCRDE plot: 빨간색, Healthy MCRDE plot: 파란색

### Sample data

# Load RRIs data(type:numpy array, row:subjects, col:RRIs data of subjects)
# CHF data
RRIs_CHF_data = io.loadmat(RRIs_CHF_path)
RRIs_CHF_1000 = RRIs_CHF_data['RRIs_CHF_1000']              # Load RRIs data of CHF(length=1000)

# Healthy data
RRIs_HEALTHY_data = io.loadmat(RRIs_HEALTHY_path)
RRIs_HEALTHY_1000 = RRIs_HEALTHY_data['RRIs_HEALTHY_1000']  # Load RRIs data of HEALTHY(length=1000)

n_s = len(RRIs_CHF_1000[:,0])                             # number of subject(CHF,HEALTHY)

### output data

# MCRDE about RRIs datas of CHF subjects
mcrde_chf = np.zeros((n_s, scale))                       # row: subject, col: scale
# MCRDE about RRIs datas of Healthy subjects
mcrde_healthy = np.zeros((n_s, scale))                       # row: subject, col: scale
         

### Calculate MultiScale Entropy, p-value(Wilcoxon rank sum test)
for i in range(n_s):
    
    # Calculate MCRDE about RRIs data of CHF subjects
    mcrde_chf[i] = MCRDE(RRIs_CHF_1000[i, :N], m, c, tau, scale)
    # Calculate MCRDE about RRIs data of Healthy subjects
    mcrde_healthy[i] = MCRDE(RRIs_HEALTHY_1000[i, :N], m, c, tau, scale)


# Calculate MCRDE average value
avg_mcrde_chf = np.mean(mcrde_chf, axis=0)
avg_mcrde_healthy = np.mean(mcrde_healthy, axis=0)


# Calculate MCRDE std value
std_mcrde_chf = np.std(mcrde_chf, axis=0)
std_mcrde_healthy = np.std(mcrde_healthy, axis=0)


# Calculate p-value(between CHF and Healthy subjects)
# p-value<0.05은 두 데이터(CHF, Healthy)가 서로 독립적인(=유의미한) 데이터라는 것을 보여준다. 
pCHF_HT = np.zeros(scale)              # p-value between CHF and Healthy subjects(scale 1~25)
for i in range(scale):
    s, pCHF_HT[i] = ranksums(mcrde_chf[:,i],mcrde_healthy[:,i])



### MCRDE 평균, 표준편차 출력 파일을 text 파일에 저장한다.
# CHF, Healthy subjects는 각각 모두 14명이다.
# MultiScale Entropy method를 통해 scale(1~25)에서의 14명의 Entropy 평균, 표준 편차를 구한다. 
# 사용한 Entropy 기법은 MultiScale Cumulative Residual Dispersion Entropy이다.
# index: scale, data: avg, std

write_txt(avg_mcrde_chf_path, avg_mcrde_healthy_path,
          std_mcrde_chf_path, std_mcrde_healthy_path,
          avg_mcrde_chf, avg_mcrde_healthy,
          std_mcrde_chf, std_mcrde_healthy )

# show the result figure
if show_fig == True:
    Entropy = plot_Entropy(subject,plt_color,
                            avg_mcrde_chf,
                            avg_mcrde_healthy,
                            std_mcrde_chf,
                            std_mcrde_healthy,
                            scale,
                            pCHF_HT)
    







    





    
    

    





