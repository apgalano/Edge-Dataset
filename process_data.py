import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

M = pd.read_csv("Edge-Dataset.csv")
cols = M.columns
frame_rate = M['Frame Rate']
confidence = M['Confidence']
NN_size = M['NN size']
Enc_rate = M['Encoding rate']
Enc_delay = M['Encoding']
Net_delay = M['Network']
confidence_avg = pd.DataFrame(0, index=np.arange(confidence.count()), columns=['Confidence'])
detections = pd.DataFrame(0, index=np.arange(confidence.count()), columns=['Confidence'])
delays = 1000 / frame_rate


for i in range(0,confidence.count()):
    temp = confidence[i].strip('[]')
    if temp == '':
        confidence_avg.iloc[i] = 0
    else:
        confidence_avg.iloc[i] = np.sum([float(j) for j in temp.split(',')])
        detections.iloc[i] = len(temp.split(','))


N_values = np.array(list(range(128,609,64)))
Q_values = np.array(list(range(25,101,25)))


frame_rate_results = np.zeros((N_values.shape[0],Q_values.shape[0]))
confidence_results = np.zeros((N_values.shape[0],Q_values.shape[0]))
detection_results = np.zeros((N_values.shape[0],Q_values.shape[0]))

Total_results = []
N = 128
Q = 25

temp1 = []
temp2 = []
temp3 = []
cnt = 0
for fr in frame_rate.iteritems():
    if NN_size.iloc[cnt] == N and Enc_rate.iloc[cnt] == Q:
        temp1.append(frame_rate.iloc[cnt])
        temp2.append(confidence_avg.iloc[cnt])
        temp3.append(detections.iloc[cnt])
    else:
        frame_rate_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp1)
        confidence_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp2)
        detection_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp3)
        Total_results.append([np.mean(temp1), np.mean(temp2)])
        N = NN_size.iloc[cnt]
        Q = Enc_rate.iloc[cnt]
        temp1 = [frame_rate.iloc[cnt]]
        temp2 = [confidence_avg.iloc[cnt]]
        temp3 = [detections.iloc[cnt]]
    cnt += 1
        
frame_rate_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp1)
confidence_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp2)
detection_results[np.where(N_values == N),np.where(Q_values == Q)] = np.mean(temp3)
Total_results.append([np.mean(temp1), np.mean(temp2)])


fig1 = plt.figure()
ax = sns.heatmap(confidence_results, annot=True, annot_kws={"size": 16}, linewidth=0.5, cmap="YlGnBu")
plt.ylabel('NN size' , fontsize=20)
ax.set_yticklabels(N_values, rotation=0, fontsize=16)
plt.ylim([0,8])
ax.set_xticklabels(Q_values, rotation=0, fontsize=16)
plt.xlabel('Encoding Rate (%)', fontsize=20)
plt.tight_layout()
plt.show()
fig1.savefig('confidence_heat.eps', format='eps')


fig2 = plt.figure()
ax = sns.heatmap(frame_rate_results, annot=True, annot_kws={"size": 16}, fmt=".1f", linewidth=0.5, cmap="YlGnBu")
plt.ylabel('NN size' , fontsize=20)
ax.set_yticklabels(N_values, rotation=0, fontsize=16)
plt.ylim([0,8])
ax.set_xticklabels(Q_values, rotation=0, fontsize=16)
plt.xlabel('Encoding Rate (%)', fontsize=20)
plt.tight_layout()
plt.show()
fig2.savefig('frame_rate_heat.eps', format='eps')
        
        
# AP and AR measurements from COCO dataset
AP = np.array([[0.12,0.12,0.12,0.12],[0.38,0.4,0.4,0.4],[0.42,0.44,0.45,0.45],[0.43,0.48,0.5,0.52],[0.41,0.47,0.5,0.52]])
AR = np.array([[0.07,0.08,0.08,0.08],[0.25,0.26,0.26,0.27],[0.28,0.3,0.3,0.31],[0.29,0.32,0.34,0.35],[0.27,0.32,0.33,0.35]])
        
fig3 = plt.figure()
ax = sns.heatmap(AP, annot=True, annot_kws={"size": 16}, linewidth=0.5, cmap="YlGnBu")
plt.ylabel('NN size' , fontsize=20)
ax.set_yticklabels([128,256,320,512,608], rotation=0, fontsize=16)
plt.ylim([0,5])
ax.set_xticklabels(Q_values, rotation=0, fontsize=16)
plt.xlabel('Encoding Rate (%)', fontsize=20)
plt.tight_layout()
plt.show()
fig3.savefig('AP_heat.eps', format='eps')


fig4 = plt.figure()
ax = sns.heatmap(AR, annot=True, annot_kws={"size": 16}, fmt=".2f", linewidth=0.5, cmap="YlGnBu")
plt.ylabel('NN size' , fontsize=20)
ax.set_yticklabels([128,256,320,512,608], rotation=0, fontsize=16)
plt.ylim([0,5])
ax.set_xticklabels(Q_values, rotation=0, fontsize=16)
plt.xlabel('Encoding Rate (%)', fontsize=20)
plt.tight_layout()
plt.show()
fig4.savefig('AR_heat.eps', format='eps')
        
        
        
        
        