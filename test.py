# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# plt.rc('font', family='Times New Roman', size=12)
#
# plt.rcParams['xtick.direction'] = 'in'
# plt.rcParams['ytick.direction'] = 'in'
#
# x=[0.88496,7,3.149,0.9823,3.307,5.8407,0.83185]
# y=[81.3,81.2,81.4,82.7,81.3,82.7,83.1]
# label=['ABCNet','BANet','MANet','UnetFormer','MAResUnet','DCSwin','MyNet']
# # for i in range(len(rdata[0])):
# #     x_value.append(rdata[0][i].split(',')[0])
# #     y_value.append(rdata[0][i].split(',')[1])
#
# plt.grid(linestyle='dashed')
# for i in (1,3,5):
#     plt.annotate(label[i], (x[i]+0.1, y[i]-0.02))
#
# plt.annotate('ABCNet', (x[0]-0.3, y[0]+0.04))
# plt.annotate('MANet', (x[2]+0.1, y[2]+0.02))
# plt.annotate('MAResUNet', (x[4]+0.1, y[4]-0.08))
# plt.annotate('Ours', (x[6]+0.2, y[6]-0.02), weight='heavy')
#
# plt.scatter(x, y, c='#a0c0f0',s=80)
#
# plt.scatter(x[6], y[6], c='#ffba55', marker="*", s=310)
#
# plt.xlabel("Inference Time (s)", size=15)
# plt.ylabel("mIoU (%)", size=15)
# plt.xlim(0, 8)
# plt.ylim(81, 83.5)
#
# plt.savefig('C:/Users/Administrator/Desktop/a.png', dpi=600, format="png")
# # plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', family='Times New Roman', size=12)  # 字体字号

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

x=[15.63,13.06,22.2,11.68,25.42,70.15,17.39]
y=[81.3,81.2,81.4,82.7,81.3,82.7,83.1]
label=['ABCNet','BANet','MANet','UnetFormer','MAResUnet','DCSwin','Ours']
# for i in range(len(rdata[0])):
#     x_value.append(rdata[0][i].split(',')[0])
#     y_value.append(rdata[0][i].split(',')[1])

plt.grid(linestyle='dashed')

plt.annotate('UnetFormer', (x[3]+1.3, y[3]-0.02))

plt.annotate('ABCNet', (x[0]-6, y[0]+0.08))
plt.annotate('BANet', (x[1]-4, y[1]-0.13))
plt.annotate('MANet', (x[2]+0.3, y[2]+0.07))
plt.annotate('MAResUNet', (x[4]-6, y[4]-0.15))
plt.annotate('DCSwin', (x[5]-1, y[5]+0.07))

plt.annotate('Ours', (x[6]+1.3, y[6]-0.02), weight='heavy')
# plt.annotate('MyNet', (x[7]+0.2, y[7]-0.02))

plt.scatter(x, y, c='#a0c0f0',s=90)  # 点的颜色 大小

plt.scatter(x[6], y[6], c='#ffba55', marker="*", s=310)  # 点的颜色、形状、大小

plt.xlabel("FLOPs (G)", size=15)
plt.ylabel("mIoU (%)", size=15)
plt.xlim(0, 80)
plt.ylim(81, 83.5)

plt.savefig('C:/Users/Administrator/Desktop/FLOPs.png', dpi=600, format="png")
# plt.show()