import matplotlib.pyplot as plt
import numpy as np
def groupSeriesFig(fig_width, fig_height, seriesData, seriesName, title, fig_width_num = 4):
    id_num = len(seriesData)
    fig_height_num = int(np.ceil(id_num/fig_width_num))

    fig, axs = plt.subplots(fig_height_num,fig_width_num,figsize=(fig_width, fig_height))
    fig.suptitle(title)
    for i in range(fig_height_num):
        for j in range(fig_width_num):
            if i*fig_width_num+j+1>id_num: # pass the others that we can't fill
                continue
            axs[i, j].plot(seriesData[i*fig_width_num+j].values)
            axs[i, j].set_title(seriesName[i*fig_width_num+j])
    return plt