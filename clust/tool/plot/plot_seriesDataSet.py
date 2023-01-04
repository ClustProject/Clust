# TODO JW 언제 쓰이는지 잘 모르겠으나, 제가 나중에 수정하도록 하겠습니다.


import matplotlib.pyplot as plt
import numpy as np
def groupSeriesFig(data, title, fig_width, fig_height, fig_width_num = 4):
    id_num = len(data)
    fig_height_num = int(np.ceil(id_num/fig_width_num))

    fig, axs = plt.subplots(fig_height_num,fig_width_num,figsize=(fig_width, fig_height))
    fig.suptitle(title)
    for i in range(fig_height_num):
        for j in range(fig_width_num):
            if i*fig_width_num+j+1>id_num: # pass the others that we can't fill
                continue
            axs[i, j].plot(data[i*fig_width_num+j].values)
            axs[i, j].set_title(seriesName[i*fig_width_num+j])
    return plt


def show_all_column_data(data, fig_width, fig_height, fig_width_num = 4):
    column_num = len(data.columns)
    fig_height_num = int(np.ceil(column_num/fig_width_num))
    ax = data.plot(subplots=True, layout=(fig_height,fig_width), figsize=(fig_width, fig_height)).bach()
    plt = ax.get_figure()

    return plt

