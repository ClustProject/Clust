import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot
def Plot_OneCloumnData(df, y_columnName, title):
    """
    This function plots one column data by index

    :param df: input dataframe
    :type df: dataFrame

    :param y_columnName: one of column name to be shown
    :type y_columnNamedf: string

    :param title: Graph title
    :type title: string

    """

    data = []
    value = go.Scatter(
        x=df.index,
        y=df[y_columnName],
        mode="lines",
        name="values",
        marker=dict(),
        text=df.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(value)

    layout = dict(
        title=title,
        xaxis=dict(title="Date", ticklen=5, zeroline=False),
        yaxis=dict(title=y_columnName, ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)  


def plot_all_feature_data(data):
    plot_cols = data.columns
    plot_features = data[plot_cols]
    _ = plot_features.plot(subplots=True)
    plt.legend()
    plt.show()


def plot_all_feature_data_one_pic(data):
    data.plot()
    plt.legend()
    plt.show()
    
def plot_all_feature_data_two_columns(data):

    import math
    fig, axes = plt.subplots(nrows=int(math.ceil(len(data.columns)/2)), ncols=2, dpi=130, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        if i<len(data.columns):
            temp = data[data.columns[i]]
            ax.plot(temp, color='red', linewidth=1)
            ax.set_title(data.columns[i])
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.spines['top'].set_alpha(0)
            ax.tick_params(labelsize=6)
    plt.tight_layout()
 


def show3Dgraph(data, colorData):
    """ 
    Show 3D data with 3 columns from DataFrame

    :param data: input data
    :type data: DataFrame 
    
    :param colorData: an array with N different values. Express a color acoording to each value.
    :type data: numeric array

    example
        >>> output = ExcludeRedundancy().get_result(data)
    """
    print(colorData.dtypes)
    from mpl_toolkits.mplot3d import Axes3D
    # scatter plot
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    dataColumns = data.columns
    ax.scatter(data[dataColumns[0]],data[dataColumns[1]], data[dataColumns[2]],c=colorData,alpha=0.5)
    ax.set_xlabel(dataColumns[0])
    ax.set_ylabel(dataColumns[1])
    ax.set_zlabel(dataColumns[2])
    plt.show()