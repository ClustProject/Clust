
import plotly.graph_objects as go
import chart_studio.plotly as py
from plotly.offline import *

def plot_predictions(df_result):
    """
    Args:
        df_result (dataFrame): input data 
        
    Example:
    
        >>> baseline = go.Scatter(
        ...                     x=df_baseline.index,
        ...                     y=df_baseline.prediction,
        ...                     mode="lines",
        ...                     line={"dash": "dot"},
        ...             name='linear regression',
        ...             marker=dict(),
        ...             text=df_baseline.index,
        ...             opacity=0.8,
        ...             )
        >>> data.append(baseline)

    """
    data = []
    columns = df_result.columns
    data0 = go.Scatter(
        x=df_result.index,
        y=df_result[columns[0]],
        mode="lines",
        name=columns[0],
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )
    data.append(data0)

    data1 = go.Scatter(
        x=df_result.index,
        y=df_result[columns[1]],
        mode="lines",
        line={"dash": "dot"},
        name=columns[1],
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )
    data.append(data1)
    
    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)
    iplot(fig)
    