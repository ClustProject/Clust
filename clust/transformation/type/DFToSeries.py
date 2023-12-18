def DFSetToSeries(dataSet):
    seriesData =[]
    for i in range(len(dataSet)):
        value = dataSet[i].values
        seriesData.append(value.reshape(len(value)))
    return seriesData