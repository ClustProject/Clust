import matplotlib.pyplot as plt
def show_clustering_result_by2DPCA(data, label):
    """
    1) getPCA result (n_components = 2)
    
    Args:
            data (numpy.ndarray): original data to be clustered : shape = (data_num, ts_data_len) 
            label (numpy.ndarray): classification (clustering) labels :shpae = (ts_data_len)
            

    """
    print("Dimension: ", data.shape[1], "---->", 2)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2) 
    
    rlt_tsm = label
    rlt_list = list(set(rlt_tsm))
    rlt_list.sort()

    rlt_pca = pca.fit_transform(data)
    for i in rlt_list:
        label_name = "cluster " +str(i)
        clust_i = rlt_pca[rlt_tsm==i]
        plt.scatter(clust_i[:,0],clust_i[:,0],label=label_name)
    plt.legend()
    plt.show()

    return rlt_pca
