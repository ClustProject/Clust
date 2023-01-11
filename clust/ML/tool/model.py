import pickle

## Pickle
def save_pickle_model(model, model_file_path):
    """ save model: dafult file type is pickle
    
    Args:
        model_file_path(str) : model_file_path
    
    """
    with open(model_file_path, 'wb') as outfile:
        pickle.dump(model, outfile)

def load_pickle_model(model_file_path):
    """ load model: : dafult file type is pickle
    Args:
        model_file_path(str) : model_file_path
    
    """
    with open(model_file_path, 'rb') as infile:
        model = pickle.load(infile)

    return model
