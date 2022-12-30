def save_model_byTsLearn(file_name, model):
    # Save model
    with open(file_name, 'wb') as f:
        model.to_pickle(file_name)
        
def load_model_byTsLearn(file_name, model):
    loaded_model = model.from_pickle(file_name)
    return loaded_model