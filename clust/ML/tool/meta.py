def save_model_meta_data(mongodb_client, model_meta):
    db_name = 'model'
    collection_name = 'meta'
    mongodb_client.insert_document(db_name, collection_name, model_meta)
    print(model_meta)


def get_model_meta_data(mongodb_client, search):
    db_name = 'model'
    collection_name = 'meta'
    meta = mongodb_client.get_document_by_json(db_name, collection_name, search)
    model_meta = meta[0]

    return model_meta