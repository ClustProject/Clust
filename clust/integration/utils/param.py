
def makeIntDataInfoSet(dataInfo, start, end):
    intDataInfo ={}
    db_info=[]

    for db_index, db_one in enumerate(dataInfo):
        db_set={}
        db_set['start'] = start
        db_set['end'] = end
        db_set['db_name'] = db_one[0]
        db_set['measurement'] = db_one[1]
        if len(db_one)>2:
            db_set ['tag_key']= db_one[2]
            db_set ['tag_value']= db_one[3]
        db_info.append(db_set)

    intDataInfo["db_info"] = db_info
    return intDataInfo


