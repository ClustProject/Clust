# google map
GENERATOR_INFO={
        "MAP_API_KEY" : "AIzaSyD_A4WM84Z-53IbbfMz1orStEHNqYDSFHo"
}
##### -------------------------- Azure Server ----------------------------------------
azureServer = "52.231.185.8" # Azure Public IP
# azureServer = "192.168.193.246" # VPN
# wiz_url = f'http://{azureServer}:5000'
# wiz_url = f'http://{azureServer}:20002'
wiz_url = "http://192.168.193.246:20002"
# # mongoDB
# CLUSTMetaInfo={
#         "USER_ID":"keti",
#         "USER_PWD":"ketiabcs",
#         "HOST_ADDR":azureServer,
#         "HOST_PORT":"27170",
#         "DB_NAME":"exploration"
# }
# AlaradMetaInfo={
#         "USER_ID":"alarad",
#         "USER_PWD":"alarad1",
#         "HOST_ADDR":azureServer,
#         "HOST_PORT":"27170",
#         "DB_NAME":"exploration"
# }
CLUSTDataServer={
        "url":f'http://{azureServer}:8086/',
        "token":"PWEQHIKUO0rrdf4IrqQC8Z3zi8rKr7vHQwlosGpvfgt8-eSyvjiv2p3_FQP30E7o0BDm61qZuLBpfl3mrsPNbw==",
        "org":"clust"
}
AlaradDataServer={
        "url":f'http://{azureServer}:8086/',
        "token":"kkz74XIA3DrooM_Rdjlzyiqv3lsFQJpMFD_-WXa7Ax64HuBJCyZiSIHp1REZB4iYOjvQWb8vbUGaNAOZauybdg==",
        "org":"clust"
}
VibeDataServer={
        "url":f'http://{azureServer}:8086/',
        "token":"9nPfYPYfH3CglBSHnEJcmzRIdA7RNJ7oq3SNiufuc-Sze8u5kjf0jpmSNlgNEdCM-9B2kwjHmEgd_59475Rr5w==",
        "org":"clust"
}
##### -------------------------- Sangam(WorkStation) ----------------------------------------
sangamServer = "10.252.107.59" # 내부망
#sangamServer = "192.168.193.48" # VPN 
wiz_url = f'http://{sangamServer}:5000'
# wiz_url = f'http://{sangamServer}:20002'
# mongoDB
CLUSTMetaInfo={
        "USER_ID":"keti",
        "USER_PWD":"ketiabcs",
        "HOST_ADDR":sangamServer,
        "HOST_PORT":"27170",
        "DB_NAME":"exploration"
}
AlaradMetaInfo={
        "USER_ID":"alarad",
        "USER_PWD":"alarad1",
        "HOST_ADDR":sangamServer,
        "HOST_PORT":"27170",
        "DB_NAME":"exploration"
}
# influxDB 2.0
CLUSTDataServer2={
        "url":f'http://{sangamServer}:8086/',
        "token":"n3rvk6b6yrk-CwPz9MrGUXFv44aCOiIMebemiPxmuRIPQv6D5Ehw7FTYLO45Fu-6qirKpZy84T6brgxfFmc2xA==",
        "org":"clust"
}
AlaradDataServer2={
        "url":f'http://{sangamServer}:8086/',
        "token":"MIPBhdhO6fr-HZPa-4w2kDJJvlgo4RDn2_u8tHRxFqZXyLxbWXys0c-jas2sQPg8WyQVpBjvlQk0dsNw4OvLkg==",
        "org":"clust"
}
VibeDataServer2={
        "url":f'http://{sangamServer}:8086/',
        "token":"G1mcrDOjxXgY_EEqtzVCOMr6tOF8baGiSbZrDtrh9W4NwF6P5LseC-G3g5180F5W94cPBcXEMkM4z0EmbyTM3A==",
        "org":"clust"
}
connection_config_path = '../../../KETIPreDataIngestion/KETI_setting/connection_config.json'