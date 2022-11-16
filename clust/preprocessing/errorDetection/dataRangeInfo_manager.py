class MinMaxLimitValueSet():
    
    def __init__(self):
        pass

    def get_data_min_max_limitSet(self, type):
        if type =='air':
            #data_clean_partial =  edd.extream_error_deletion(data_raw_partial)
            data_min_max_limit = {'max_num':{"Humidity" : 100,"Temperature" : 100, 'CO2ppm':10000, 'H2Sppm':100, 'NH3ppm':300, 'OptimalTemperature':45, 
                                        'RichTemperature':45, 'RichHumidity':100, 'comp_temp':45, 'comp_humid':100,
                                        '2ndPanSpeed':200, 'UpperPanSpeed':200, 'out_pressure':2000,
                                        "in_pm01_raw":1000, "in_pm25_raw":1000, "in_pm10_raw":1000,
                                        "temp":80, "in_humi":100, "co2":99999, "voc":99999,  # in_temp to temp
                                        'in_temp':100,'in_cici_humi':100,'in_cici_noise':100, 'in_cici_temp':100, 'in_cici_voc':100,
                                        'in_voc':99999,
                                        "in_noise":5000, "in_pm10":99999, "in_pm25":99999, "in_pm01":99999, 'pm10':9999, "in_co2":99999,
                                        "cici_pm10":100, "cici_pm25":100, "cici_temp":100, "cici_humi":100, 
                                        "cici_noise":100, "cici":100, "ciai":100, 'cici_voc':100, 'cici_co2':100,
                                        "out_pm25_raw":1000, "out_pm10_raw":1000, "out_temp":80, "out_humi":100, "out_noise":90,
                                        "out_pm10":1000, "out_pm25":1000, "ultraviolet_rays":16, "light_intensity":120000, "blacksphere_temp":80,
                                        "coci_pm10":100, "coci_pm25":100, "coci_temp":100, "coci_humi":100, "coci":100, "coai":100,
                                        "nh3":100, "h2s":100, "o3":2, "co":12.5, "no2":2, "so2":2, "vibration_x":16, "vibration_y":16, 
                                        "vibration_z":16, "vibration_max_x":16, "vibration_max_y":16, "vibration_max_z":16},
                           'min_num':{"Humidity" : 0, "Temperature" : -100,'CO2ppm':0, 'H2Sppm':0, 'NH3ppm':0, 'OptimalTemperature':-20, 'RichTemperature':-20, 
                                        'RichHumidity':0, 'comp_temp':-20, 'comp_humid':0, '2ndPanSpeed':0, 'UpperPanSpeed':0, 'out_pressure':0, 
                                        "in_pm01_raw":0, "in_pm25_raw":0, "in_pm10_raw":0,
                                        "temp":-40, "in_humi":0, "co2":0, "voc":0,  #in_temp to temp
                                        'in_temp':-100,'in_cici_humi': 0, 'in_cici_noise':0,'in_cici_temp':0, 'in_cici_voc':0,
                                        'in_voc':0,
                                        "in_noise":0, "in_pm10":0, "in_pm25":0, "in_pm01":0, "pm10":0, "in_co2":0,
                                        "cici_pm10":0, "cici_pm25":0, "cici_temp":0, "cici_humi":0, "in_cici_co2":0, 
                                        "cici_noise":0, "cici":0, "ciai":0, 'cici_voc':0, 'cici_co2':0,
                                        "out_pm25_raw":0, "out_pm10_raw":0, "out_temp":-40, "out_humi":0, "out_noise":35,
                                        "out_pm10":0, "out_pm25":0, "ultraviolet_rays":0, "light_intensity":0, "blacksphere_temp":-40,
                                        "coci_pm10":0, "coci_pm25":0, "coci_temp":0, "coci_humi":0, "coci":0, "coai":0,
                                        "nh3":0, "h2s":0, "o3":0, "co":0, "no2":0, "so2":0, "vibration_x":0, "vibration_y":0, 
                                        "vibration_z":0, "vibration_max_x":0, "vibration_max_y":0, "vibration_max_z":0}}
        
        return data_min_max_limit