import csv
import numpy as np
import Constant

min_lat = Constant.min_lat
max_lat = Constant.max_lat
min_lng = Constant.min_lng
max_lng = Constant.max_lng

size = Constant.size

lat_step = Constant.lat_step
lng_step = Constant.lng_step


import matplotlib.pyplot as plt
# import seaborn as sns
import time

min_time = 1640880000  # 1月 1640880000 2月1643644800
length = 24*60*(31+28+31)/30  # 129600
map_ = np.zeros((int(length), 16, 16))

# 将时间字符串转换为10位时间戳，时间字符串默认为2017-10-01 13:37:04格式
# 2022-01-18 08:23:52


def date_to_timestamp(date, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.strptime(date, format_string)
    time_stamp = int(time.mktime(time_array))
    return time_stamp


# start_stamp = date_to_timestamp('2022-01-01 00:00:00')


def indexLat(lat):
    return int((lat-min_lat)/lat_step)


def indexLng(lng):
    return int((lng-min_lng)/lng_step)


def load_(filename, isStart=True):
    print(filename)
    with open(filename) as f:
        reader = csv.reader(f)
        head_row = next(reader)
        # print(head_row)
        # for index,column_header in enumerate(head_row):
        #     print(index,column_header)
        line = next(reader)
        count = 1
        while line:
            ride_id, rideable_type, started_at, ended_at, start_station_name, start_station_id, end_station_name, \
            end_station_id, start_lat, start_lng, end_lat, end_lng, member_casual = line

            if count % 10000 == 0:
                print(count)
            count += 1
            time_ = date_to_timestamp(started_at)
            if isStart:
                if start_lat != '':
                    index = int((time_ - min_time) / (60 * 30))
                    if index>=length or index<0:
                        line = next(reader)
                        continue
                    start_lat, start_lng = float(start_lat), float(start_lng)
                    if start_lng > min_lng and start_lng < max_lng and start_lat < max_lat and start_lat > min_lat:
                        map_[index][size-1-indexLat(start_lat)][indexLng(start_lng)] += 1
            else:
                if end_lat != '':
                    index = int((time_ - min_time) / (60 * 30))
                    if index>=length or index<0:
                        line = next(reader)
                        continue
                    end_lat, end_lng = float(end_lat), float(end_lng)
                    if end_lng > min_lng and end_lng < max_lng and end_lat < max_lat and end_lat > min_lat:
                        map_[index][size - 1 - indexLat(end_lat)][indexLng(end_lng)] += 1

            try:
                line = next(reader)
            except Exception as e:
                print(e)
                break
            # finally:
            #     print()
        print(count)

print(1052419 +1233715 +1893445)
# load_("data/202201-citibike-tripdata.csv") # 1052419 1233715 1893445
# load_("data/202202-citibike-tripdata.csv")
# load_("data/202203-citibike-tripdata.csv")
# np.save('start_map_Philadelphia.npy', map_)

# map_ = np.zeros((int(length), 16, 16))
# load_("data/202201-citibike-tripdata.csv", False)
# load_("data/202202-citibike-tripdata.csv", False)
# load_("data/202203-citibike-tripdata.csv", False)
# np.save('end_map_Philadelphia.npy', map_)