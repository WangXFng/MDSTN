import csv
import numpy as np
import re
import Constant
import time


min_lat = Constant.min_lat
max_lat = Constant.max_lat
min_lng = Constant.min_lng
max_lng = Constant.max_lng
size = Constant.size
lat_step = Constant.lat_step
lng_step = Constant.lng_step

poi_map = np.zeros((13, 16, 16))

p1 = re.compile(r"[(](.*?)[)]", re.S)  # 最小匹配


def date_to_timestamp(date, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.strptime(date, format_string)
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def indexLat(lat):
    return int((lat-min_lat)/lat_step)


def indexLng(lng):
    return int((lng-min_lng)/lng_step)


count = 1
with open('data/Point_Of_Interest.csv', 'r') as f:
    f_csv = csv.reader(f)
    header = f_csv.__next__()
    print(header)
    row = next(f_csv)
    while row:
        lng, lat = str(re.findall(p1, row[0])[0]).split(" ")
        lng, lat = float(lng), float(lat)
        index = int(row[11])-1
        if lng > min_lng and lng < max_lng and lat < max_lat and lat > min_lat:
            poi_map[index][size - 1 - indexLat(lat)][indexLng(lng)] += 1
        # print(row[11], index, lng, lat)

        try:
            row = next(f_csv)
        except Exception as e:
            print(e)
            break
        # row = next(f_csv)
        if count % 10000 == 0:
            print(count)
        count += 1

        # break
    # start_lat, start_lng = float(start_lat), float(start_lng)

np.save('poi_map', poi_map)
print(sum(poi_map, 0))
print(poi_map)
print(count)