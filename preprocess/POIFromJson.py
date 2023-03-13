import json
import numpy as np
url = 'data/yelp_academic_dataset_business.json'

# min_lat = 40.1992655
# max_lat = 40.2392655
# min_lng = -74.7533059
# max_lng = -74.7233059
# -74.02974864189738,40.59011660290622
# -73.7643318314063,40.77942344008019

# -74.07806838200426,40.66344210179837
min_lat = 39.88625598218755
max_lat = 39.9504334672952
min_lng = -75.21510240324996
max_lng = -75.14126430508557
size = 16
lat_step = (max_lat-min_lat)/size
lng_step = (max_lng-min_lng)/size
poi_map = np.zeros((size, size))



def indexLat(lat):
    return int((lat-min_lat)/lat_step)


def indexLng(lng):
    return int((lng-min_lng)/lng_step)


# print(lng_step, lat_step)

count = 1
count_exist = 0
with open(url, 'r') as f:
    line = f.readline()
    while line:
        # print(line)
        business = json.loads(line)
        # print(business)
        lat = business['latitude']
        lng = business['longitude']
        # print(type(lat))
        # print(lng, lat)
        if lng > min_lng and lng < max_lng and lat < max_lat and lat > min_lat:
            # print(business)
            poi_map[size - 1 - indexLat(lat)][indexLng(lng)] += 1
            count_exist += 1
            # break
        line = f.readline()
        if count%10000==0:
            print(count)
        count += 1
    print(count_exist)

print(poi_map)

    # for key,value in new_dict:
    #     if
    #         {"business_id": "Pns2l4eNsfO8kk83dixA6A", "name": "Abby Rappoport, LAC, CMQ",
    #          "address": "1616 Chapala St, Ste 2", "city": "Santa Barbara", "state": "CA", "postal_code": "93101",
    #          "latitude": 34.4266787, "longitude": -119.7111968, "stars": 5.0, "review_count": 7, "is_open": 0,
    #          "attributes": {"ByAppointmentOnly": "True"},
    #          "categories": "Doctors, Traditional Chinese Medicine, Naturopathic\/Holistic, Acupuncture, Health & Medical, Nutritionists",
    #          "hours": null}
