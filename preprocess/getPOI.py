

# # 保存excel文件
from datetime import datetime
import json
import xlwt
import urllib.request
import ssl
import pymysql

#  113.719,34.70;113.751,34.74
min_lat = 40.708810121617036
max_lat = 40.75811792366235
min_lng = -74.00690286635771
max_lng = -73.97481545284578


def excel_(result):
    wbk = xlwt.Workbook()
    # 新建一个名为Sheet1的excel sheet。此处的cell_overwrite_ok =True是为了能对同一个单元格重复操作。
    sheet = wbk.add_sheet('Sheet1',cell_overwrite_ok=True)
    # 获取当前日期，得到一个datetime对象如：(2016, 8, 9, 23, 12, 23, 424000)
    today = datetime.today()
    # 将获取到的datetime对象仅取日期如：2016-8-9
    today_date = datetime.date(today)
    
    sheet.write(0,0,'pcode')
    sheet.write(0,1,'name')
    sheet.write(0,2,'type')
    sheet.write(0,3,'location')
    sheet.write(0,4,'type_sum')
    
    # 遍历result中的没个元素。
    for i in range(len(result)):
        #对result的每个子元素作遍历，
        sheet.write(i+1,0,result[i]['typecode'])
        sheet.write(i+1,1,result[i]['name'])
        sheet.write(i+1,2,result[i]['type'])
        sheet.write(i+1,3,result[i]['location'])
        sheet.write(i+1,4,result[i]['typecode'][0:2])
    # 以传递的name+当前日期作为excel名称保存。
    wbk.save('{},{};{},{}.xls'.format(min_lng, min_lat, max_lng, max_lat))
    



keyword = ['购物','餐饮','生活','体育','医疗','住宿','风景','政府机构','科教','交通设施','金融保险','公司企业','汽车销售','汽车服务'] # 

global result
result = []

def getDataByPageAndKeyword(page,keyword):
    url = 'http://restapi.amap.com/v3/place/polygon?&offset=50&key=6d0a9077cb31f6a3dbb9869fe5237a38&extensions=base&polygon={},{};{},{}&keywords={}&page={}'.format(min_lng, min_lat, max_lng, max_lat, urllib.parse.quote(keyword), str(page+1))
    # print(url)
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url)
    response = urllib.request.urlopen(url=request,context=context)
    data = response.read().decode('utf-8')
    # print(data)
    return json.loads(data)

def getPOI(keyword):
    data = getDataByPageAndKeyword(0,keyword)
    count = int(data['count'])
    for i in range(int(count/50)+1):# #113.719,34.70;113.751,34.74   113.687,34.70;113.719,34.74 
        data = getDataByPageAndKeyword(i,keyword)
        result.extend(data['pois'])


for i in range(len(keyword)):
    getPOI(keyword[i])

excel_(result)
#
# # print()
# # data1
# # 113.719,34.70;113.751,34.74
# now = 'data2'
#
# import numpy as np
# min_lng = 113.719 if now == 'data1' else 113.687
# max_lng = 113.751 if now == 'data1' else 113.719
# min_lat = 34.70
# max_lat = 34.74
# step = 16
#
# lng_step = (max_lng - min_lng)/step
# lat_step = (max_lat - min_lat)/step
#
# # data2
# # 113.687,34.70;113.719,34.74
#
# def checkRecordLocation(groupData):
#     submap = np.zeros((step, step))
#     for i in groupData:
#         coordinate = i[2].split(',')
#         lng = float(coordinate[0])
#         lat = float(coordinate[1])
#         lng_index = 0
#         lat_index = 0
#         for i in range(16):
#             if(lng<min_lng+(i+1)*lng_step):
#                 lng_index = i
#                 break
#         for i in range(16):
#             if(lat<min_lat+(i+1)*lat_step):
#                 lat_index = i
#                 break
#         submap[lng_index][lat_index] += 1
#     return submap
#
#
#
# db = pymysql.connect(host='127.0.0.1', port=3306, user='root', passwd='root', db='trace', charset='utf8')
# cursor = db.cursor()
#
# map_ = np.zeros((20, step, step))
#
# for i in range(20):
#
#     padded_index = '0'+str(i) if i<10 else str(i)
#     # print(str(i[0])+" "+i[1])
#     # code, name, location, type
#     cursor.execute("select * from {0} where code = {1}".format(now,padded_index))
#     groupData = cursor.fetchall()
#
#     submap = checkRecordLocation(groupData)
#     print(submap)
#     map_[i] = submap
#
# print(map_.shape)
#
# np.save('N2D/data2/dataPOI.npy', map_)

