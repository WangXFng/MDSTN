#!/usr/bin/env python
# -*- coding:utf-8 -*-

from urllib.parse import quote
from urllib import request
import json
import xlwt

#TODO 替换为申请的密钥
amap_web_key = '你的密钥'

poi_search_url = "http://restapi.amap.com/v3/place/text"
poi_boundary_url = "https://ditu.amap.com/detail/get/detail"
from transCoordinateSystem import gcj02_to_wgs84

#TODO cityname为需要爬取的POI所属的城市名，city_areas为城市下面的行政区，classes为多个POI分类名的集合.
# (中文名或者代码都可以，代码详见高德地图的POI分类编码表)
cityname = '武汉'
city_areas = ['江岸区','江岸区','汉南区']
classes = [ '药房','商城']


# 根据城市名称和分类关键字获取poi数据
def getpois(cityname, keywords):
    i = 1
    poilist = []
    while True:  # 使用while循环不断分页获取数据
        url = 'https://uri.amap.com/nearby?key=6d0a9077cb31f6a3dbb9869fe5237a38&location=-73.99451258597276,40.72494529701557&service=movie'
        result = getpoi_page(cityname, keywords, i)
        print(result)
        result = json.loads(result)  # 将字符串转换为json
        if result['count'] == '0':
            break
        hand(poilist, result)
        i = i + 1
    return poilist


# 数据写入excel
def write_to_excel(poilist, cityname, classfield):
    # 一个Workbook对象，这就相当于创建了一个Excel文件
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet(classfield, cell_overwrite_ok=True)

    # 第一行(列标题)
    sheet.write(0, 0, 'x')
    sheet.write(0, 1, 'y')
    sheet.write(0, 2, 'count')
    sheet.write(0, 3, 'name')
    sheet.write(0, 4, 'address')
    sheet.write(0, 5, 'adname')


    for i in range(len(poilist)):
        location = poilist[i]['location']
        name = poilist[i]['name']
        address = poilist[i]['address']
        adname = poilist[i]['adname']
        lng = str(location).split(",")[0]
        lat = str(location).split(",")[1]

        #坐标转换
        result = gcj02_to_wgs84(float(lng), float(lat))
        lng = result[0]
        lat = result[1]


        # 每一行写入
        sheet.write(i + 1, 0, lng)
        sheet.write(i + 1, 1, lat)
        sheet.write(i + 1, 2, 1)
        sheet.write(i + 1, 3, name)
        sheet.write(i + 1, 4, address)
        sheet.write(i + 1, 5, adname)

    # 最后，将以上操作保存到指定的Excel文件中
    book.save(r'' + cityname + "_" + classfield + '.xls')