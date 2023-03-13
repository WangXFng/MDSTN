# import re
# from urllib import request
# import json
#
# class Spider():  # 私有方法，模拟请求，获取html内容
#     url = 'https://blog.csdn.net/phoenix/web/blog/weekly-rank?page=0&pageSize=100'
#
#     def __fetch_content(self):
#         r = request.urlopen(Spider.url)
#         html = r.read()  # 这里保存的是字节流文件
#         html = str(html,encoding='UTF-8')  # 全部转换成字符串
#         # print(html)  # 输出结果测试
#         return html
#
#
# #     def __analysis(self,html):
# #         root_pattern = '\{([\s\S]*?)\}'
# #         name_pattern = '"nickName":"([\s\S)]*?)"'
# #         currentRank_pattern = '"currentRank":"([\s\S)]*?)"'
# #         root_html = re.findall(root_pattern, html)
# #         #print(root_html)
# #         anchors = []
# #         for html in root_html:
# #             name= re.findall(name_pattern, html)
# #             currentRank= re.findall(currentRank_pattern, html)
# #             if name:
# #                 anchor = {'name': name,'currentRank': currentRank}
# #                 anchors.append(anchor)
# #         print(anchors)
# #         return anchors
# #
# #     def go(self):  # 公开的入口方法
# #         html = self.__fetch_content()  # 调用
# #         html = self.__analysis(html)
#
#
#     def parser(self):
#         html = self.__fetch_content()  # 调用
#         contents = json.loads(html)
#         for item in contents['data']['weeklyRankListItem']:
#             print('姓名：{}, 排名：{}'.format(item['nickName'], item['currentRank']))
#
#
# spider = Spider()
# # spider.go()
# spider.parser()
#
# a = 'false'
# if 'false' == a:
#     print(True)
#
# print(type(a))
# print(type(True))

from urllib import request
import re

class Spider():  # 私有方法，模拟请求，获取html内容
    url = 'https://www.runoob.com/python/att-string-format.html'

    def __fetch_content(self):
        r = request.urlopen(Spider.url)
        html = r.read()  # 这里保存的是字节流文件
        html = str(html,encoding='UTF-8')  # 全部转换成字符串
        #print(html)  # 输出结果测试
        return html
    codes = []
    def __analysis(self, html):
        root_pattern = '<div class="hl-main">[\s\S]*?</div>'
        code_pattern = '<span class="hl-string">([\s\S]*?)</span>'
        root_html = re.findall(root_pattern, html)
        print(root_html)
        for html in root_html:
            codenumber = re.findall(code_pattern, str(root_html))
            print(codenumber)
            #code = {'code': codenumber}
            #print(code)
            #codes.append(code)
        #return codes
    def go(self):
        html = self.__fetch_content()
        html = self.__analysis(html)
spider = Spider()
spider.go()

