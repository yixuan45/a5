# -*- coding: utf-8 -*-
import requests
from lxml import etree
import pandas as pd
import numpy as np


class spyderDataProject(object):
    def __init__(self):
        self.url1 = "https://xa.bendibao.com/ditie/linemap.shtml"
        self.headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
            "sec-ch-ua": "\"Microsoft Edge\";v=\"129\", \"Not=A?Brand\";v=\"8\", \"Chromium\";v=\"129\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\""
        }

    def url1_get(self, url, headers, *args, **kwargs):
        """获取一级网页的数据"""
        response = requests.get(url, headers=headers)
        response.encoding = "utf-8"
        html = etree.HTML(response.text)
        line_dict = {'line1': [], 'line2': [], 'line3': [], 'line4': [], 'line5': [], 'line6': [], 'line8': [],
                     'line9': [], 'line14': [], 'line16': []}
        line_list = [1, 2, 3, 4, 5, 6, 8, 9, 14, 16]
        index = 0
        # 编写数据解析内容
        data_list = html.xpath('//div[@class="line-list-body"]/div[@class="line-list-station"]')
        for data in data_list:
            lineTotalLine = data.xpath('.//div[@class="station"]')
            for line in lineTotalLine:
                line_dict["line{}".format(line_list[index])].append(line.xpath('string(./a/text())'))
            index += 1
        self.saveData(line_dict=line_dict)

    def saveData(self, line_dict):
        """保存数据，生成excel文件"""
        # 使用 pd.Series 处理不同长度的列
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in line_dict.items()]))
        df.to_excel('lineDataxlsx.xlsx', index=False, encoding='utf-8')
        print("文件保存成功")

    def run(self):
        """程序运行的主程序"""
        self.url1_get(url=self.url1, headers=self.headers)


if __name__ == '__main__':
    spD = spyderDataProject()
    spD.run()
