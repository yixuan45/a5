# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class complexGraphProject(object):
    def __init__(self):
        self.line_dict = {}  # 线路字典
        self.lineIndex_dict = {}  # 线路赋值字典
        self.stationMatrix = []  # 地铁邻接矩阵

    def topological_graph(self, lineDict=None, lineIndexDict=None, stationMatrix=None):
        """
        绘制拓扑图
        :param lineDict: 线路字典
        :param lineIndexDict: 地铁线路赋值字典
        :param stationMatrix: 地铁邻接矩阵
        :return:
        """
        n = len(lineIndexDict)

        # 使用networkx绘制拓扑图
        G = nx.Graph()

        # 添加边到图
        for i in range(n):
            for j in range(n):
                if stationMatrix[i][j] == 1:
                    G.add_edge(list(lineIndexDict.keys())[i], list(lineIndexDict.keys())[j])

        # 使用spring布局，增加节点之间的间距
        pos = nx.spring_layout(G, k=1.5, iterations=1500)  # k值增大以增加节点之间的距离
        # 图2 k=3,iterations=2000
        # 图3

        # 增大图的尺寸
        plt.figure(figsize=(18, 12))  # 增大尺寸

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', edgecolors='black', alpha=0.9)

        # 绘制边
        lines = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown', 'pink']  # 线路颜色
        styles = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed', 'dotted', 'dashdot', 'solid',
                  'dashed']  # 线路样式

        for index, line in enumerate(lineDict.keys()):
            subnodes = lineDict[line]
            for i in range(len(subnodes)):
                for j in range(i + 1, len(subnodes)):
                    if G.has_edge(subnodes[i], subnodes[j]):
                        nx.draw_networkx_edges(G, pos, edgelist=[(subnodes[i], subnodes[j])],
                                               width=2, alpha=0.5, edge_color=lines[index % len(lines)],
                                               style=styles[index % len(styles)])

                        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

        # 设置标题和背景色
        plt.title('分散型地铁线路图', fontsize=20)
        plt.gca().set_facecolor('whitesmoke')  # 设置背景色

        # 去掉坐标轴
        plt.axis('off')

        plt.show()

    def adjacent_matrix(self, lineDict=None, lineIndexDict=None):
        """
        绘制邻接矩阵
        :param lineDict: 线路字典
        :param lineIndexDict: 线路赋值字典
        :return:
        """
        rows = len(lineIndexDict)
        stationMatrix = []
        for _ in range(rows):
            row = []  # 新建一行
            for _ in range(rows):
                row.append(0)
            stationMatrix.append(row)
        line_list = [1, 2, 3, 4, 5, 6, 8, 9, 14, 16]  # 存储了每天站线的名字
        for index in line_list:
            lineTotalList = lineDict["line{}".format(index)]
            for station_index, station_name in enumerate(lineTotalList[1:]):
                stationMatrix[lineIndexDict[lineTotalList[station_index - 1]]][lineIndexDict[station_name]] = 1
                stationMatrix[lineIndexDict[station_name]][lineIndexDict[lineTotalList[station_index - 1]]] = 1
        return stationMatrix

    def prepareData(self):
        """准备数据"""
        df = pd.read_csv('./lineData_new.csv', encoding='utf-8')
        line_dict = {'line1': [], 'line2': [], 'line3': [], 'line4': [], 'line5': [], 'line6': [], 'line8': [],
                     'line9': [], 'line14': [], 'line16': []}
        index_dict = {}  # 用于存储每个站点对应的值
        df_unique = df.drop_duplicates()
        line_list = [1, 2, 3, 4, 5, 6, 8, 9, 14, 16]  # 存储了每天站线的名字
        for index in line_list:
            line_dict["line{}".format(index)] = df_unique["line{}".format(index)].tolist()

        for index in line_list:
            lineTotalList = line_dict["line{}".format(index)]
            for index_number, data in enumerate(lineTotalList[::-1]):
                if data == np.nan:
                    line_dict["line{}".format(index)].pop(index_number)

        cleanLine_dict = {
            line: [s for s in stations
                   if not (isinstance(s, float) and str(s) == 'nan')]
            for line, stations in line_dict.items()
        }
        index_number = 1
        for index in line_list:
            lineTotalList = cleanLine_dict["line{}".format(index)]
            for vacation_name in lineTotalList:
                if vacation_name in index_dict.keys():
                    continue
                else:
                    index_dict[vacation_name] = index_number - 1
                    index_number += 1
        return cleanLine_dict, index_dict
        # 返回一个cleanLine_dict

    def run(self):
        """程序入口"""
        self.line_dict, self.lineIndex_dict = self.prepareData()
        print(self.line_dict)
        print(self.lineIndex_dict)
        self.stationMatrix = self.adjacent_matrix(self.line_dict, self.lineIndex_dict)
        self.topological_graph(lineDict=self.line_dict, lineIndexDict=self.lineIndex_dict,
                               stationMatrix=self.stationMatrix)


if __name__ == '__main__':
    cgp = complexGraphProject()
    cgp.run()
