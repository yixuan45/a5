# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
from copy import deepcopy
from collections import deque

matplotlib.use('Agg')
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class complexGraphProject(object):
    def __init__(self):
        self.line_dict = {}  # 线路字典
        self.lineIndex_dict = {}  # 线路赋值字典
        self.stationMatrix = []  # 地铁邻接矩阵
        self.k_dict = {}  # 存储节点度的字典
        self.k_division = [0, 0, 0, 0, 0]  # 存储每个度的分布，比如度默认为索引值，从1开始
        self.k_division_acc = []  # 存储度的累计概率

    def topological_graph(self, lineDict=None):
        """
        绘制拓扑图
        :param lineDict: 线路字典
        :param lineIndexDict: 地铁线路赋值字典
        :param stationMatrix: 地铁邻接矩阵
        :return:
        """
        G = nx.Graph()
        edges_by_line = {}
        station_count = {}

        # 数据处理保持不变
        if lineDict is not None:
            for line_name, stations in lineDict.items():
                line_edges = [(stations[i], stations[i + 1]) for i in range(len(stations) - 1)]
                edges_by_line[line_name] = line_edges
                for station in stations:
                    station_count[station] = station_count.get(station, 0) + 1
                G.add_nodes_from(stations)
                G.add_edges_from(line_edges)

        transfer_stations = [s for s, c in station_count.items() if c > 1]
        regular_stations = list(set(G.nodes) - set(transfer_stations))
        pos = nx.spring_layout(G, k=1.0, iterations=500, seed=42, scale=4)

        # 创建画布
        fig = plt.figure(figsize=(20, 20), dpi=100)
        ax = fig.add_subplot(111)

        # 绘制所有边
        line_colors = plt.cm.tab20.colors
        for i, (line_name, edges) in enumerate(edges_by_line.items()):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=line_colors[i % 20],
                width=2,
                alpha=0.6,
                ax=ax
            )

        # 绘制所有节点（基础圆点）
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=regular_stations,
            node_size=200,  # 较小尺寸
            node_color="#1f77b4",  # 标准蓝色
            edgecolors="white",  # 白边
            linewidths=0.8,
            ax=ax
        )

        # 突出显示换乘站
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=transfer_stations,
            node_size=400,  # 较大尺寸
            node_color="#ff7f0e",  # 醒目橙色
            edgecolors="black",
            linewidths=1.2,
            ax=ax
        )

        # 绘制标签（带背景框）
        label_options = {
            "font_size": 9,
            "font_weight": "bold",
            "font_family": "sans-serif",
            "verticalalignment": "center",
            "horizontalalignment": "center",
            "bbox": {
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.8,
                "boxstyle": "round,pad=0.2"
            }
        }
        nx.draw_networkx_labels(G, pos, ax=ax, **label_options)

        # 创建自定义图例
        legend_elements = [
            Line2D([0], [0],
                   marker='o', color='w', label='Regular Station',
                   markerfacecolor='#1f77b4', markersize=10,
                   markeredgecolor='white'),
            Line2D([0], [0],
                   marker='o', color='w', label='Transfer Station',
                   markerfacecolor='#ff7f0e', markersize=15,
                   markeredgecolor='black'),
            *[Line2D([0], [0], color=c, lw=3, label=l)
              for l, c in zip(edges_by_line.keys(), line_colors)]
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            title="Legend:",
            title_fontsize=12,
            frameon=True,
            framealpha=0.9
        )

        plt.savefig("metro_topology.png", bbox_inches='tight', dpi=150)
        plt.close(fig)

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

    def kScatterPlot(self, x, y):
        """绘制度的散点图"""
        plt.figure(figsize=(8, 6))  # 设置画布大小
        plt.scatter(x, y, s=50, c='blue', alpha=0.7, edgecolors='w')  # s=点大小, alpha=透明度
        plt.title('西安地铁站点度分布散点图')  # 标题
        plt.xlabel('节点编号')  # x轴标签
        plt.ylabel('节点的度值k')  # y轴标签
        plt.grid(True, linestyle='--')  # 显示网格
        plt.savefig('kscatter.png')

    def kHistPlot(self, data):
        """绘制k值分布的直方图"""
        x = [1, 2, 3, 4, 5]
        sum_k = sum(data)
        for index, ki in enumerate(data):
            data[index] = ki / sum_k
        plt.bar(x, data, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title("西安地铁线路节点度的直方图")
        plt.xlabel("节点的度k")
        plt.ylabel("节点为k的概率p(k)")
        plt.grid(linestyle='--', alpha=0.6)
        plt.savefig('kHist.png')
        return data  # 为k节点分布的概率列表

    def kCurveFit(self, k_division=None):
        """
        回去度分布的曲线拟合图
        :param k_division:度分布的概率列表
        :return:
        """
        x = [1, 2, 3, 4, 5]
        k = np.array(x)
        p_k = np.array(k_division)
        # 曲线拟合
        x_fit = np.linspace(0.8, 5.2, 200)  # 扩展拟合范围
        # 高斯拟合
        gauss_params, _ = curve_fit(self.gaussian, k, p_k, p0=[0.5, 2, 1])

        # 幂律拟合（添加边界条件避免负值）
        power_params, _ = curve_fit(self.power_law, k, p_k,
                                    bounds=([0, -np.inf], [np.inf, np.inf]))

        # 多项式拟合（三阶）
        poly_params, _ = curve_fit(self.polynomial, k, p_k, p0=[1, -5, 5, -1])

        # 可视化设置
        plt.figure(figsize=(10, 6), dpi=100)
        plt.scatter(k, p_k, s=80, marker='s', facecolors='none',
                    edgecolors='#1f77b4', linewidths=1.5, label='数据点')

        # 绘制拟合曲线
        plt.plot(x_fit, self.gaussian(x_fit, *gauss_params),
                 label='高斯分布', color='#2ca02c', lw=2)
        plt.plot(x_fit, self.power_law(x_fit, *power_params), '--',
                 label='幂律分布', color='#ff7f0e', lw=2, dashes=(5, 2))
        plt.plot(x_fit, self.polynomial(x_fit, *poly_params), '-.',
                 label='多项式分布', color='#d62728', lw=2)

        # +++ 新增参数打印语句 +++
        print("高斯分布拟合参数：")
        print(f"幅度 a = {gauss_params[0]:.3f}")
        print(f"均值 μ = {gauss_params[1]:.3f}")
        print(f"标准差 σ = {gauss_params[2]:.3f}\n")

        # 图表美化
        plt.xlabel('k', fontsize=12, labelpad=10)
        plt.ylabel('p(k)', fontsize=12, labelpad=10)
        plt.title('西安地铁线路节点度分布的拟合曲线图', fontsize=12)
        plt.xticks(np.arange(1, 6))
        plt.ylim(-0.25, 1)
        plt.grid(True, alpha=0.3, linestyle=':')
        plt.legend(frameon=False, fontsize=10, loc='upper right')
        plt.tight_layout()

        plt.savefig('kCurveFit.png')

    def gaussian(self, x, a, x0, sigma):
        """高斯分布模型"""
        return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    def power_law(self, x, a, b):
        """幂律分布模型"""
        return a * x ** b

    def polynomial(self, x, *coeffs):
        """多项式模型（三阶示例）"""
        return np.polyval(coeffs, x)

    def kLinePlot(self, k_division=None):
        """
        绘制网络累计度分布图
        :param k_division: 度分布的概率列表
        :return:
        """
        # 数据

        k_division_fuben = deepcopy(k_division)  # 将k_division的值进行深拷贝，只赋值，不赋地址
        x = [1, 2, 3, 4, 5]
        for i in range(5):
            k_division[i] = sum(k_division_fuben[0:i + 1])

        # 创建折线图
        plt.plot(x, k_division, marker='o')

        # 添加标题和标签
        plt.title('西安地铁线路节点度分度的网络累计度分布图')
        plt.xlabel('节点度的k')
        plt.ylabel('节点度的累计概率值')
        plt.ylim(0, 1.01)
        plt.grid(True, color="green")
        plt.savefig('kLinePlot.png')
        k_division_acc = k_division
        return k_division_acc

    def k_sum(self, stationMatrix=None, lineIndex_dict=None):
        """计算每个节点的度"""
        k_dict = {}  # 存储节点度的字典
        scatter_x = []  # 存储站点的标号
        scatter_y = []  # 存储站点的度的数量
        k_division = [0, 0, 0, 0, 0]  # 存储每个度的分布，比如度默认为索引值，从1开始
        for index, i in enumerate(stationMatrix):
            for station_name, station_index in lineIndex_dict.items():
                if station_index == index:
                    k_dict[station_name] = sum(i)
        for k_name, k_index in k_dict.items():
            scatter_y.append(k_index)
            for station_name, station_index in lineIndex_dict.items():
                if k_name == station_name:
                    scatter_x.append(station_index)
        # self.kScatterPlot(x=scatter_x, y=scatter_y)  # 用于绘制西安地铁线路节点度的散点图
        for k_name, k_number in k_dict.items():
            k_division[k_number - 1] += 1
        k_division = self.kHistPlot(data=k_division)  # 用于绘制西安地铁线路节点度的直方图
        # self.kCurveFit(k_division=k_division)  # 用于绘制西安地铁线路节点度分布的拟合曲线图
        self.k_division_acc = self.kLinePlot(k_division=k_division)  # 用于绘制西安地铁线路节点度分度的网络累计度分布图

        return k_dict, k_division

    def shortest_paths(self, stationMatrix):
        n = len(stationMatrix)
        dist_matrix = [[-1] * n for _ in range(n)]  # 初始化距离矩阵

        for i in range(n):
            queue = deque([i])
            dist = [-1] * n
            dist[i] = 0  # 起点到自身距离为0

            while queue:
                current = queue.popleft()
                # 遍历邻接矩阵中当前节点的所有邻居
                for j in range(n):
                    if stationMatrix[current][j] == 1 and dist[j] == -1:
                        dist[j] = dist[current] + 1
                        queue.append(j)

            dist_matrix[i] = dist  # 存储单源最短路径结果

        return dist_matrix

    def averPath(self, stationMatrix=None):
        """
        求解平均路径长度
        :param stationMatrix: 有向地铁邻接矩阵
        :return:
        """
        dist_matrix = []  # 单源最短路径的结果
        averShortPathDict = {}  # 节点间最短路径的长度分布
        averShortPathList = []  # 节点间最短路径的长度分布列表
        averShortPathMatrix = 0  # 平均路径长度的结果
        dist_matrix = self.shortest_paths(stationMatrix=stationMatrix)
        # 求解每个节点最短路径的概率分布
        for i_index, i in enumerate(dist_matrix):
            for j_index, j in enumerate(i):
                if str(j) in averShortPathDict:
                    averShortPathDict[str(j)] += 1
                else:
                    averShortPathDict[str(j)] = 1
        len_averShortPathDict = len(averShortPathDict)
        for i in range(len_averShortPathDict):
            averShortPathList.append(0)
        for path_key, path_value in averShortPathDict.items():
            averShortPathList[int(path_key)] = int(path_value)
        for i_index, i in enumerate(dist_matrix):
            for j_index, j in enumerate(i[i_index:]):
                averShortPathMatrix += j
        averShortPathMatrix = 2 / (len(dist_matrix) * (len(dist_matrix) + 1)) * averShortPathMatrix
        print(f"网络的平均路径长度为{averShortPathMatrix}")

    def run(self):
        """程序入口"""
        self.line_dict, self.lineIndex_dict = self.prepareData()
        # 获取地铁邻接矩阵
        self.stationMatrix = self.adjacent_matrix(self.line_dict, self.lineIndex_dict)
        # 绘制地铁拓扑图
        # self.topological_graph(lineDict=self.line_dict)
        # 求度和度的概率分布
        # self.k_dict, self.k_division = self.k_sum(stationMatrix=self.stationMatrix, lineIndex_dict=self.lineIndex_dict)
        # 求平均路径长度
        self.averPath(stationMatrix=self.stationMatrix)


if __name__ == '__main__':
    cgp = complexGraphProject()
    cgp.run()
