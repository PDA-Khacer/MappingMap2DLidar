import math
import numpy
import numpy as np
from matplotlib import pyplot as plt

dPTL = 80
dPTP = 80
dPTL_rg = 100
dPTP2 = 200


def dataSet(fileName):
    f = open(fileName, "r")
    all_piece = []
    piece = []
    count = 0
    while count != 3:
        line = f.readline()
        if line[0] == '\n':
            count += 1
            if count == 2:
                all_piece.append(piece)
                piece = []
        else:
            count = 0
            point = {}
            ele = line.split(',')
            d = float(ele[0])
            b = float(ele[1])
            point['x'] = math.sin(math.radians(b)) * d
            point['y'] = math.cos(math.radians(b)) * d
            point['theta'] = b
            piece.append(point)
    return all_piece


def distantPointToLine(point, line):
    return abs(line['a'] * point.get("x") + line.get("b") * point["y"] + line.get("c")) / math.sqrt(
        line.get("a") ** 2 + line.get("b") ** 2)


def distantPointToPoint(point1, point2):
    return math.sqrt((point1.get("x") - point2.get("x")) ** 2 + (point1.get("y") - point2.get("y")) ** 2)


def convertMatrix(points):
    init_matrix = np.empty((0, 3), int)
    for index in range(len(points)):
        add_matrix = np.array([[points[index].get('x'), 1, points[index].get('y')]])
        init_matrix = np.append(init_matrix, add_matrix, axis=0)
    return init_matrix


def fitSeed(points):
    line = dict(a=0, b=0, c=0)
    matrix = convertMatrix(points)
    matrix_a = matrix[:, 0:2]
    matrix_b = matrix[:, 2]
    matrix_x = (np.transpose(matrix_a).dot(matrix_b)).dot(np.linalg.inv(np.transpose(matrix_a).dot(matrix_a)))
    line["a"] = matrix_x[0]
    line["b"] = -1
    line["c"] = matrix_x[1]
    return line


def predictPoint(point, line):
    pre_point = dict(d=0, theta=0, x=0, y=0)
    theta = point.get('theta') - int(point.get('theta') / 90) * 90
    theta = math.radians(90 - theta)
    if 0 < point.get('theta') < 90 or 180 < point.get('theta') < 270:
        pre_point["x"] = -line.get("c") * math.cos(theta) / (
                line.get("a") * math.cos(theta) + line.get("b") * math.sin(theta))
        pre_point["y"] = -line.get("c") * math.sin(theta) / (
                line.get("a") * math.cos(theta) + line.get("b") * math.sin(theta))
    else:
        pre_point["x"] = -line.get("c") * math.sin(theta) / (
                line.get("a") * math.sin(theta) + line.get("b") * math.cos(theta))
        pre_point["y"] = -line.get("c") * math.cos(theta) / (
                line.get("a") * math.sin(theta) + line.get("b") * math.cos(theta))
    return pre_point


def seedSegment(points, Snum, Pmin):
    result = []
    for index in range(len(points) - Snum):
        flag = True
        j = index + Pmin
        line = fitSeed(points[index:j])
        for k in range(index, j):
            pre_point = predictPoint(points[k], line)
            d1 = distantPointToPoint(points[k], pre_point)
            d2 = distantPointToLine(points[k], line)
            d3 = -1.0
            if k != j:
                d3 = distantPointToPoint(points[k], points[k + 1])
            # #################--------------------------------------------------------
            # if index > 210:
            #     print_point(res[0])
            #     x = []
            #     y = []
            #     for index1 in range(index, j):
            #         x.append(points[index1].get('x'))
            #         y.append(points[index1].get('y'))
            #     plt.scatter(x, y, label="point " + str(index) + " " + str(j), s=10, color='red')
            #     x = np.linspace(min(x), max(x), 2000)
            #     y = line.get('a') * x + line.get('c')
            #     plt.plot(x, y, color='b', label="line | " + str(d3))
            #     x = []
            #     y = []
            #     x.append(pre_point['x'])
            #     y.append(pre_point['y'])
            #     plt.scatter(x, y, s=10, marker="*", color='black')
            #     x = []
            #     y = []
            #     x.append(points[k]['x'])
            #     y.append(points[k]['y'])
            #     plt.scatter(x, y, label="point "+str(k)+' d1 = '+str(d1)+'d2 = '+str(d2), s=10, marker="*", color='g')
            #     mng = plt.get_current_fig_manager()
            #     mng.full_screen_toggle()
            #     plt.legend()
            #     plt.show()
            # ######################--------------------------------------------------
            # if d1 > dPTP:
            #     flag = False
            #     break
            if d2 > dPTL:
                flag = False
                break
            if d3 > dPTP2 and k != j:
                flag = False
                break
        if flag is True:
            result.append(dict(i=index, j=j, line=line))
    return result


# cần xác định lại phần return để tối ưu hơn


def regionGrowing(points, seed, Np, Pmin, Lmin):
    line = dict(pb=seed.get("i"), pf=seed.get("j"), line=seed.get("line"))
    # ####################
    # x = []
    # y = []
    # for j in range(line['pb'], line['pf']):
    #     x.append(points[j]['x'])
    #     y.append(points[j]['y'])
    # plt.scatter(x, y, label='line root', s=5, c='b')
    # plt.scatter([points[line['pb']]['x']], [points[line['pf']+1]['y']], label='point add', s=5, c='r')
    # line["pf"] += 1
    # plt.legend()
    # plt.show()
    # ####################
    while distantPointToLine(points[line["pf"]], line["line"]) < dPTL_rg:
        if line["pf"] >= Np:  # line["pf"] < len(points)
            break
        else:
            line["line"] = fitSeed(points[line["pb"]:line["pf"]])
        line["pf"] += 1
    line["pf"] -= 1
    line["pb"] -= 1
    while distantPointToLine(points[line["pb"]], line["line"]) < dPTL_rg:
        if line["pb"] < 0:  # line["pf"] < len(points)
            break
        else:
            line["line"] = fitSeed(points[line["pb"]:line["pf"]])
        line["pb"] -= 1
    line["pb"] += 1

    # line_001 = create_endpoint(points, line)
    if line["pf"] - line["pb"] + 1 > Pmin and 999 >= Lmin:  # define lại Ll
        return line
    else:
        return dict(pb=seed.get("i"), pf=seed.get("j"), line=seed.get("line"))


def overlap_region(points, segments):
    for index1 in range(len(segments) - 1):
        index2 = index1 + 1
        segments = sorted(segments, key=lambda k: k['pb'])
        # ##################################
        # x = []
        # y = []
        # for j in range(segments[index1]['pb'], segments[index1]['pf']):
        #     x.append(points[j]['x'])
        #     y.append(points[j]['y'])
        # plt.scatter(x, y, label='line' + str(index1), s=5, c=numpy.random.rand(3, ))
        # x = []
        # y = []
        # for j in range(segments[index2]['pb'], segments[index2]['pf']):
        #     x.append(points[j]['x'])
        #     y.append(points[j]['y'])
        # plt.scatter(x, y, label='line' + str(index2), s=5, c=numpy.random.rand(3, ))
        # plt.legend()
        # plt.show()
        # ##################################
        if segments[index1]['pf'] >= segments[index2]['pb']:
            d = -1
            for k in range(segments[index2]['pb'], segments[index1]['pf']):
                d = k
                d1 = distantPointToLine(points[k], segments[index1]['line'])
                d2 = distantPointToLine(points[k], segments[index2]['line'])
                if d1 < d2:
                    continue
                else:
                    break
            # if d != -1 and abs(segments[index1]['pb'] - d) > 1 and abs(segments[index2]['pf'] - d - 1) > 1:
            if d != -1:
                segments[index1]['pf'] = d - 1
                segments[index2]['pb'] = d + 1
                if abs(segments[index1]['pb'] - d + 1) > 1:
                    segments[index1]['line'] = fitSeed(points[segments[index1]['pb']:segments[index1]['pf']])
                if abs(segments[index2]['pf'] - d - 1) > 1:
                    segments[index2]['line'] = fitSeed(points[segments[index2]['pb']:segments[index2]['pf']])
    return segments


def create_endpoint(points, line):
    newline = dict()
    p0 = line.get('pb')
    pn = line.get('pf')
    a = float(line['line']['a'])
    b = float(line['line']['b'])
    c = float(line['line']['c'])
    p = dict(x=0, y=0, line=None)
    p['x'] = (b ** 2 * float(points[p0]['x']) - a * b * float(points[p0]['y']) - a * c) / (a ** 2 + b ** 2)
    p['y'] = (a ** 2 * float(points[p0]['y']) - a * b * float(points[p0]['x']) - b * c) / (a ** 2 + b ** 2)
    newline['pb'] = p
    p = dict(x=0, y=0, line=None)
    p['x'] = (b ** 2 * points[pn]['x'] - a * b * points[pn]['y'] - a * c) / (a ** 2 + b ** 2)
    p['y'] = (a ** 2 * points[pn]['y'] - a * b * points[pn]['x'] - b * c) / (a ** 2 + b ** 2)
    newline['pf'] = p
    return newline


def removeExcess(segment_line):
    token_segment = [segment_line[0]]
    for item in segment_line:
        flag_func = True
        for item_token in token_segment:
            if item_token['pb'] >= item['pb'] and item_token['pf'] <= item['pf']:
                item_token['pb'] = item['pb']
                item_token['pf'] = item['pf']
                item_token['line'] = item['line']
                flag_func = False
            elif item_token['pb'] <= item['pb'] and item_token['pf'] >= item['pf']:
                flag_func = False
        if flag_func is True:
            token_segment.append(item)
    segment_line = token_segment
    token_segment = [segment_line[0]]
    for item in segment_line:
        flag_func = True
        for item_token in token_segment:
            if item_token['pb'] == item['pb'] and item_token['pf'] == item['pf']:
                flag_func = False
        if flag_func is True:
            token_segment.append(item)
    return token_segment


def linkLine(points, segment_line, pMin):
    re = []
    # loại bỏ những đường có số điểm < p min và nối chúng với đường trước nó
    # có thể phát triển điều kiện kiếm tra 2 đường đó gần đường nào hơn để tiến hành ghép
    # hàm nối với đường trước nó
    # for index in range(len(segment_line)-1):
    #     if segment_line[index]['pf'] - segment_line[index]['pb'] + 1 < pMin:
    #         if len(re) != 0:
    #             re[len(re)-1]["pf"] = segment_line[index]['pf']
    #     else:
    #         re.append(segment_line[index])
    # hàm nối với đường trước hoặc sau
    for index in range(len(segment_line)):
        if segment_line[index]['pf'] - segment_line[index]['pb'] + 1 < pMin:
            if len(re) != 0 and index != len(segment_line) - 1:
                d1 = distantPointToLine(points[segment_line[index]['pb']], re[len(re) - 1]['line'])
                d2 = distantPointToLine(points[segment_line[index]['pf']], segment_line[index + 1]['line'])
                if d1 < d2:
                    re[len(re) - 1]["pf"] = segment_line[index]['pf']
                else:
                    segment_line[index + 1]['pb'] = segment_line[index]['pb']
        re.append(segment_line[index])
    # kiểm tra điểm ở giữa 2 đường có thể ghép vào đường nào
    for index in range(len(re) - 2):
        if re[index + 1]['pb'] - re[index]['pf'] + 1 <= pMin:
            flag_left = True
            flag_right = True
            index2 = re[index]['pf'] + 1
            # for index2 in range(re[index]['pf'] + 1, re[index+1]['pb'] - 1):
            while index2 <= re[index + 1]['pb'] - 1:
                if flag_left is True:
                    d10 = distantPointToLine(points[index2], re[index]['line'])
                    d11 = distantPointToPoint(points[index2], points[re[index]['pf']])
                    if d10 > dPTL or d11 > 200:
                        flag_left = False
                if flag_right is True:
                    d20 = distantPointToLine(points[index2], re[index + 1]['line'])
                    d22 = distantPointToPoint(points[index2], points[re[index + 1]['pb']])
                    if d20 > dPTL or d22 > 200:
                        flag_right = False
                if flag_left == flag_right is False:
                    break
                else:
                    if flag_left == flag_right is True:
                        if d10 > d20:
                            re[index + 1]['pb'] = index2
                            re[index + 1]['line'] = fitSeed(points[re[index + 1]["pb"]:re[index + 1]["pf"]])
                        else:
                            re[index]['pf'] = index2
                            re[index]['line'] = fitSeed(points[re[index]["pb"]:re[index]["pf"]])
                    elif flag_left is True:
                        re[index]['pf'] = index2
                        re[index]['line'] = fitSeed(points[re[index]["pb"]:re[index]["pf"]])
                    elif flag_right is True:
                        re[index + 1]['pb'] = index2
                        re[index + 1]['line'] = fitSeed(points[re[index + 1]["pb"]:re[index + 1]["pf"]])
                index2 += 1
    return re


def print_0(item):
    for it in item:
        print(it)
    print()


def print_1(item):
    for i in range(len(item)):
        x = []
        y = []
        for j in range(item[i]['i'], item[i]['j']):
            x.append(res[0][j]['x'])
            y.append(res[0][j]['y'])
        plt.scatter(x, y, label='line' + str(i), s=5, c=numpy.random.rand(3, ))
    plt.legend()
    plt.show()


def print_11(item):
    for i in range(len(item)):
        x = []
        y = []
        for j in range(item[i]['i'], item[i]['j']):
            x.append(res[0][j]['x'])
            y.append(res[0][j]['y'])
        plt.scatter(x, y, label='line' + str(i), s=5, c=numpy.random.rand(3, ))
        plt.legend()
        plt.show()


def print_2(item):
    for i in range(len(item)):
        x = []
        y = []
        for j in range(item[i]['pb'], item[i]['pf']):
            x.append(res[0][j]['x'])
            y.append(res[0][j]['y'])
        plt.scatter(x, y, label='line' + str(i), s=5, c=numpy.random.rand(3, ))
    plt.legend()
    plt.show()


def print_point(res):
    x = []
    y = []
    for i in res:
        x.append(i['x'])
        y.append(i['y'])
    plt.scatter(x, y, s=10, color='teal')


res = dataSet("D:\\Document\\MTA\\BigProject\\DTKH\\2020\\Source\\Data\\myOutput1.csv")
for ij in range(len(res)):
    print_point(res[ij])
    segment = seedSegment(res[ij], 30, 8)
    # print_1(segment)
    # print_0(segment)
    segment_seed = []
    for i in range(len(segment)):
        segment_seed.append(regionGrowing(res[ij], segment[i], len(res[ij]), 8, 3))
    # print_0(segment_seed)
    # remove vùng con và trùng nhau
    segment_seed = removeExcess(segment_seed)
    lineRes = segment_seed
    lineRes = overlap_region(res[ij], sorted(segment_seed, key=lambda k: k['pb'], reverse=False))
    lineRes = linkLine(res[ij], lineRes, 3)
    for i in range(len(lineRes)):
        lines = create_endpoint(res[ij], lineRes[i])
        x = []
        y = []
        x.append(lines['pb']['x'])
        x.append(lines['pf']['x'])
        y.append(lines['pb']['y'])
        y.append(lines['pf']['y'])
        plt.plot(x, y, label='line' + str(i))
    plt.legend()
    plt.show()

# ##########################################################################################+
#     for i in range(len(lineRes)):
#         x = []
#         y = []
#         print(i, lineRes[i])
#         for j in range(lineRes[i]['pb'], lineRes[i]['pf']):
#             x.append(res[0][j]['x'])
#             y.append(res[0][j]['y'])
#         plt.scatter(x, y, label='line' + str(i), s=5, c=numpy.random.rand(3, ))
#     plt.legend()
#     plt.show()
# ##########################################################################################
