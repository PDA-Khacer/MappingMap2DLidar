import math
import numpy
import numpy as np
from matplotlib import pyplot as plt

dPTL = 80
dPTP = 80
dPTL_rg = 80
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
            d2 = distantPointToLine(points[k], line)
            d3 = -1.0
            if k != j:
                d3 = distantPointToPoint(points[k], points[k+1])
            if d2 > dPTL:
                flag = False
                break
            if d3 > dPTP2 and k != j:
                flag = False
                break
        if flag is True:
            result.append(dict(i=index, j=j, line=line))
    return result


def regionGrowing(points, seed, Np, Pmin, Lmin):
    line = dict(pb=seed.get("i"), pf=seed.get("j"), line=seed.get("line"))
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
    if line["pf"] - line["pb"] + 1 > Pmin and 999 >= Lmin:  # define lại Ll
        return line
    else:
        return dict(pb=seed.get("i"), pf=seed.get("j"), line=seed.get("line"))


def overlap_region(points, segments):
    for index1 in range(len(segments) - 1):
        index2 = index1 + 1
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
            if d != -1 and abs(segments[index1]['pb'] - d) > 1 and abs(segments[index2]['pf'] - d) > 1:
                segments[index1]['pf'] = d
                segments[index2]['pb'] = d
                segments[index1]['line'] = fitSeed(points[segments[index1]['pb']:segments[index1]['pf']])
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


def print_point(res):
    x = []
    y = []
    for i in res:
        x.append(i['x'])
        y.append(i['y'])
    plt.scatter(x, y, s=10, color='teal')


res = dataSet("..\\data\\data1.txt")
for ij in range(len(res)):
    print_point(res[ij])
    segment = seedSegment(res[ij], 30, 8)
    segment_seed = []
    for i in range(len(segment)):
        segment_seed.append(regionGrowing(res[ij], segment[i], len(res[ij]), 8, 3))
    segment_seed = removeExcess(segment_seed)
    lineRes = overlap_region(res[ij], segment_seed)
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

