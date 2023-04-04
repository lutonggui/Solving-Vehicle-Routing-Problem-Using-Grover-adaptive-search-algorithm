import math

from pyqpanda import *
import pyqpanda as py
import matplotlib.pyplot as plt
import numpy as np

# locations为加上配送中心后的位置总数，vihicles是车辆数目
Locations = 3
vehicles = 2
n = 6  # 变量（键）寄存器的比特数量
m = 5  # 系数（值）寄存器的比特数量
q = 3


# 经典部分的数据准备
def Classical_preparation(Locations, vehicles):
    Target = np.zeros((Locations, Locations, Locations - 1))  # 生成一个存储着5个5行四列矩阵的数组,将5个ZT向量存储在其中
    for number in range(Locations):
        for i in range(Locations):
            for j in range(Locations - 1):
                if i < number:
                    Target[number][i][number - 1] = 1
                else:
                    if i > number:
                        Target[number][i][number] = 1
                    else:
                        if i == number:
                            Target[number][i] = 0

    Source = np.zeros((Locations, Locations, Locations - 1))  # 生成一个存储着5个5行四列矩阵的数组,将5个ZS向量存储在其中

    for number in range(Locations):
        for i in range(Locations):
            for j in range(Locations - 1):
                if i == number:
                    Source[number][i] = 1
    '''
    for k in range(5):
        print(k, Target[k])
    for k in range(5):
        print(k, Source[k])
    '''

    ZT = []  # ZT里面存储的是行向量
    for k in range(Locations):
        ZT.append(Target[k].copy().reshape(1, Locations * (Locations - 1)))

    ZS = []  # ZS里面存储的是行向量
    for k in range(Locations):
        ZS.append(Source[k].copy().reshape(1, Locations * (Locations - 1)))

    # Q_HB矩阵完成
    Q_HB = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HB = Q_HB + np.multiply(ZS[m].T, ZS[m])
    # Q_HC矩阵完成
    Q_HC = np.zeros((Locations * (Locations - 1), Locations * (Locations - 1)))
    for m in range(1, Locations):
        Q_HC = Q_HC + np.multiply(ZT[m].T, ZT[m])
    # Q_HD矩阵完成
    Q_HD = np.multiply(ZS[0].T, ZS[0])
    # Q_HE矩阵完成
    Q_HE = np.multiply(ZT[0].T, ZT[0])

    print("二次限制矩阵Q_HB为", Q_HB)
    print("二次限制矩阵Q_HC为", Q_HC)
    print("二次限制矩阵Q_HD为", Q_HD)
    print("二次限制矩阵Q_HE为", Q_HE)
    # 下面开始准备生成g

    # 随机生成配送中心以外的位置的坐标
    loca = [0]
    loca[0] = np.zeros((1, 2))
    np.random.seed(9)
    for L in range(Locations - 1):
        loca.append(np.random.uniform(1, 5, 2))

    # 下面计算各个点之间的距离
    w = np.zeros((Locations, Locations - 1))  # 为了方便，先把权重放在矩阵里面

    for n in range(Locations):
        for p in range(Locations):
            if p > n:
                w[n][p - 1] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)
            else:
                if p < n:
                    w[n][p] = pow(np.sum((loca[n] - loca[p]) ** 2), 0.5)

    W = w.copy().reshape((1, Locations * (Locations - 1)))

    # 为了方便，将W中的各项进行四舍五入的取整
    for x in range(Locations * (Locations - 1)):
        W[0][x] = round(W[0][x])
    print("当前权重为，", W)

    g_HB = -2 * (ZS[1] + ZS[2])
    g_HC = -2 * (ZT[1] + ZT[2])
    g_HD = -2 * vehicles * ZS[0]
    g_HE = -2 * vehicles * ZT[0]
    print("一次项系数g_HB为", g_HB)
    print("一次项系数g_HC为", g_HC)
    print("一次项系数g_HD为", g_HD)
    print("一次项系数g_HE为", g_HE)
    # 下面准备常数系数

    c_HB = 2
    c_HC = 2
    c_HD = vehicles ** 2
    c_HE = vehicles ** 2

    print("待编码的经典数据生成完毕")
    return W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE


W, Q_HB, Q_HC, Q_HD, Q_HE, g_HB, g_HC, g_HD, g_HE, c_HB, c_HC, c_HD, c_HE = Classical_preparation(Locations,
                                                                                                  vehicles)
c = -16  # f(x)函数-阈值
print("当前fx的阈值为", -c)
# 量子算法部分

# 量子虚拟机初始化
machine = py.init_quantum_machine(py.QMachineType.CPU)

# 电路中最多可以使用的比特数量

xbits = machine.qAlloc_many(n)
zbits = machine.qAlloc_many(m)
HBbits = machine.qAlloc_many(q)
HCbits = machine.qAlloc_many(q)
HDbits = machine.qAlloc_many(q)
HEbits = machine.qAlloc_many(q)
cbits = machine.cAlloc_many(q)
anc = machine.qAlloc_many(1)
# 构建量子程序
prog = py.create_empty_qprog()


# 定义编码部分使用的UG门

def UG(k, m, bits, control=[]):
    number = (2 * np.pi * k) / pow(2, m)  # 当前待编码的数字k
    theta = []
    for i in range(m):
        angle = number * (2 ** i)
        theta.append(angle)
    if len(control) == 2:
        for j in range(m):
            prog.insert(CR(xbits[control[0]], bits[j], theta[j]).control(xbits[control[1]]))
    else:
        if len(control) == 1:
            for j in range(m):
                prog.insert(CR(xbits[control[0]], bits[j], theta[j]))
                prog.insert(BARRIER(xbits))
        else:
            if len(control) == 0:
                for j in range(m):
                    prog.insert(U1(bits[j], theta[j]))


def UG_dagger(k, m, bits, control=[]):
    number = (2 * np.pi * k) / pow(2, m)  # 当前待编码的数字k
    theta = []
    for i in range(m):
        angle = number * (2 ** i)
        theta.append(angle)
    if len(control) == 2:
        for j in range(m):
            prog.insert(CR(xbits[control[0]], bits[m - 1 - j], -theta[m - 1 - j]).control(xbits[control[1]]))
    else:
        if len(control) == 1:
            for j in range(m):
                prog.insert(CR(xbits[control[0]], bits[m - 1 - j], -theta[m - 1 - j]))
                prog.insert(BARRIER(xbits))
        else:
            if len(control) == 0:
                for j in range(m):
                    prog.insert(U1(bits[j], -theta[j]))


# 电路初始化

prog.insert(H(xbits))
prog.insert(H(zbits))
prog.insert(H(HBbits))
prog.insert(H(HCbits))
prog.insert(H(HDbits))
prog.insert(H(HEbits))


# 数据编码


# Q矩阵编码
def Q_Matrix(Q, bits, Locations, m):
    for hang in range(Locations * (Locations - 1)):
        for lie in range(Locations * (Locations - 1)):
            if hang == lie:
                UG(Q[hang][lie], m, bits, [hang])
            else:
                UG(Q[hang][lie], m, bits, [hang, lie])


def Q_Matrix_dagger(Q, bits, Locations, m):
    for hang in range(Locations * (Locations - 1)):
        for lie in range(Locations * (Locations - 1)):
            if hang == lie:
                UG_dagger(Q[m - 1 - hang][m - 1 - lie], m, bits, [m - 1 - hang])
            else:
                UG_dagger(Q[m - 1 - hang][m - 1 - lie], m, bits, [m - 1 - hang, m - 1 - lie])


# g编码
def Yi_ci(g, bits, Locations, m):
    for index in range(Locations * (Locations - 1)):
        UG(g[0][index], m, bits, [index])


def Yi_ci_dagger(g, bits, Locations, m):
    for index in range(Locations * (Locations - 1)):
        UG_dagger(g[0][m - 1 - index], m, bits, [m - 1 - index])


# HA限制条件
Yi_ci(W, zbits, Locations, m)
UG(c, m, zbits)
# HB限制条件
Q_Matrix(Q_HB, HBbits, Locations, q)
Yi_ci(g_HB, HBbits, Locations, q)
UG(c_HB, q, HBbits)

# HC限制条件
Q_Matrix(Q_HC, HCbits, Locations, q)
Yi_ci(g_HC, HCbits, Locations, q)
UG(c_HC, q, HCbits)

# HD限制条件
Q_Matrix(Q_HD, HDbits, Locations, q)
Yi_ci(g_HD, HDbits, Locations, q)
UG(c_HD, q, HDbits)

# HE限制条件
Q_Matrix(Q_HE, HEbits, Locations, q)
Yi_ci(g_HE, HEbits, Locations, q)
UG(c_HE, q, HEbits)

prog.insert(py.QFT(zbits).dagger())
prog.insert(py.QFT(HBbits).dagger())
prog.insert(py.QFT(HCbits).dagger())
prog.insert(py.QFT(HDbits).dagger())
prog.insert(py.QFT(HEbits).dagger())

# 开始执行GAS算法
K = 2  # G算子执行的次数范围
r = 2  # np.random.randint(K + 1)  G算子执行的次数

print("r", r)


# 定义oracle
def Oracle():
    # 检测当前函数值是否小于设定的阈值，以及限制条件是否满足
    prog.insert(X(HBbits))
    prog.insert(X(HCbits))
    prog.insert(X(HDbits))
    prog.insert(X(HEbits))

    cvec = [HBbits[0], HBbits[1], HBbits[2], HCbits[0], HCbits[1], HCbits[2], HDbits[0], HDbits[1], HDbits[2],
            HEbits[0],
            HEbits[1], HEbits[2]]

    prog.insert(CNOT(zbits[m - 1], anc[0]).control(cvec))  # 如果条件们都满足，anc将被反转为1.

    prog.insert(X(HBbits))
    prog.insert(X(HCbits))
    prog.insert(X(HDbits))
    prog.insert(X(HEbits))

    prog.insert(CZ(anc[0], zbits[m - 1]))  # 反转目标态的相位


# 定义G算子

def Grover():
    prog.insert(py.QFT(HEbits))
    prog.insert(py.QFT(HDbits))
    prog.insert(py.QFT(HCbits))
    prog.insert(py.QFT(HBbits))
    prog.insert(py.QFT(zbits))
    # HE部分的dagger
    UG_dagger(c_HE, q, HEbits)
    Yi_ci_dagger(g_HE, HEbits, Locations, q)
    Q_Matrix_dagger(Q_HE, HEbits, Locations, q)

    # HD部分的dagger
    UG_dagger(c_HD, q, HDbits)
    Yi_ci_dagger(g_HD, HDbits, Locations, q)
    Q_Matrix_dagger(Q_HD, HDbits, Locations, q)

    # HC部分的dagger
    UG_dagger(c_HC, q, HCbits)
    Yi_ci_dagger(g_HC, HCbits, Locations, q)
    Q_Matrix_dagger(Q_HC, HCbits, Locations, q)

    # HB部分的dagger
    UG_dagger(c_HB, q, HBbits)
    Yi_ci_dagger(g_HB, HBbits, Locations, q)
    Q_Matrix_dagger(Q_HB, HBbits, Locations, q)

    # HA部分的dagger
    UG_dagger(c, m, zbits)
    Yi_ci_dagger(W, zbits, Locations, m)

    prog.insert(H(xbits))
    prog.insert(H(zbits))
    prog.insert(H(HBbits))
    prog.insert(H(HCbits))
    prog.insert(H(HDbits))
    prog.insert(H(HEbits))
    for l in range(n):
        prog.insert(X(xbits[l]))
    for t in range(m):
        prog.insert(X(zbits[t]))
    for p in range(q):
        prog.insert(X(HBbits[p]))
    for p in range(q):
        prog.insert(X(HCbits[p]))
    for p in range(q):
        prog.insert(X(HDbits[p]))
    for p in range(q):
        prog.insert(X(HEbits[p]))
    prog.insert(X(anc[0]))

    vec = [xbits[1], xbits[2], xbits[3], xbits[4], xbits[5], zbits[0], zbits[1], zbits[2], zbits[3], HBbits[0],
           HBbits[1], HBbits[2], HCbits[0], HCbits[1], HCbits[2], HDbits[0], HDbits[1], HDbits[2],
           HEbits[0], HEbits[1], HEbits[2], anc[0]]
    prog.insert(CZ(xbits[0], zbits[m - 1]).control(vec))

    for l in range(n):
        prog.insert(X(xbits[l]))
    for t in range(m):
        prog.insert(X(zbits[t]))
    for p in range(q):
        prog.insert(X(HBbits[p]))
    for p in range(q):
        prog.insert(X(HCbits[p]))
    for p in range(q):
        prog.insert(X(HDbits[p]))
    for p in range(q):
        prog.insert(X(HEbits[p]))
    prog.insert(X(anc[0]))

    prog.insert(H(xbits))
    prog.insert(H(zbits))
    prog.insert(H(HBbits))
    prog.insert(H(HCbits))
    prog.insert(H(HDbits))
    prog.insert(H(HEbits))
    # HA限制条件
    Yi_ci(W, zbits, Locations, m)
    UG(c, m, zbits)
    # HB限制条件
    Q_Matrix(Q_HB, HBbits, Locations, q)
    Yi_ci(g_HB, HBbits, Locations, q)
    UG(c_HB, q, HBbits)

    # HC限制条件
    Q_Matrix(Q_HC, HCbits, Locations, q)
    Yi_ci(g_HC, HCbits, Locations, q)
    UG(c_HC, q, HCbits)

    # HD限制条件
    Q_Matrix(Q_HD, HDbits, Locations, q)
    Yi_ci(g_HD, HDbits, Locations, q)
    UG(c_HD, q, HDbits)

    # HE限制条件
    Q_Matrix(Q_HE, HEbits, Locations, q)
    Yi_ci(g_HE, HEbits, Locations, q)
    UG(c_HE, q, HEbits)

    prog.insert(py.QFT(zbits).dagger())
    prog.insert(py.QFT(HBbits).dagger())
    prog.insert(py.QFT(HCbits).dagger())
    prog.insert(py.QFT(HDbits).dagger())
    prog.insert(py.QFT(HEbits).dagger())



for times in range(r):
    Oracle()
    Grover()

# print(prog)
# 对量子程序进行概率测量
qvec = [xbits[0], xbits[1], xbits[2], xbits[3], xbits[4], xbits[5], zbits[0], zbits[1], zbits[2], zbits[3], zbits[4],
        anc[0]]

result = py.prob_run_dict(prog, qvec, -1)
py.destroy_quantum_machine(machine)

out = {}
for key, value in result.items():
    if value > 10 ** (-4):
        out[key] = value
print(len(out))
# print(out)

# Save
np.save('my_file.npy', out)  # 注意带上后缀名
# 打印测量结果
for key in out:
    print(key + ": " + str(out[key]))
