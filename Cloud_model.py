# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np 
import random 
import math
import sys
    
Ex = 20
En = 1
He = 0.1



def Cloud_positive(Ex,En,He,court):
    Cloud_x = []
    Cloud_u = []
    i = 0
    while i<court:
        y = random.gauss(En,He)
        x = random.gauss(Ex,y)
        u = math.e**(-pow((x - Ex),2) / (2 * pow(y,2)))
        
        Cloud_x.append(x)
        Cloud_u.append(u)
        i += 1

    return Cloud_x,Cloud_u

def Cloud_reverse_1(x,u):
    Cloud_y = []
    Ex_y = 0
    Dx_y = 0
    i = 0
    Ex = sum(x) / len(x)
    i = 0
    while i<len(x):
                       
        y = round(-(pow((x[i] - Ex),2)) / (2 * math.log(u[i], math.e)),5)

        Cloud_y.append(y)
        i += 1
    i = 0
    while i<len(Cloud_y):
        Ex_y = Ex_y + Cloud_y[i]
        i += 1
    Ex_y = Ex_y / len(Cloud_y)
    
    i = 0
    while i<len(Cloud_y):
        Dx_y = Dx_y + pow((Cloud_y[i] - Ex_y),2)
        i += 1
    Dx_y = Dx_y / len(Cloud_y)
    En = pow(0.5 * pow(abs(4 * pow(Ex_y,2) - 2 * Dx_y),0.5),0.5)
    He = pow(abs(Ex_y - pow(En,2)),0.5) 
    
    return Ex,En,He

def Cloud_reverse_2(x):
    S1 = 0
    S2 = 0
    i = 0
    Ex = sum(x) / len(x)
    while i<len(x):
        S1 = S1 + math.fabs(x[i] - Ex)
        i += 1
    S1 = S1 / len(x)
    i = 0
    while i<len(x):
        S2 = S2 + math.pow((x[i] - Ex),2)
        
        i += 1
    S2 = S2 / (len(x) - 1)
    En = math.pow(abs(math.pi / 2),0.5) * S1
    He = math.pow(abs(S2 - math.pow(En,2)),0.5) 
        
    return Ex,En,He

def Cloud_reverse_3(x):
    S1 = 0
    S2 = 0
    S4 = 0
    i = 0
    Ex = sum(x) / len(x)
    while i<len(x):
        S1 = S1 + math.fabs(x[i] - Ex)
        i += 1
    S1 = S1 / len(x)
    
    i = 0
    while i<len(x):
        S2 = S2 + math.pow((x[i] - Ex),2)        
        i += 1
    S2 = S2 / (len(x) - 1)
    
    i = 0
    while i<len(x):
        S4 = S4 + math.pow((x[i] - Ex),4)        
        i += 1
    S4 = S4 / (len(x) - 1)
    En = math.pow(abs((9 * pow(S2,2) - S4) / 6),0.25)
    He = math.pow(abs(S2 - pow(abs((9 * pow(S2,2) - S4) / 6),0.5)),0.5)
    
    return Ex,En,He

def gaosi_g(y,miu,seita):
    return 1/(math.sqrt(2*math.pi)*seita)*math.e**((-(y-miu)**2)/(2*seita**2))

def L_k(x,miu,seita,aerfa,k):
    fenzi = gaosi_g(x,miu[k],seita[k]) * aerfa[k]
    fenmu = 0
    for n in range(len(miu)):
        fenmu += aerfa[n] * gaosi_g(x,miu[n],seita[n])
    return fenzi / fenmu

def cal_J(y,y_count,aerfa,miu,seita,k):
    J = 0
    for y_d in y_count:
        J_ln = 0
        for i in range(k):
            J_ln += aerfa[i] * gaosi_g(y_d,miu[i],seita[i])
        J_ln = math.log(J_ln)
        J += y_d * J_ln
    return J

def update_can(data,aerfa,miu,seita,k):
    for k_ in range(k):
        fenzi_m = 0
        fenmu_m = 0
        fenzi_s = 0
        fenmu_s = 0
        aerfa_update = 0
        
        for xi in data:
            fenzi_m += L_k(xi,miu,seita,aerfa,k_) * xi
            fenmu_m += L_k(xi,miu,seita,aerfa,k_)
            
            fenzi_s += L_k(xi,miu,seita,aerfa,k_) * (xi - miu[k_])**2
            fenmu_s += L_k(xi,miu,seita,aerfa,k_)
            
            aerfa_update += L_k(xi,miu,seita,aerfa,k_)
            
        miu[k_] = fenzi_m / fenmu_m
        seita[k_] = math.sqrt(fenzi_s / fenmu_s)
        aerfa[k_] = aerfa_update / data.shape[0]
        
    return aerfa,miu,seita

def gaosi_tran(data,k,target):
    miu = []
    seita = []
    aerfa = []
    y = []
    y_count = []
    for d in data:
        if d in y:
            y_count[y.index(d)] += 1
        else:
            y.append(d)
            y_count.append(1)
    y_count = np.array(y_count)/data.shape[0]
    for i in range(k):
        miu.append((i+1)*data.max()/(k+1))
        seita.append(data.max())
        aerfa.append(1/k)
        
    while True:
        J = cal_J(y,y_count,aerfa,miu,seita,k)
        aerfa,miu,seita = update_can(data,aerfa,miu,seita,k)
        J_update = cal_J(y,y_count,aerfa,miu,seita,k) 
        if abs(J_update - J)<target:
            break
    return aerfa,miu,seita

def cloud_tran(data,k,target):
    aerfa,miu,seita = gaosi_tran(data,k,target)
    Ex = []
    En = []
    He = []
    CD = []
    for i in range(len(aerfa)):
        Ex.append(miu[i])
        En.append((1+aerfa[i])*seita[i]/2)
        He.append((1-aerfa[i])*seita[i]/6)
        CD.append((1-aerfa[i])/(1+aerfa[i]))
    return Ex,En,He,CD

def calculate_certain(Ex,En,He,x):
    cer = []
    for i in range(len(Ex)):
        y = random.gauss(En[i],He[i])
        cer.append(math.e**(-pow((x - Ex[i]),2) / (2 * pow(y,2))))    
    return cer, cer.index(max(cer))  

def cal_muti_all(Ex,En,He,data,k):
    multi_sharp = np.zeros((k,data.shape[0],data.shape[1]))
    muti_count = np.zeros(k)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _, t = calculate_certain(Ex,En,He,data[i][j])
            multi_sharp[t][i][j] = 1
            muti_count[t] += 1
            
    return multi_sharp, muti_count

def cal_muti_par(Ex,En,He,data,k):
    multi_sharp = np.zeros((k,data.shape[0],data.shape[1]))
    muti_count = np.zeros(k)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for t in range(k):
                if data[i][j] >= Ex[t] - En[t] and data[i][j] <= Ex[t] + En[t]:
                    multi_sharp[t][i][j] = 1
                    muti_count[t] += 1
            
    return multi_sharp, muti_count

def t_gaosi_g(y,miu,seita):
    result = np.ones_like(y) * math.e
    temp = np.power((y - miu),2) * -1 / (2*seita**2+0.000000000001)
    result = np.power(result,temp)
    return result / (math.sqrt(2*math.pi)*seita)

def t_L_k(x,miu,seita,aerfa,k):
    fenzi = t_gaosi_g(x,miu[k],seita[k]) * aerfa[k]
    fenmu = np.zeros_like(fenzi)
    for n in range(len(miu)):
        fenmu += aerfa[n] * t_gaosi_g(x,miu[n],seita[n])
    return fenzi / (fenmu+0.000000000001)

def t_cal_J(y,y_count,aerfa,miu,seita,k):
    J = 0
    J_ln = np.zeros_like(y_count)
    for i in range(k):
        J_ln += aerfa[i] * t_gaosi_g(y,miu[i],seita[i])
    J_ln = np.log(J_ln)
    J = y_count * J_ln
    return np.sum(J)

def t_update_can(data,aerfa,miu,seita,k):
    for k_ in range(k):            
        miu[k_] = np.sum(t_L_k(data,miu,seita,aerfa,k_) * data) / (np.sum(t_L_k(data,miu,seita,aerfa,k_))+0.000000000001)
        seita[k_] = math.sqrt(np.sum(t_L_k(data,miu,seita,aerfa,k_) * np.power((data-miu[k_]),2)) / (np.sum(t_L_k(data,miu,seita,aerfa,k_)))+0.000000000001)
        aerfa[k_] = np.sum(t_L_k(data,miu,seita,aerfa,k_)) / len(data)
        
    return aerfa,miu,seita

def t_gaosi_tran(data,k,target,y,y_count):
    miu = []
    seita = []
    aerfa = []
    for i in range(k):
        miu.append((i+1)*data.max()/(k+1))
        seita.append(data.max())
        aerfa.append(1/k)  
    count = 1    
    while True:
        J = t_cal_J(y,y_count,aerfa,miu,seita,k)
        aerfa,miu,seita = t_update_can(data,aerfa,miu,seita,k)
        J_update = t_cal_J(y,y_count,aerfa,miu,seita,k)
        count += 1
        if abs(J_update - J)<target:
            break
    return aerfa,miu,seita

def t_cloud_tran(data,k,target,y,frequn):
    aerfa,miu,seita = t_gaosi_tran(data,k,target,y,frequn)
    Ex = []
    En = []
    He = []
    CD = []
    for i in range(len(aerfa)):
        Ex.append(miu[i])
        En.append((1+aerfa[i])*seita[i]/2)
        He.append((1-aerfa[i])*seita[i]/6)
        CD.append((1-aerfa[i])/(1+aerfa[i]+0.000000000001))
    return Ex,En,He,CD

def t_calculate_certain(Ex,En,He,x):
    cer = []
    for i in range(len(Ex)):
        y = random.gauss(En[i],He[i])
        cer.append(math.e**(-pow((x - Ex[i]),2) / (2 * pow(y,2)+0.000000000001)))
    return cer, cer.index(max(cer))

def t_cal_muti_all(Ex,En,He,data,k):
    multi_sharp = np.zeros((k,data.shape[0],data.shape[1]))
    muti_count = np.zeros(k)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            _, t = t_calculate_certain(Ex,En,He,data[i][j])
            multi_sharp[t][i][j] = 1
            muti_count[t] += 1
            
    return multi_sharp, muti_count

def t_cal_muti_par(Ex,En,He,data,k):
    multi_sharp = np.zeros((k,data.shape[0],data.shape[1]))
    muti_count = np.zeros(k)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for t in range(k):
                if data[i][j] >= Ex[t] - En[t] and data[i][j] <= Ex[t] + En[t]:
                    multi_sharp[t][i][j] = 1
                    muti_count[t] += 1
            
    return multi_sharp, muti_count

