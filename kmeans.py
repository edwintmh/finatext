#!/usr/bin/python
# vi: set fileencoding=utf-8 :

import numpy as np
import random
import itertools
import matplotlib.pyplot as plt


def initDataSet(input_file):
    input=open(input_file,'r')
    dataArray=input.readlines()
    for i in range(len(dataArray)):
        dataArray[i]=dataArray[i].split()
        dataArray[i]=dataArray[i][0].split(",")
        dataArray[i]=(float(dataArray[i][0]),float(dataArray[i][1]))
    input.close()
    return dataArray

def cluster_points(X, mu):#それぞれの点が属するクラスタを決める
    clusters  = {}
    
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(np.asarray(x)-np.asarray(mu[i[0]]))) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[mu[bestmukey][0],mu[bestmukey][1]].append(x)
        except KeyError:
            clusters[mu[bestmukey][0],mu[bestmukey][1]] = [x]
    return clusters
def reevaluate_centers(mu, clusters):#各クラスタに属する点の平均でクラスタの中心を決める
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        try:
            newmu.append((np.mean(clusters[k], axis = 0)[0],np.mean(clusters[k], axis = 0)[1]))
        except KeyError:
            newmu=(np.mean(clusters[k], axis = 0)[0],np.mean(clusters[k], axis = 0)[1])
    return newmu

def merge_and_split_centers(clusters,mu):#クラスタの数を増やすか減らすかの判断をする
    newmu = []
    keys = sorted(clusters.keys())
   
    for k in itertools.combinations(keys,2):
        if np.linalg.norm(np.asarray(k[0])-np.asarray(k[1]))<((np.std(clusters[k[0]]+np.std(clusters[k[1]])))):#両クラスタの中心の距離が両クラスタの標準偏差の和より小さい時なら1つのクラスタにまとめる(データの特性に合わせて変更することが可能)
            mean=(np.mean([i[0] for i in k]),np.mean([i[1] for i in k])) 
            try:
                newmu.append(mean)
            except KeyError:
                newmu=[mean]
            if k[0] in keys:
                keys.remove(k[0])
            if k[1] in keys:
                keys.remove(k[1])
            break
    for k in keys:
        meanContainer1=[]
        meanContainer2=[]
        for i in range(len(clusters[k])):
            try:
                meanContainer1.append(clusters[k][i][0])
            except KeyError:
                meanContainer1=clusters[k][i][0]
            try:
                meanContainer2.append(clusters[k][i][1])
            except KeyError:
                meanContainer2=clusters[k][i][1]
        devrange1=((np.mean(meanContainer1)-3*np.std(meanContainer1)),(np.mean(meanContainer1)+3*np.std(meanContainer1)))
        devrange2=((np.mean(meanContainer2)-3*np.std(meanContainer2)),(np.mean(meanContainer2)+3*np.std(meanContainer2)))
        if not devrange1[0]<=clusters[k][i][0]<=devrange1[1]:#ある点がクラスタの3倍の標準偏差の範囲外にあるなら、新しいクラスタの中心として扱う(データの特性に合わせて変更することが可能)
            if not devrange2[0]<=clusters[k][i][1]<=devrange2[1]:
                try:
                    newmu.append(clusters[k][i])
                except KeyError:
                    newmu=clusters[k][i]
        else:
            try:
                newmu.append(k)
            except KeyError:
                newmu=[k]
    return newmu
def find_centers(X):#メインプログラム。Xはデータ
    K=random.randint(3,7)
    oldmu= random.sample(X, K)
    mu= random.sample(X, K)
    while not has_converged(mu, oldmu):
        while not has_converged(mu, oldmu):
            oldmu = mu
            clusters = cluster_points(X, mu)
            mu = reevaluate_centers(oldmu, clusters)
        mu = merge_and_split_centers(clusters,mu)
    for i in mu:
        plotarray=[[],[]]
        for m in clusters[i]:
            try:
                plotarray[0].append(m[0])
            except KeyError:
                plotarray[0]=m[0]
            try:
                plotarray[1].append(m[1])
            except KeyError:
                plotarray[1]=m[0]
        plt.plot(plotarray[0],plotarray[1],'o')
        
        
    return(mu, clusters)

def has_converged(mu, oldmu):
    for i in range(len(mu)):
        if mu[i]==oldmu[i]:
            continue
        else:
            return False
    return True
   