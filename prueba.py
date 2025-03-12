import scipy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import random

def generarData(media:int, desv:int, n:int):
    data = np.random.normal(media, desv, n)
    return data.round(2)

def generarAnomalías(media:int,desv:int,n:int,nA:int):
    data=generarData(media,desv,n)
    for i in range(nA):
        datoRand=random.randint(0,len(data)-1)
        data[datoRand] = random.randint(round(data.mean(),0)-4,round(data.mean(),0)+4)
        print(f"Posición del data random: {datoRand}")
        print(f"Valor del dato random: {data[datoRand]}")
    return data

media=25
desv=0.1
n=50
nA=2
datosGenerados=generarData(media,desv,n)
datosModificados=generarAnomalías(media,desv,n,nA)

#ahora quiero ver de encontrar las anomalías entrenando con datosGenerados y luego mostrando los datos que tienen nA anomalías, datosModificados


clf = IsolationForest(random_state=0).fit(datosGenerados.reshape(-1,1))


