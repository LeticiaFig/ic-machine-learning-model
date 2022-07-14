import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

notas = pd.Series([2,7,5,10,6])
print(notas)
print(notas.values)
notas = pd.Series([2,7,5,10,6], index=["Wilfred", "Abbie", "Harry", "Julia", "Carrie"])
print(notas)
print(notas["Julia"])

print("Média:", notas.mean())
print("Desvio padrão:", notas.std())

print(notas.describe()) 
print(notas**2)
print(np.log(notas))

df = pd.DataFrame({'Aluno' : ["Wilfred", "Abbie", "Harry", "Julia", "Carrie"],
                   'Faltas' : [3,4,2,1,4],
                   'Prova' : [2,7,5,10,6],
                   'Seminário': [8.5,7.5,9.0,7.5,8.0]})

print('\n\n\nData Frame:\n', df)
print(df.dtypes)
print(df.columns)
print(df["Seminário"])
print(df.describe()) #O que é 25%, 50%??
df.sort_values(by="Seminário")
print(df)
print(df.loc[3])
print(df[df["Seminário"] > 8.0] & (df["Prova"] > 3))