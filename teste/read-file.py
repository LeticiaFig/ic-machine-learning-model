import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Group1_BancoTCE_24h_FUP_Impulsividade.csv")
# print(df)
# print(df.head(n=10))
# print(df.tail())
# print(df["age"].unique())
# print(df["sex"].value_counts(normalize=True))
# print(df.groupby("sex").mean()["BIS motor"].sort_values())


def zeroColumn(bairro):
    return 0

# df2 = df["Angiotensin II"].apply(zeroColumn)
# print(df2)

# df2 = df.head()
# df2 = df2.replace({"age": {37: np.nan}})
# print(df2)
# print(df2.dropna()) #remoção de quaisquer linhas ou colunas que possuem um np.nan
# print(df2.fillna(99)) #Preencher todos os valores NaN por um outro específico
# print(df2.isna()) #is NAN

# df["age"].plot.hist()
# df["age"].value_counts().plot.barh()

plt.style.use('ggplot')
df.plot.scatter(x='age', y='Copeptin')
df["age"].value_counts().plot.pie()
plt.show()