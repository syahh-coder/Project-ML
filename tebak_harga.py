import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

df = pd.read_csv("data_rumah.csv")
sample = df.head()

harga= df.iloc[:, :-1]
luas_tanah= df.iloc[:, 1]

x_train,x_test,y_train,y_test = train_test_split(harga,luas_tanah,test_size=0.25,random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)

luas_tanah_pred = regressor.predict(x_test)

# plt.scatter(x_train,y_train)
# plt.plot(x_train,regressor.predict(x_train))
# plt.xlabel("harga rumah")
# plt.ylabel("luas tanah")
# plt.show()


input_harga = int(input("masukkan harga (dalam M):"))
luas_tanah_prediksi = regressor.predict([[input_harga]])
print(f"prediksi luas tanah = {luas_tanah_prediksi} km2")
