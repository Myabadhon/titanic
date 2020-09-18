import pandas as pd
import matplotlib.pyplot as plt

male_color = "#28B463"
female_color = "#E74C3C"

df = pd.read_csv("train.csv")
fig = plt.figure(figsize=(18,6))

plt.subplot2grid((3,4),(0,0))
df.Survived.value_counts(normalize=True).plot(kind = "bar", alpha = 0.5)
plt.title("Survived")

plt.subplot2grid((3,4),(0,1))
df.Survived[df.Sex == "male"].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5,color=male_color)
plt.title("Men Survived")

plt.subplot2grid((3,4),(0,2))
df.Survived[df.Sex == "female"].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5,color = female_color)
plt.title("Women Survived")

plt.subplot2grid((3,4),(0,3))
df.Sex[df.Survived == 1].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5,color = [female_color,male_color])
plt.title("Sex Of Survived")

plt.subplot2grid((3,4),(1,0),colspan=4)
for x in [1,2,3]:
    df.Survived[df.Pclass == x].plot(kind = "kde")
plt.title("Class wrt Survived")
plt.legend(("1st","2nd","3rd"))

plt.subplot2grid((3,4),(2,0))
df.Survived[(df.Sex == "male") & (df.Pclass == 1)].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5)
plt.title("Rich Men Survived")

plt.subplot2grid((3,4),(2,1))
df.Survived[(df.Sex == "male") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5)
plt.title("Poor Men Survived")

plt.subplot2grid((3,4),(2,2))
df.Survived[(df.Sex == "female") & (df.Pclass == 1)].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5,color= female_color)
plt.title("Rich Women Survived")

plt.subplot2grid((3,4),(2,3))
df.Survived[(df.Sex == "female") & (df.Pclass == 3)].value_counts(normalize=True).plot(kind = "bar", alpha = 0.5, color = female_color)
plt.title("Poor Women Survived")


plt.show()