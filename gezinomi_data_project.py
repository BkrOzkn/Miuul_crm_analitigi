#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Gezinomi yaptığı satışların bazı özelliklerini kullanarak seviye tabanlı (level based) yeni satış tanımları
# oluşturmak ve bu yeni satış tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
# Örneğin: Antalya’dan Herşey Dahil bir otele yoğun bir dönemde gitmek isteyen bir müşterinin ortalama ne kadar kazandırabileceği belirlenmek isteniyor.

import pandas as pd
pd.set_option("display.max_rows", None)

df = pd.read_excel("miuul_gezinomi.xlsx")

df.head(25)

df.shape

df.columns

df["SaleCityName"].value_counts()

df["ConceptName"].value_counts()

df.groupby(["SaleCityName"]).agg({"Price": ["sum"]})

df.groupby(["ConceptName"]).agg({"Price": ["sum"]})

df.groupby(["SaleCityName"]).agg({"Price": ["mean"]})

df.loc[(df["SaleCityName"] == "Antalya") , "Price"].sum()

df.groupby(["ConceptName"]).agg({"Price": ["sum"]})

df.groupby(["SaleCityName","ConceptName",]).agg({"Price": ["sum"],
                                               "SaleId": ["count"]})

df["SaleCheckInDayDiff"] = df["SaleCheckInDayDiff"].astype("category")

df["SaleCheckInDayDiff"].nunique()

df["New_Score"] = pd.cut(df["SaleCheckInDayDiff"], [0,7,30,90,df["SaleCheckInDayDiff"].max()],
                        labels = ["Last_Minuters","Potential Planners","Planners","Early Bookers"])

df.groupby(["SaleCityName","ConceptName","New_Score"]).agg({"Price": ["mean","count"]})

a = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)

a.head(100)

df.index

df["SaleCityName"] = df.index

df["ConceptName"] = df.index

df["New_Score"] = df.index

a.reset_index(inplace=True)

df["SaleCheckInDayDiff"]

df.head()
pd.set_option('display.max_columns', None)

df.index = df["SaleCityName"]

df["SaleCheckInDayDiff"]

df.columns

df.reset_index(inplace=True)

a["sales_level_based"] = a[["SaleCityName" , "ConceptName" , "Seasons"]].agg(lambda x: "_".join(x).upper(),axis=1)

df["sales_level_based"]


a["Segment"] = pd.qcut(df["Price"], 4,labels=["D", "C", "B", "A"])

a.groupby(["Segment"]).agg({"Price" : "sum"})

a.sort_values(by="Price")

new_user = "MUĞLA_HERŞEY DAHIL_LOW"

a[a["sales_level_based"] == new_user]

pd.set_option("display.max_columns", None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)






