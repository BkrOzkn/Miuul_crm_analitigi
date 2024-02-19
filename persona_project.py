#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.

import pandas as pd

pd.set_option("display.max_rows", None)

df = pd.read_csv("datasets/persona.csv")

df.head()

df.info

df.index

df.columns

df.shape

df["SOURCE"].nunique()

df["SOURCE"].value_counts()

df["PRICE"].nunique()

df["PRICE"].value_counts()

df["COUNTRY"].nunique()

df["COUNTRY"].value_counts()

df.groupby(["COUNTRY"]).agg({"PRICE": ["mean"]})

df["SOURCE"].value_counts()

df.groupby(["SOURCE"]).agg({"PRICE": ["count", "sum", "mean", "std"]})

df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": ["mean"]})

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": ["mean"]})

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE",ascending=False)

agg_df.head(25)

agg_df.reset_index(inplace=True)



agg_df["AGE"].info

agg_df["AGEa"] = pd.cut(df["AGE"], [0,18,23,30,40,df["AGE"].max()],
                        labels = ["0_18","19_23","24_30","31_40","41_70"])

agg_df.head()

agg_df["AGE"].dtype

agg_df["Customers_level_based"] = agg_df[["COUNTRY","SOURCE","SEX", "AGEa"]].agg(lambda x: "_".join(x).upper(),axis=1)

[agg_df.groupby([s]).agg({"PRICE": ["mean"]}) for s in agg_df.columns if s == "Customers_level_based"]


agg_df["Customers_level_based"].value_counts()

agg_df["Segment"] = pd.qcut(agg_df["PRICE"], 4,labels=["D", "C", "B", "A"])

agg_df.groupby(["Segment"]).agg({"PRICE" : "sum"})

agg_df.sort_values(by="PRICE")

new_user = "TUR_ANDROID_FEMALE_31_40"

agg_df[agg_df["Customers_level_based"] == new_user]








