###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

# 1. İş Problemi (Business Problem)
# 2. Veriyi Anlama (Data Understanding)
# 3. Veri Hazırlama (Data Preparation)
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
# 5. RFM Skorlarının Hesaplanması (Calculating RFM Scores)
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi (Creating & Analysing RFM Segments)
# 7. Tüm Sürecin Fonksiyonlaştırılması

###############################################################
# 1. İş Problemi (Business Problem)
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.


###############################################################
# 2. Veriyi Anlama (Data Understanding)
###############################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

#veri setinde boş değerler işimize yaramayacağından sildik
df.isnull().sum()
df.dropna(inplace=True)

df.head()
df.shape

# essiz urun sayisi nedir?
df["Description"].nunique()

df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].nunique()

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

df.describe().T

###############################################################

df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

#Her bir faturaya ne kadar ödendiğini buluyoruz.
df["TotalPrice"] = df["Quantity"] * df["Price"]

#Geri iadeleri çıkartıyoruz.
df = df[~df["Invoice"].str.contains("C",na = False)]

df["InvoiceDate"].max() #2011-12-09

today_date = dt.datetime(2011, 12, 11)

type(today_date)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

rfm = rfm[rfm["monetary"] > 0]

rfm.describe().T

rfm.shape

###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=["5", "4", "3", "2", "1"])

#çok fazla tekrar eden frekans var ve bunun için rank methodunu kullandık
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=["1", "2", "3", "4", "5"])

rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels=["1", "2", "3", "4", "5"])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

###############################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating', #Derin uykuda
    r'[1-2][3-4]': 'at_Risk', #Riskliler
    r'[1-2]5': 'cant_loose', #Kaybedemiyeceklerimiz
    r'3[1-2]': 'about_to_sleep', #Uyumak Üzere
    r'33': 'need_attention', #Dikkat isteyenler
    r'[3-4][4-5]': 'loyal_customers', #Sadık müşteriler
    r'41': 'promising', #Umut vericiler
    r'51': 'new_customers', #Yeni Müşteri
    r'[4-5][2-3]': 'potential_loyalists', #Potansiyel sadık müşteriler
    r'5[4-5]': 'champions' #Şampiyonlar
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

###############################################################

new_df = pd.DataFrame()

new_df["loyal Customers_id"] = rfm[rfm["segment"] == "loyal_customers"].index

new_df.to_csv("new_customers.csv")

###############################################################