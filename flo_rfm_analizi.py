import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#pip install lifetimes
df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.dropna(inplace=True)

df.info()

df.describe().T

df.head(10)

df.columns

df.isnull().sum()

######################################################################################################

df["Omnichannel_Total_Price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df["Omnichannel_Total_Value"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

######################################################################################################

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df.info()
######################################################################################################

df.groupby(["order_channel"]).agg({"master_id": ["count"],
                                   "Total_Price": ["sum"],
                                   "Total_Value": ["sum"]})

######################################################################################################

df.sort_values("Omnichannel_Total_Price", ascending=False).head(10)

######################################################################################################

df.sort_values("Omnichannel_Total_Value", ascending=False).head(10)

######################################################################################################

def Verı_on_ıs (Dataframe,Datesec = False):
    Dataframe.dropna(inplace=True)
    if Datesec == True:
        date_columns = Dataframe.columns[Dataframe.columns.str.contains("date")]
        Dataframe[date_columns] = Dataframe[date_columns].apply(pd.to_datetime)
    return Dataframe
Verı_on_ıs(df, Datesec=True)
df.info()
######################################################################################################
df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)
rfm = df.groupby('master_id').agg({'last_order_date': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Omnichannel_Total_Value': lambda Invoice: Invoice.sum(),
                                     'Omnichannel_Total_Price': lambda TotalPrice: TotalPrice.sum()})
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.head()
rfm.describe().T
######################################################################################################

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm.shape

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))
######################################################################################################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])



new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

#new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

new_df.to_csv("new_customers.csv")

rfm.to_csv("rfm.csv")
######################################################################################################

target = rfm[rfm["segment"].isin(["champions","loyal_customers"])].index
cust_ids = df[(df["master_id"].isin(target)) &(df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape


######################################################################################################

target = rfm[rfm["segment"].isin(["hibernating","cant_loose","need_attention"])].index
cust_ids = df[(df["master_id"].isin(target)) &(df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

######################################################################################################

def create_rfm(dataframe):
    # Veriyi Hazırlma
    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)


    # RFM METRIKLERININ HESAPLANMASI
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm["customer_id"] = dataframe["master_id"]
    rfm["recency"] = (analysis_date - dataframe["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = dataframe["order_num_total"]
    rfm["monetary"] = dataframe["customer_value_total"]

    # RF ve RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id", "recency","frequency","monetary","RF_SCORE","RFM_SCORE","segment"]]

rfm_df = create_rfm(df)