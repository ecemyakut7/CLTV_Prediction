
# Hafta 3 - Ödev 2

# BGNBD & GG ile CLTV Tahmini ve Sonuçların Uzak Sunucuya Gönderilmesi

# Bir e-ticaret sitesi müşteri aksiyonları için müşterilerinin CLTV değerlerine göre ileriye
# dönük bir projeksiyon yapılmasını istemektedir.
# Elinizdeki veriseti ile 1 aylık yada 6 aylık zaman periyotları içerisinde en çok gelir getirebilecek
# müşterileri tespit etmek mümkün müdür?


# Veri Seti

# Online Retail II isimli veri seti İngiltere merkezli online bir satış
# mağazasının 01/12/2009 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Bu şirketin ürün kataloğunda hediyelik eşyalar yer alıyor. Promosyon ürünleri olarak da düşünülebilir.
#Çoğu müşterisinin toptancı olduğu bilgisi de mevcut.

# Değişkenler

# InvoiceNo: Fatura Numarası
# Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder.
# StockCode: Ürün kodu
# Her bir ürün için eşsiz numara.
# Quantity: Ürün adedi
# Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# Description: Ürün ismi
# InvoiceDate: Fatura tarihi
# UnitPrice: Fatura fiyatı (Sterlin)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi


# Veri Setine Erişim

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_period_transactions
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

df_ = pd.read_excel("C:/Users/Hp/Desktop/DSMLBC6/Sena Ecem Yakut_3/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df.head()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Veri Ön İşleme

df.describe().T
df.isnull().sum().sort_values(ascending=False)
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Lifetime Veri Yapısının Hazırlanması

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (daha önce (cltv) analiz gününe göre, burada kullanıcı özelinde)
# T: Analiz tarihinden ne kadar süre önce ilk satın alma yapılmış. Haftalık.
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary_value: satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [ lambda date: (date.max() - date.min()).days,
                                                          lambda date: (today_date - date.min()).days],
                                         "Invoice": lambda num: num.nunique(),
                                         "TotalPrice" : lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

# Satın alma başına ortalama kazanç

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df = cltv_df[cltv_df["monetary"] > 0]

# Müşteri yaşı haftalık cinsten ifade edilmişti, bundan dolayı recency
# ve T'yi haftalığa çevirmeliyiz. Kendi içlerinde kaç haftalık müşteri
# olduğunu görürüz.

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df.head()

# BG-NBD Modelinin Kurulması

# Buy Till You Die
# Buy ve drop out prosesleriyle "expected number of transaction"ı hesaplar.
# Beklenen satışı (işlem sayısı) bulmayı hedefliyoruz.

bgf = BetaGeoFitter(penalizer_coef= 0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# Bütün kitlenin beklenen satış sayılarını tahmin etmek adına
# olasılık modelini kurduk.

# 1 hafta içinde en çok satın alma beklediğimiz  10 müşteri kimdir?

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False).head(10)

# Yukarıdaki fonksiyonun daha kısa hali:
# (1 hafta içerisinde beklediğim satın almaları ifade ediyor.)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])


# 6 ay içinde en çok satın alma beklediğimiz  10 müşteri kimdir?

cltv_df["expected_purc_6_month"] = bgf.predict(4 * 6,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])

cltv_df["expected_purc_6_month"].sort_values(ascending=False).head(10)


# 6 ay içinde tüm şirketin beklenen satış sayısı:

bgf.predict(4 * 6,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

# 6 ayda 10392 adet işlem (transaction) bekleniyor.

# Tahmin sonuçlarının değerlendirilmesi:

plot_period_transactions(bgf)
plt.show()

# 1 ve 12 ay içinde en çok satın alma beklediğimiz  10 müşteri kimdir?
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df["frequency"],
                                               cltv_df["recency"],
                                               cltv_df["T"])

cltv_df["expected_purc_1_month"].sort_values(ascending=False).head(10)


cltv_df["expected_purc_12_month"] = bgf.predict(4 * 12,
                                                cltv_df["frequency"],
                                                cltv_df["recency"],
                                                cltv_df["T"])

cltv_df["expected_purc_12_month"].sort_values(ascending=False).head(10)


# 1 ay ve 12 ay içinde tüm şirketin beklenen satış sayısı nedir?

bgf.predict(4,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()

bgf.predict(4 * 3,
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sum()



# Gamma- Gamma Modelinin Kurulması

# Gamma-Gamma modelindeki amacımız "Expected Average Profit"ini bulmak.
# Beklenen ortalama karlılık bilgisini bulmayı hedefliyoruz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

# GG modelini kurmak için istenen frekans ve monetary bilgisidir.

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

# Her bir müşteri için karlılığı veri setimize ekliyoruz.

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])


cltv_df.head()

# BG-NBD ve GG Modelleri ile CLTV'nin Hesaplanması

# Tüm müşteriler için lifetime value hesabı:

# Görev 1: 6 aylık CLTV Prediction

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time= 6, # 6 aylık
                                   freq="W", # T'nin frekans bilgisi(haftalık)
                                   discount_rate=0.01)

cltv = cltv.reset_index()

cltv.head()

cltv.sort_values(by= "clv", ascending=False).head(50)

# Customer ID'lerine göre CLTV değerlerini büyükten küçüğe doğru sıraladık. Elde edilen sonuçlara göre 14646 Customer
# ID'sine sahip müşterinin, şirket için en değerli müşteri olduğunu söyleyebiliriz.


# Görev 2: Farklı zaman periyotlarından oluşan CLTV analizi

# 1 Aylık CLTV

cltv_1 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time= 1, # 1 aylık
                                   freq="W", # T'nin frekans bilgisi(haftalık)
                                   discount_rate=0.01)


# 12 Aylık CLTV

cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time= 12, # 12 aylık
                                   freq="W", # T'nin frekans bilgisi(haftalık)
                                   discount_rate=0.01)


# 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.

cltv_1 = cltv_1.reset_index()
cltv_1.sort_values(by="clv", ascending=False).head(10)

cltv_12 = cltv_12.reset_index()
cltv_12.sort_values(by="clv", ascending=False).head(10)

# Fark var mı? Varsa sizce neden olabilir?

# Fark vardır. Bunun nedeni 1 aya kıyasla, 12 ay içerisinde aynı ID'li müşterinin gelip alışveriş yapma olasılığının
# artması, zaman içerisinde yaptığı harcamaların fazlalaşması olabilir.


cltv_final = cltv_df.merge(cltv, on= "Customer ID", how = "left")

cltv_final.sort_values(by="clv", ascending=False).head(10)

# Değerleri standartlaştırma:

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])


cltv_final.sort_values(by="scaled_clv", ascending=False).head()

# Clv: Müşterinin bırakacağı değer olarak söylenebilir.

# Görev 3: Segmentasyon ve Aksiyon Önerileri

# 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba
# (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"],4, labels=["D","C","B","A"])

cltv_final.sort_values(by="scaled_clv", ascending=False).head(5)


cltv_final.groupby("segment")["expected_purc_1_week", "expected_purc_6_month","expected_purc_1_month"
                   ,"expected_purc_12_month", "expected_average_profit", "clv","scaled_clv"].agg({"count", "mean", "sum"})


# A ile D segmentlerine bakarsak;

# ikisinin de aylar içerisinde yapılan işlem sayılarında artış bulunmaktadır. 6 aylık beklenen ortalama karlılık
# A segmentinde D segmentine kıyasla daha fazladır.

# D segmenti, tek seferlik gelen müşteriler olarak görülebilir. Böyle müşterileri yakalamak, elde tutmak, sık gelen
# müşterilere kıyasla daha zordur. Bu tarz müşterilerin devamlı müşteriler olmasını sağlamak için örneğin;
# ilk 3 alışverişlerine özel indirim uygulaması yapılabilir.
# Kupon kullanılabilir, mesela kupon 10 bölümden oluşsun; 10 alışveriş yapan ve kuponunu dolduran müşteri için
# harcamalarına göre bir değerde hediye verilebilir. Böylece müşterinin ayağı alışır ve mağaza ilk tercihleri
# arasında yer alır.


# A segmentindeki müşteriyi elde tutmak, D'yi yakalamaktan daha kolaydır. Bu müşteri mağaza/marka için oldukça
# değerlidir. Müşterinin elde tutulması için özel ayrıcalıklar sağlanabilir. Örneğin devamlı müşterilere özel
# kampanyalardan ilk haberdar olma ve ilk yararlanma fırsatı mesajla veya e-mail ile haber verilebilir. Müşteri
# ile sıkı bir iletişim halinde olmak, ürünle ilgili bir sıkıntısında yardımcı olmaya çalışmak ve sorunu çözmek
# müşteriye güven verir ve başka mağaza arayışında bulunmaz.