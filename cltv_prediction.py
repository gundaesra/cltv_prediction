#############################
# Required Libraries and Functions
#############################

import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

#####################################
# TASK 1: Preparing the Data
#####################################

# 1. Read the OmniChannel.csv data. Create a copy of the dataframe.

df_ = pd.read_csv("WLast_git_projects/cltv_prediction/flo_data_20k.csv")
df = df_.copy()

# 2. Define the outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
# Note: When calculating cltv, frequency values must be integer. Therefore, round the lower and upper limits with round().

def outlier_thresholds(dataframe, variable):
    quartile_1 = dataframe[variable].quantile(0.01)
    quartile_3 = dataframe[variable].quantile(0.99)

    interquantile_range = quartile_3 - quartile_1

    up_limit = round(interquantile_range * 1.5 + quartile_3)
    low_limit = round(interquantile_range * 1.5 + quartile_1)

    return up_limit, low_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    # low_limit replace
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

    # up_limit replace
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# 3. Suppresses the "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline",
# "customer_value_total_ever_online" variables if they have outliers.

outlier_variables = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for cols in outlier_variables:
    replace_with_thresholds(df, cols)

# 4. Omnichannel means that customers shop both online and offline platforms.
# Create new variables for each customer's total number of purchases and spending.

df["omnichannel_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["omnichannel_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# 5. Examine variable types. Change the type of variables expressing date to date.

date_cols = df.columns[df.columns.str.contains("date")]
df[date_cols] = df[date_cols].apply(pd.to_datetime)

####################### ##############
# TASK 2: Creating the CLTV Data Structure
####################### ##############

# 1. Take 2 days after the date of the last purchase in the data set as the analysis date.

today_date = dt.datetime(2021, 6, 2)

# 2.Create a new cltv dataframe containing customer_id, recency_cltv_weekly, T_weekly,
# frequency and monetary_cltv_avg values.

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]"))/7
cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["frequency"] = df["omnichannel_order"]
cltv_df["monetary_cltv_avg"] = df["omnichannel_value"] / df["omnichannel_order"]

#####################################
# TASK 3: BG/NBD, Establishing Gamma-Gamma Models, Calculating 6-month CLTV
#####################################

# 1. Install the BG/NBD model.

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# Estimate the expected purchases from customers within 3 months and add it to the cltv dataframe as exp_sales_3_month.

cltv_df["exp_sales_3_month"] = bgf.predict(12,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# Estimate the expected purchases from customers within 6 months and add it to the cltv dataframe as exp_sales_6_month.

cltv_df["exp_sales_6_month"] = bgf.predict(24,
                                           cltv_df["frequency"],
                                           cltv_df["recency_cltv_weekly"],
                                           cltv_df["T_weekly"])

# Check out the 10 people who will make the most purchases in the 3rd and 6th months.

cltv_df["exp_sales_3_month"].sort_values(ascending=False).head(10)

cltv_df["exp_sales_6_month"].sort_values(ascending=False).head(10)

# 2. Fit the Gamma-Gamma model. Estimate the average value that customers will leave and add it to the cltv dataframe
# as exp_average_value.

ggf = GammaGammaFitter(penalizer_coef= 0.001)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])

# 3. Calculate 6-month CLTV and add it to the dataframe with the name cltv.

cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                                              cltv_df["frequency"],
                                              cltv_df["recency_cltv_weekly"],
                                              cltv_df["T_weekly"],
                                              cltv_df["monetary_cltv_avg"],
                                              time=6,
                                              freq="W")

# Observe the 20 people with the highest CLTV values.

cltv_df["cltv"].sort_values(ascending=False).head(20)

#####################################
# TASK 4: Creating Segments Based on CLTV
#####################################

# 1. Divide all your customers into 4 groups (segments) according to 6-month CLTV and add the group names to the data
# set.
# Assign it with the name cltv_segment.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels = ["D", "C", "B", "A"])

# 2. Examine the recency, frequency and monetary averages of the segments.

cltv_df.groupby("cltv_segment")["recency_cltv_weekly", "frequency", "monetary_cltv_avg"].agg("mean")

#####################################
# BONUS: Functionalize the entire process.
#####################################

def create_cltv_df(dataframe):

    # Preparing the Data
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creating the CLTV data structure
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # Establishing the BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # Establishing the Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)


cltv_df.head(10)


