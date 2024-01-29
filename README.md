# CLTV Prediction with BG-NBD and Gamma-Gamma

1. Data Preparation
2. Expected Number of Transaction with BG-NBD Model
3. Expected Average Profit with Gamma-Gamma Model
4. Calculation of CLTV with BG-NBD and Gamma-Gamma Model
5. Creating Segments Based on CLTV
6. Functionalization of the work

# Data Preparation

An e-commerce company wants to segment its customers and determine marketing strategies according to these segments.

# Dataset Story

https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

The data set named Online Retail II includes the sales of a UK-based online store between 01/12/2009 and 09/12/2011.

# Variables

InvoiceNo: Invoice number. Unique number for each transaction, i.e. invoice. If it starts with C, the transaction is cancelled.

StockCode: Product code. Unique number for each product.

Description: Product name

Quantity: Number of products. It indicates how many of the products on the invoices were sold.

InvoiceDate: Invoice date and time.

UnitPrice: Product price (in Pounds Sterling)

CustomerID: Unique customer number

Country: Country name. The country where the customer lives.