---
layout: post
title:  "Autoencoders & Data Augmentation"
date:   2018-02-24 22:26:00 +0800
categories: Default
tags: Imbalanced Oversampling Data-Augmentation Deep-Learning Autoencoders
---

## Data Augmentation Using Autoencoder Generated Data

---

_24th Feb 2018_<br>
_Cheo Rui Ming (SG)_<br>
_Refer [here](https://github.com/CheoRM/cheorm.github.io/tree/master/accompanying_blog_notes/Auto-Credit-Card-Fraud) for the accompanying notes. This blog is an extension of the [blog](https://cheorm.github.io/2018/02/21/GAN-Credit-Card-Fraud) on the usage of GANs to augment data._<br>

---

Following the usage of GANs to augment a highly imbalanced dataset, a classical autoencoder was implemented with its performance evaluated.


### Preliminary Overview of Data

From the given dataset, the aim is to detect frauds regardless of time, otherwise the underlying detection method would not be able to pick up frauds if fraudsters change their time of operation.
- Time will be dropped for the sake of creating a model that can detect fraudulent transactions regardless of the time of day

It may be thought that 'Amount' should also be removed since fraudulent transaction amounts may be a subset of normal transaction amounts. Regardless, we shall perform statistical tests to determine feature relevance and to verify whether unique transaction 'Amount' patterns in fraudulent ones exist.


```python
### Overview of data in 'credit' ###

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('max_info_columns', 1000)
pd.set_option('max_info_rows', 1000)

# 1st 5 rows
credit.head()
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>0.090794</td>
      <td>-0.551600</td>
      <td>-0.617801</td>
      <td>-0.991390</td>
      <td>-0.311169</td>
      <td>1.468177</td>
      <td>-0.470401</td>
      <td>0.207971</td>
      <td>0.025791</td>
      <td>0.403993</td>
      <td>0.251412</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>-0.166974</td>
      <td>1.612727</td>
      <td>1.065235</td>
      <td>0.489095</td>
      <td>-0.143772</td>
      <td>0.635558</td>
      <td>0.463917</td>
      <td>-0.114805</td>
      <td>-0.183361</td>
      <td>-0.145783</td>
      <td>-0.069083</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>0.207643</td>
      <td>0.624501</td>
      <td>0.066084</td>
      <td>0.717293</td>
      <td>-0.165946</td>
      <td>2.345865</td>
      <td>-2.890083</td>
      <td>1.109969</td>
      <td>-0.121359</td>
      <td>-2.261857</td>
      <td>0.524980</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>-0.054952</td>
      <td>-0.226487</td>
      <td>0.178228</td>
      <td>0.507757</td>
      <td>-0.287924</td>
      <td>-0.631418</td>
      <td>-1.059647</td>
      <td>-0.684093</td>
      <td>1.965775</td>
      <td>-1.232622</td>
      <td>-0.208038</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>0.753074</td>
      <td>-0.822843</td>
      <td>0.538196</td>
      <td>1.345852</td>
      <td>-1.119670</td>
      <td>0.175121</td>
      <td>-0.451449</td>
      <td>-0.237033</td>
      <td>-0.038195</td>
      <td>0.803487</td>
      <td>0.408542</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dtype; checking for nulls
credit.info(null_counts=True)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 284807 entries, 0 to 284806
    Data columns (total 31 columns):
    Time      284807 non-null float64
    V1        284807 non-null float64
    V2        284807 non-null float64
    V3        284807 non-null float64
    V4        284807 non-null float64
    V5        284807 non-null float64
    V6        284807 non-null float64
    V7        284807 non-null float64
    V8        284807 non-null float64
    V9        284807 non-null float64
    V10       284807 non-null float64
    V11       284807 non-null float64
    V12       284807 non-null float64
    V13       284807 non-null float64
    V14       284807 non-null float64
    V15       284807 non-null float64
    V16       284807 non-null float64
    V17       284807 non-null float64
    V18       284807 non-null float64
    V19       284807 non-null float64
    V20       284807 non-null float64
    V21       284807 non-null float64
    V22       284807 non-null float64
    V23       284807 non-null float64
    V24       284807 non-null float64
    V25       284807 non-null float64
    V26       284807 non-null float64
    V27       284807 non-null float64
    V28       284807 non-null float64
    Amount    284807 non-null float64
    Class     284807 non-null int64
    dtypes: float64(30), int64(1)
    memory usage: 67.4 MB
    

```python
# Descriptive statistics
credit.describe(include='all')
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>V10</th>
      <th>V11</th>
      <th>V12</th>
      <th>V13</th>
      <th>V14</th>
      <th>V15</th>
      <th>V16</th>
      <th>V17</th>
      <th>V18</th>
      <th>V19</th>
      <th>V20</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>284807.000000</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>2.848070e+05</td>
      <td>284807.000000</td>
      <td>284807.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>94813.859575</td>
      <td>1.165980e-15</td>
      <td>3.416908e-16</td>
      <td>-1.373150e-15</td>
      <td>2.086869e-15</td>
      <td>9.604066e-16</td>
      <td>1.490107e-15</td>
      <td>-5.556467e-16</td>
      <td>1.177556e-16</td>
      <td>-2.406455e-15</td>
      <td>2.239751e-15</td>
      <td>1.673327e-15</td>
      <td>-1.254995e-15</td>
      <td>8.176030e-16</td>
      <td>1.206296e-15</td>
      <td>4.913003e-15</td>
      <td>1.437666e-15</td>
      <td>-3.800113e-16</td>
      <td>9.572133e-16</td>
      <td>1.039817e-15</td>
      <td>6.406703e-16</td>
      <td>1.656562e-16</td>
      <td>-3.444850e-16</td>
      <td>2.578648e-16</td>
      <td>4.471968e-15</td>
      <td>5.340915e-16</td>
      <td>1.687098e-15</td>
      <td>-3.666453e-16</td>
      <td>-1.220404e-16</td>
      <td>88.349619</td>
      <td>0.001727</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47488.145955</td>
      <td>1.958696e+00</td>
      <td>1.651309e+00</td>
      <td>1.516255e+00</td>
      <td>1.415869e+00</td>
      <td>1.380247e+00</td>
      <td>1.332271e+00</td>
      <td>1.237094e+00</td>
      <td>1.194353e+00</td>
      <td>1.098632e+00</td>
      <td>1.088850e+00</td>
      <td>1.020713e+00</td>
      <td>9.992014e-01</td>
      <td>9.952742e-01</td>
      <td>9.585956e-01</td>
      <td>9.153160e-01</td>
      <td>8.762529e-01</td>
      <td>8.493371e-01</td>
      <td>8.381762e-01</td>
      <td>8.140405e-01</td>
      <td>7.709250e-01</td>
      <td>7.345240e-01</td>
      <td>7.257016e-01</td>
      <td>6.244603e-01</td>
      <td>6.056471e-01</td>
      <td>5.212781e-01</td>
      <td>4.822270e-01</td>
      <td>4.036325e-01</td>
      <td>3.300833e-01</td>
      <td>250.120109</td>
      <td>0.041527</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-5.640751e+01</td>
      <td>-7.271573e+01</td>
      <td>-4.832559e+01</td>
      <td>-5.683171e+00</td>
      <td>-1.137433e+02</td>
      <td>-2.616051e+01</td>
      <td>-4.355724e+01</td>
      <td>-7.321672e+01</td>
      <td>-1.343407e+01</td>
      <td>-2.458826e+01</td>
      <td>-4.797473e+00</td>
      <td>-1.868371e+01</td>
      <td>-5.791881e+00</td>
      <td>-1.921433e+01</td>
      <td>-4.498945e+00</td>
      <td>-1.412985e+01</td>
      <td>-2.516280e+01</td>
      <td>-9.498746e+00</td>
      <td>-7.213527e+00</td>
      <td>-5.449772e+01</td>
      <td>-3.483038e+01</td>
      <td>-1.093314e+01</td>
      <td>-4.480774e+01</td>
      <td>-2.836627e+00</td>
      <td>-1.029540e+01</td>
      <td>-2.604551e+00</td>
      <td>-2.256568e+01</td>
      <td>-1.543008e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>54201.500000</td>
      <td>-9.203734e-01</td>
      <td>-5.985499e-01</td>
      <td>-8.903648e-01</td>
      <td>-8.486401e-01</td>
      <td>-6.915971e-01</td>
      <td>-7.682956e-01</td>
      <td>-5.540759e-01</td>
      <td>-2.086297e-01</td>
      <td>-6.430976e-01</td>
      <td>-5.354257e-01</td>
      <td>-7.624942e-01</td>
      <td>-4.055715e-01</td>
      <td>-6.485393e-01</td>
      <td>-4.255740e-01</td>
      <td>-5.828843e-01</td>
      <td>-4.680368e-01</td>
      <td>-4.837483e-01</td>
      <td>-4.988498e-01</td>
      <td>-4.562989e-01</td>
      <td>-2.117214e-01</td>
      <td>-2.283949e-01</td>
      <td>-5.423504e-01</td>
      <td>-1.618463e-01</td>
      <td>-3.545861e-01</td>
      <td>-3.171451e-01</td>
      <td>-3.269839e-01</td>
      <td>-7.083953e-02</td>
      <td>-5.295979e-02</td>
      <td>5.600000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>84692.000000</td>
      <td>1.810880e-02</td>
      <td>6.548556e-02</td>
      <td>1.798463e-01</td>
      <td>-1.984653e-02</td>
      <td>-5.433583e-02</td>
      <td>-2.741871e-01</td>
      <td>4.010308e-02</td>
      <td>2.235804e-02</td>
      <td>-5.142873e-02</td>
      <td>-9.291738e-02</td>
      <td>-3.275735e-02</td>
      <td>1.400326e-01</td>
      <td>-1.356806e-02</td>
      <td>5.060132e-02</td>
      <td>4.807155e-02</td>
      <td>6.641332e-02</td>
      <td>-6.567575e-02</td>
      <td>-3.636312e-03</td>
      <td>3.734823e-03</td>
      <td>-6.248109e-02</td>
      <td>-2.945017e-02</td>
      <td>6.781943e-03</td>
      <td>-1.119293e-02</td>
      <td>4.097606e-02</td>
      <td>1.659350e-02</td>
      <td>-5.213911e-02</td>
      <td>1.342146e-03</td>
      <td>1.124383e-02</td>
      <td>22.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>139320.500000</td>
      <td>1.315642e+00</td>
      <td>8.037239e-01</td>
      <td>1.027196e+00</td>
      <td>7.433413e-01</td>
      <td>6.119264e-01</td>
      <td>3.985649e-01</td>
      <td>5.704361e-01</td>
      <td>3.273459e-01</td>
      <td>5.971390e-01</td>
      <td>4.539234e-01</td>
      <td>7.395934e-01</td>
      <td>6.182380e-01</td>
      <td>6.625050e-01</td>
      <td>4.931498e-01</td>
      <td>6.488208e-01</td>
      <td>5.232963e-01</td>
      <td>3.996750e-01</td>
      <td>5.008067e-01</td>
      <td>4.589494e-01</td>
      <td>1.330408e-01</td>
      <td>1.863772e-01</td>
      <td>5.285536e-01</td>
      <td>1.476421e-01</td>
      <td>4.395266e-01</td>
      <td>3.507156e-01</td>
      <td>2.409522e-01</td>
      <td>9.104512e-02</td>
      <td>7.827995e-02</td>
      <td>77.165000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>172792.000000</td>
      <td>2.454930e+00</td>
      <td>2.205773e+01</td>
      <td>9.382558e+00</td>
      <td>1.687534e+01</td>
      <td>3.480167e+01</td>
      <td>7.330163e+01</td>
      <td>1.205895e+02</td>
      <td>2.000721e+01</td>
      <td>1.559499e+01</td>
      <td>2.374514e+01</td>
      <td>1.201891e+01</td>
      <td>7.848392e+00</td>
      <td>7.126883e+00</td>
      <td>1.052677e+01</td>
      <td>8.877742e+00</td>
      <td>1.731511e+01</td>
      <td>9.253526e+00</td>
      <td>5.041069e+00</td>
      <td>5.591971e+00</td>
      <td>3.942090e+01</td>
      <td>2.720284e+01</td>
      <td>1.050309e+01</td>
      <td>2.252841e+01</td>
      <td>4.584549e+00</td>
      <td>7.519589e+00</td>
      <td>3.517346e+00</td>
      <td>3.161220e+01</td>
      <td>3.384781e+01</td>
      <td>25691.160000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
# Validate only fraud and non-fraud classes
credit['Class'].value_counts()
```


    0    284315
    1       492
    Name: Class, dtype: int64
    

As defined from source: 'Time' is the seconds elapsed between each transaction and the first transaction in the dataset. The last transaction was 172,792 seconds after the first was made. This computes to approximately 2 days worth of transaction data.

In this case, it is unknown during what time of the day the transactions actually began. Then, it is not possible to say that the first data point of 'Time' value == 0 imply transactions began at midnight. The column can at most inform us how close the transactions were made in between 2 fraudulent ones.

Since the aim is to predict frauds regardless of transaction time, 'Time' was dropped earlier.

#### Managing train-test split data

As 'V1' to 'V28' were obtained via PCA-transformation of the original features to anonymise the clientele, only 'Amount' needs to be standardised for the sake of consistency of the entire dataset

Also, the 'Amount' columns of both the training and test set are standardised based on the training set's mean and standard deviations. This is done so to model reality where the characteristics of the test set is unknown to us. If the training set is a good representative of reality, then there should not be any differences between itself and the unlabelled (fraud/non-fraud) data.


    Number of frauds in training data: 379 out of 213605 cases (0.1774303036% fraud)
    Number of frauds in test data: 113 out of 71202 cases (0.1587034072% fraud)
    

Examine the total value of 'Amount' in all the fraud transactions in both the training and test datasets.


    Total "Amount" in training set: $42513.46 | Total "Amount" in test set: $17614.51
    

### Exploratory Data Analysis (Training Set Only)

A PCA plot was implemented to observe for the differences between fraudulent and normal transactions.


    explained variance ratio (first two components): [ 0.12503723  0.10004168]


<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_18_1.png" width="100%" height="100%" />


Clealy some form of distinction between fraud and non-fraud data exits. However, since the fraud data is a subset of the non-frauds (i.e. fraudulent clusters within the area of the non-fraud clusters), it will not be unexpected to find the precision of identifying fraud data to be exceptionally poor when maximising recall.


```python
# Kernel Density Distribution subplots against target - 'Class'   
```


<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_20_0.png" width="100%" height="100%" />


```python
# Histogram Distribution subplots against target - 'Class'  
```


<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_21_0.png" width="100%" height="100%" />


Based on the distribution graphs above, it appears that for certain features, the distribution of fraudulent cases are different from non-fraudulent cases. However, the differences are difficult to make definitive judgements based on visual inspection.

For the sake of reaching a robust conclusion (on which features are important to distinguish normal transactions from fradulent ones), we will perform equality of distribution tests. In this case, the Wilcoxon Rank-Sum Test is chosen here since outliers were found in the data. This will be used to check for equality of the distributions of each feature, between fraud and non-fraud cases.

    
     Wilcoxon Rank-Sum Relevant Features:  ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
     'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V24', 'V25', 'V27', 'V28']
    Total number of features selected: 23
    
    Unselected features: ['V15', 'V23', 'V22', 'V26', 'V13', 'Amount']
    

#### Results from Wilcoxon Rank-Sum Tests:
* The null hypothesis is rejected at 5% significance for all features
* Features that are removed: 'V15', 'V22', 'V26', 'V13', 'V23', 'Amount'

The top 6 most correlated feature pairs, each feature selected by the Wilcoxon tests, were plotted to visualise the scatter patterns.


```python
# Obtain the top 6 most wilcoxon-selected correlated variables' to observe their scatter pair plot relations
```


<div style="overflow-x:auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>pearson_rho</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>V5</th>
      <th>V7</th>
      <td>0.030143</td>
    </tr>
    <tr>
      <th>V6</th>
      <th>V5</th>
      <td>0.016899</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">V7</th>
      <th>V3</th>
      <td>0.016640</td>
    </tr>
    <tr>
      <th>V6</th>
      <td>0.016007</td>
    </tr>
    <tr>
      <th>V2</th>
      <th>V1</th>
      <td>0.015015</td>
    </tr>
    <tr>
      <th>V1</th>
      <th>V7</th>
      <td>0.012172</td>
    </tr>
  </tbody>
</table>
</div>


<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_27_0.png" width="100%" height="100%" />


#### Autoencoder Implementation of Data Augmentation

A simple autoencoder network was implemented to generate data for augementing the dataset. With minimal tuning, it was executed for 6300 epochs. The epoch with the lowest MSE loss, frame of epochs with stabilised losses and the final epoch had their generated data retrieved for augmenting the training set's minority (fraud) class.


<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_33_0.png" width="100%" height="100%" />


It seems that after slightly less than 500 epochs, the autoencoder generating synthetic fraud transactions stabilised between 25-28 MSE loss values. No observable MSE loss deviations from this range were observed by the final epoch at 6300.

The steady epoch frame was next found using 2 s.d. from the median loss value instead of the intended +/-0.75 due to the fluctuations of MSE loss value across all epochs. No steady frame of epochs could be found when a limit of 0.75 of 1 s.d. MSE losses was set. This was progressively found by relaxing 0.25 of 1 s.d. at a time.

Since there was no convergence to 0 losses across epochs, the epoch with the minimum MSE loss was evaluated in addition to the steady and final epochs.


    Lowest Loss Fraud Epoch: 3376
    Steady epoch frame found at epoch 1207 as final
    Fraud Steady Epoch: 1207
    

Retrieving generated data from the relevant epochs, the dataset was next augmented to train the classifer (stochastic gradient descent classifer of 'scikit-learn' was implemented across all models). Classification results are shown below.

    
     ############################################# LOWEST MSE #############################################
    Prediction Score / Lowest MSE:  0.974719811241 
    
    Confusion Matrix / Lowest MSE:  
     [[69300  1789]
     [   11   102]] 
    
    Classification Report / Lowest MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.97      0.90      0.99      0.23      0.06     71089
              1       0.05      0.90      0.97      0.10      0.23      0.05       113
    
    avg / total       1.00      0.97      0.90      0.99      0.23      0.06     71202
    
    =======================================================================================================
    Prediction Score (Standardised) / Lowest MSE:  0.965548720542 
    
    Confusion Matrix (Standardised) / Lowest MSE:  
     [[68647  2442]
     [   11   102]] 
    
    Classification Report (Standardised) / Lowest MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.97      0.90      0.98      0.20      0.04     71089
              1       0.04      0.90      0.97      0.08      0.20      0.04       113
    
    avg / total       1.00      0.97      0.90      0.98      0.20      0.04     71202
    
    
---

    
     ############################################# STEADY MSE #############################################
    Prediction Score / Steady MSE:  0.938667453161 
    
    Confusion Matrix / Steady MSE:  
     [[66731  4358]
     [    9   104]] 
    
    Classification Report / Steady MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.94      0.92      0.97      0.15      0.03     71089
              1       0.02      0.92      0.94      0.05      0.15      0.02       113
    
    avg / total       1.00      0.94      0.92      0.97      0.15      0.03     71202
    
    =======================================================================================================
    Prediction Score (Standardised) / Steady MSE:  0.875270357574 
    
    Confusion Matrix (Standardised) / Steady MSE:  
     [[62214  8875]
     [    6   107]] 
    
    Classification Report (Standardised) / Steady MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.88      0.95      0.93      0.11      0.01     71089
              1       0.01      0.95      0.88      0.02      0.11      0.01       113
    
    avg / total       1.00      0.88      0.95      0.93      0.11      0.01     71202
    
    
---

    
     ############################################# FINAL EPOCH #############################################
    Prediction Score / Final MSE:  0.900536501784 
    
    Confusion Matrix / Final MSE:  
     [[64014  7075]
     [    7   106]] 
    
    Classification Report / Final MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.90      0.94      0.95      0.12      0.02     71089
              1       0.01      0.94      0.90      0.03      0.12      0.01       113
    
    avg / total       1.00      0.90      0.94      0.95      0.12      0.02     71202
    
    =======================================================================================================
    Prediction Score (Standardised) / Final MSE:  0.843417319738 
    
    Confusion Matrix (Standardised) / Final MSE:  
     [[59945 11144]
     [    5   108]] 
    
    Classification Report (Standardised) / Final MSE: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.84      0.96      0.91      0.10      0.01     71089
              1       0.01      0.96      0.84      0.02      0.10      0.01       113
    
    avg / total       1.00      0.84      0.96      0.91      0.10      0.01     71202
    
    
The scatter patterns were next plotted for the top 6 correlated feature pairs to observe for changes in the data.

```python
# Plot scatter plots of these 6 correlation pairs
corr_top6 = (('V5', 'V7'), ('V6', 'V5'), ('V7', 'V3'), ('V7', 'V6'), ('V2', 'V1'), ('V1', 'V7'))
```

<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_41_0.png" width="100%" height="100%" />

<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_42_1.png" width="100%" height="100%" />

<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_43_1.png" width="100%" height="100%" />

<img src="https://cheorm.github.io/accompanying_blog_notes/Auto-Credit-Card-Fraud/output_44_1.png" width="100%" height="100%" />


While the recall on fraudulent cases are very high, it makes no sense to augment data using autoencoders. Though no visible changes in the scatter relations were observed above _(the changes are exceptionally minor to the eye, enlarge the images to see these minor differences)_, the classification results show very high misclassification rates from the minimal to the final epoch generated data used. This suggests that the generated data is highly overfitting to the training set when an autoencoder is used and this method **should not be used to augment imbalanced datasets at all even if the data reflects a 'true' or stable posterior distribution**.

There is no proper learning of the real data's distributions unlike GANs. To conclude, using an autoencoder's synthetic data is no different from performing random oversampling for numerical-based features as it is unable to generate data to resemble a distribution. It is capable of 'vertically shifting' the distribution upwards rather than scaling it proportionately.

### References
\[1] [Autoencoder Code Reference](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb)

_**Noteworthy Python Dataset, Library & Tips**_ <br>
\[2] [Understanding Xavier Initialisation](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) <br>
\[3] [Source Credit Card Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) <br>
\[4] [Original Dataset Contributor & Publication](http://www.ulb.ac.be/di/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf) <br>
