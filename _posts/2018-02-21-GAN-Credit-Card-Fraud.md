---
layout: post
title:  "Handling Imbalanced Data with GANs, Fraud Detection on Augmented Data"
date:   2018-02-21 16:16:00 +0800
categories: Deep-Learning
tags: Imbalanced SMOTE Random-Sampling Oversampling Deep-Learning Generative-Adversarial-Networks GAN Wasserstein-GAN 
---

## Generative Adversarial Networks in Credit Card Fraud Detection

---

_21st Feb 2018_<br>
_Cheo Rui Ming (SG)_<br>
_Refer [here](https://github.com/CheoRM/cheorm.github.io/blob/master/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud-Walkthru.ipynb) for a detailed technical walkthrough and code. [For accompanying notes, refer here.](https://github.com/CheoRM/cheorm.github.io/tree/master/accompanying_blog_notes/GAN_Credit_Card_Fraud)_

---

### Preamble
Credit card frauds are a problem to businesses. According to a 2016 study by [Lexis Nexis](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf), the average cost of fraud, as a percentage of revenue, stood at 1.47% in 2016, up from 1.32% in 2015. While automated fraud mitigation systems are likely in place, a large proportion of frauds flagged by these automated systems are still manually reviewed. Of the 50-54% of transactions flagged, between 42-47% of these flagged transactions were sent for manual review. Collectively, expenditures by these reviews constitute 25-36% of the fraud mitigation budgets.

In light of the cost of manual reviews, an efficient machine learning algorithm capable of identifying frauds accurately would be valuable to the business. However, detecting frauds can be a problem when they only constitute a tiny fraction of the total number of transactions conducted within a given period. To deal with this, sampling methods both on the benign and fraudulent transactions may be implemented to assist the machine when it is learning to identify frauds. For more details on how to handle imbalanced datasets, I recommend this [blog on how to deal with imbalanced datasets using various sampling methods](https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/).

In this blog, random uniform upsampling and Synthetic Minority Oversampling Techniques (SMOTE) were explored to see whether these methods of augmenting the minority class are useful in improving the classifier's fraud detection capabilities. The results were next compared to augmenting data with generated data from Generative Adversarial Networks (GAN). In brief, GANs attempt to learn the underlying feature distributions of the real data to generate data that identify with these distributions. Random upsampling duplicates the minority class observations found in the training set while SMOTE looks at the minority class data and add some level of noise to them to generate new data observations. For all data augmentation methods, the stochastic gradient descent classifier was being implemented as the common machine classifier.

The end purpose of doing data augmentation is to improve the classifier's accuracy in sieving out frauds, thereby reducing the cost of fraud to businesses. Referencing to the reported cost of fraud at 1.47% of total revenue by Lexis Nexis, this will be used as the benchmark to evaluate each model with different data augmentation techniques.

### Overview of the Data

In this study, the widely used credit card dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) (originally contributed by [Andrea Dal Pozzolo](http://www.ulb.ac.be/di/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf) used in his Ph.D. thesis on frauds) was obtained for the proof of concept in this study.

The dataset was train-test split into 75-25. All models trained on these 75% of the data while the remaining 25 was used to validate each model's performances.


```python
### Load credit card data and preprocessing ###

# Import 'creditcard.csv'
credit_csv_dir = 'creditcard.csv'
credit = pd.read_csv(credit_csv_dir)

# Convert data into 'mini-batchable' numpy NORMAL arrays, drop 'Time'
credit_X = credit.drop(labels=['Time','Class'], axis='columns')
credit_y = credit['Class'].as_matrix()

# Train-test split data: 75/25
X_train, X_test, y_train, y_test = train_test_split(credit_X, credit_y, test_size=0.25, random_state=seed)

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



In it, it consists of 31 columns. Features 'V1' through 'V28' were obtained via a PCA-transformation of original features to anonymise the data. In the following sections, 'Amount' will have to be standardised later as well to ensure consistency in the data.

Since the maximum value of 'Time' is 172,792 seconds from the first transaction of the dataset, we know that the data covers 2 days' worth of transactions.

In addition, the number of frauds in the dataset is only a **mere 0.172% out of 284,807 transaction entries!**


```python
# Compute number of fraud and non-fraud classes
print(credit['Class'].value_counts(), '\n')
```

    0    284315
    1       492
    Name: Class, dtype: int64 
    
    

It seems there are only 492 fraudulent transactions! It is highly probable that the classifier would fail to learn how to identify frauds since there are only so few data points available to observe for differences from normal transactions.

The total amount of fraudulent transactions was next computed for the purpose of cost computations of model implementation later. In total, the monetary transactions in frauds summed to \$42,513.46 and \$17,614.51 in the training and test sets respectively. Therefore, this figure totals to \$102,032.30 for the entire dataset.

Before moving on to train the classifiers on the data set, feature selection is necessary to assist the models in identifying fraudulent transactions from benign. To formalise this process, the distributions of each feature is first visualised to check for differences between frauds and benign.

In addition, 'Time' was pre-removed to produce models capable of detecting frauds regardless of time of action by fraudsters. The concern is that had this feature been included into the training process, the trained models may fail to detect frauds when the modus operandi of fraudsters change.

### Exploratory Data Analysis - Visualising Distributions

Adopting a conservative approach, all visualisations of the data for differences between fraudulent and non-fraud transactions were limited to the training set alone. If the training data is representative of fraudulent behaviours, then working within the boundaries of the training set features' characteristics would produce good generated data for the classifier to learn. Otherwise, poor results may signal high variances within the data generating process.

#### A PCA Plot of the Training Set

A PCA plot was first plotted to observe for an aggregated pattern about the training data.


```python
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
```

    explained variance ratio (first two components): [ 0.12503723  0.10004168]
    

<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_8_1.png" width="100%" height="100%" />


To illustrate the difficulty for any classifier to learn differences between classes in this dataset, the PCA plot of the first 2 components show that frauds (minority) are highly intersecting the area of the non-fraud (majority) clusters. Reducing the level of misclassification will not be an easy task since classification models can potentially identify non-frauds as malicious. There are hardly differences between both classes.

#### Plotting Feature Distributions

The distributions of fraud and benign transactions were next plotted to identify differences. If they are different, intuitively they ought to help assist the classifier's predictive ability to discern frauds out from thousands of transactions.


```python
# Kernel Density Distribution subplots against target - 'Class'

# Scatter subplots
plt.figure(figsize=(20,18))
for plot, feat in enumerate(X_cols):
    
    plt.subplot(5, 6, (plot+1))
    title = 'Fraud/Non-Fraud & ' + feat
    
    # Normalise to visualise the differences in distributions
    temp_df = pd.concat([X_train_df[[feat]], y_train_df], axis='columns')
    temp_df.groupby(by='Class')[feat].plot(kind='kde', alpha=0.7, legend='best', lw=2.5)
    plt.title(title)
    plt.tight_layout(); plt.margins(0.02)
    
plt.show()    
```


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_10_0.png" width="100%" height="100%" />


It is fortunate to note that the distribution of fraudulent cases are quite different from non-fraudulent ones in some of the features. However, it would be too informal and presumptuous to make definitive judgements via visual inspections. For the sake of reaching a robust conclusion, equality of distribution tests were implemented. If there are little statistical differences between fraud and non-frauds for a particular feature, then this feature would be a poor predictor of fraud transactions.

In this case, the Wilcoxon Rank-Sum Test was chosen since outliers were found in the data such as 'Amount', 'V27' and 'V28'.


```python
# Rank-Sum Test determined differences in classes within feature
wilcox_feat = [feat for feat in wilcox_result.keys() if wilcox_result[feat][0] == 'Diff']
print('\n', 'Wilcoxon Rank-Sum Relevant Features: ', wilcox_feat)
print('Total number of features selected: {}'.format(len(wilcox_feat)))
```

    Feature "V13" failed to be rejected at 5% level with p-value 0.0973465482
    Feature "V15" failed to be rejected at 5% level with p-value 0.8092477538
    Feature "V22" failed to be rejected at 5% level with p-value 0.3224055980
    Feature "V23" failed to be rejected at 5% level with p-value 0.7011952304
    Feature "V26" failed to be rejected at 5% level with p-value 0.5115929524
    Feature "Amount" failed to be rejected at 5% level with p-value 0.9531649675
    
     Wilcoxon Rank-Sum Relevant Features:  
     ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 
     'V14', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V24', 'V25', 'V27', 'V28']
    Total number of features selected: 23
    

From the Wilcoxon tests, the null hypothesis for features 'V15', 'V22', 'V26', 'V13', 'V23' and 'Amount' fail to be rejected at 5% level - indicating that these features will not be useful in helping machine classifiers identify frauds.

Following, the scatter relations for the top 6 most correlated selected features were observed as shown below to visualise how different these features are:


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


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_15_0.png" width="100%" height="100%" />


Like the PCA plot in the overview section, fraudulent transactions are overlapping non-frauds heavily although (fortunately) not entirely. If the strategy is to upsample the minority class to assist algorithms in identifying frauds, it will not be unexpected to find high rates of misclassification in this dataset after the upsampling process.

In the section that follows, models training on data without augmentation of the minority class were first implemented. Across all sampling methods, the base classifier used is a stochastic gradient descent classifier.

### Baseline & Gridsearched Baseline Training

Before evaluating models in terms of their cost to businesses, a few assumptions and cost parameters will need to be defined to build the evaluation metrics. These assumptions may not be the best representation to every business but it provides a basis of cost comparisons between fraud detection models.

#### Cost of Fraud Computation Methodology & Assumptions

_A few assumptions defined to aid in the cost computations_:<br>


\[1] The cost of fraud as a percentage to businesses is [1.47% of total revenue](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf). This figure is attributed solely from fraudulent transactions. It does not contain any fraud mitigation and management expenditures.

\[2] 42-47% of transactions flagged as fraud by automated systems are sent for [manual review](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf).

\[3] The cost of each manual review ranges between [$40-70](https://www.quora.com/How-much-does-it-cost-issuing-banks-to-investigate-a-credit-card-fraud-case).

\[4] [Every dollar of fraud costs $2.40 to the business](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf). With 492 instances of fraud transactions amounting to \$102,032.30, the average cost per case of fraud is taken as \$293.31.

\[5] Since it costs \$2.40 per dollar of fraud and 1.47% of business revenue, the revenue for the institution which provided this dataset is reversed calculated, holding the assumptions above true.

\[6] For any model, it is the only line of defence against fraud prevention.

_Constraints were further added into the calculations_:<br>


\[1] Of the 42-47% sent for manual review, there is a possibility that **all** investigations sent for review turn out benign. Then, no benefits can be reaped through opening these investigations.

\[2] Other verification processes on whether the flagged transactions are fraudulent have negligible costs.

\[3] [The percentage of successful credit card frauds in 2016 was 58%.](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf) Therefore, the minimal rate of recall on frauds (ability to identify all frauds) by each model needs to be at least 42%.

\[4] Costs incurred from implementing these models come from 2 sources; the cost of manual reviews flagged by automated systems as well as false negatives, frauds failed to be picked up.

#### Baseline Model: Stochastic Gradient Descent Classifier, Untuned




    Time elapsed to train:  0.35260605812072754
    Prediction Score:  0.977612988399 
    
    Confusion Matrix:  
     [[69505  1584]
     [   10   103]] 
    
    Classification Report: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.98      0.91      0.99      0.25      0.07     71089
              1       0.06      0.91      0.98      0.11      0.25      0.06       113
    
    avg / total       1.00      0.98      0.91      0.99      0.25      0.07     71202
    
    

Without much tuning, the basic model performs unexpectedly decent. With a recall of 0.91 on the minority (fraudulent) class, only 9% of all flagged frauds were left undetected. Unfortunately this comes with its own caveats. The level of misclassification is substantial, leaving the precision of the classifier on the minority class at 6%.

Computing the costs, the baseline model incurs costs between 1.09-2.03% as a percentage of revenue. Extending the baseline, a gridsearched version was implemented to evaluate for improvements.

#### Baseline Model: Stochastic Gradient Descent Classifier, Gridsearched Tuning




    Fitting 5 folds for each of 1260 candidates, totalling 6300 fits
    

    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:   10.3s
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:   46.9s
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed:  3.2min
    [Parallel(n_jobs=-1)]: Done 1242 tasks      | elapsed:  5.0min
    [Parallel(n_jobs=-1)]: Done 1792 tasks      | elapsed:  7.3min
    [Parallel(n_jobs=-1)]: Done 2442 tasks      | elapsed:  9.9min
    [Parallel(n_jobs=-1)]: Done 3192 tasks      | elapsed: 12.9min
    [Parallel(n_jobs=-1)]: Done 4042 tasks      | elapsed: 16.2min
    [Parallel(n_jobs=-1)]: Done 4992 tasks      | elapsed: 20.0min
    [Parallel(n_jobs=-1)]: Done 6042 tasks      | elapsed: 24.1min
    [Parallel(n_jobs=-1)]: Done 6300 out of 6300 | elapsed: 25.1min finished
    

    Time elapsed to train:  1505.3255302906036
    Prediction Score:  0.999353950732 
    
    Confusion Matrix:  
     [[71071    18]
     [   28    85]] 
    
    Classification Report: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      1.00      0.75      1.00      0.91      0.84     71089
              1       0.83      0.75      1.00      0.79      0.91      0.81       113
    
    avg / total       1.00      1.00      0.75      1.00      0.91      0.84     71202
    
    


```python
# Print best parameters used in 'GridSearchCV'
print(sgd_gridsearch.best_params_)
```

    {'alpha': 316.26417517967889, 'class_weight': 'balanced', 'eta0': 0.0001, 'l1_ratio': 0.0001, 
    'loss': 'modified_huber', 'n_jobs': -1, 'penalty': 'elasticnet', 'random_state': 42}
    

While recall on fraud cases has fallen, the precision in accurately identifying frauds has risen. This heightened precision reduces expenditures on manual reviews to businesses significantly. By implementing this model, it reduces the costs to between 0.35-0.40%, a substantial decrease from the results of the untuned baseline earlier.

Following the baselines' performances, random upsampling and SMOTE were implemented to augment the training set's fraud observations with the aim of improving the classifier's precision and recall on the minority cases. 

### Random Uniform Oversampling & SMOTE Augmented Data Training

#### A little more about Random Uniform Oversampling & SMOTE (Caveats)

_Random Uniform Oversampling_<br>

In random uniform oversampling methods, there is a risk of overfitting the dataset because observations in the dataset are repeatedly duplicated. When training is conducted on this artificially inflated dataset, the duplicates may be over represented which may not be representative of the natural underlying feature distributions of the minority class(es).

_Synthetic Minority Oversampling Technique_<br>

SMOTE and variations of it chiefly introduce random noises to the actual observed data points in order to generate new data points. In this case, the likelihood of overfitting to the actual dataset is lower than random oversampling. However, this comes with caveats.

With the inclusion of noise, it is unknown how this would adversely affect the features' distribution patterns. Suppose the features of the minority class are already representative of reality's distribution patterns, generating synthetic datapoints may shift the distributions in unpredictable ways that they are no longer reflective of reality and or the data.

#### Augmented Model: Random Oversampling

As the training set contains only 379 cases of fraud, the number of frauds were augmented with an addition of 5000 observations, raising the number of frauds to 5379 in the data set via random oversampling.




    Time elapsed to train:  0.3977847099304199
    Confusion Matrix:  
     [[68543  2546]
     [   10   103]] 
    
                       pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.96      0.91      0.98      0.20      0.04     71089
              1       0.04      0.91      0.96      0.07      0.20      0.04       113
    
    avg / total       1.00      0.96      0.91      0.98      0.20      0.04     71202
    
    

While the data augmented via random uniform oversampling produced a high recall on frauds (0.91), its precision remains poor at 0.04 which is lower than the regular baseline at 0.06. While small in percentage, the absolute number of false positives amount to significant costs incurred through increased manual reviews required.

This would have pushed the costs, as a percentage of revenue, to at least 1.65% and as high as 3.13%. This is sharply higher than the cost of not preventing frauds at all which costs approximately 1.47%. Implementing this model makes irrational sense.

#### Augmented Model: SMOTE

SMOTE was next being implemented to upsample the minority (fraud) class' observations. Again, an additional 5000 synthetic fraud observations were generated on top of the existing real frauds.



    Time elapsed to train:  0.5044991970062256
    Confusion Matrix:  
     [[63691  7398]
     [    9   104]] 
    
                       pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.90      0.92      0.95      0.12      0.02     71089
              1       0.01      0.92      0.90      0.03      0.12      0.01       113
    
    avg / total       1.00      0.90      0.92      0.94      0.12      0.02     71202
    
    

Via SMOTE, the degree of misclassification worsens from the previous. For a relatively similar level of recall on the minority class, misclassification rose by almost 5000 observations. The cost has now risen to between 4.47-8.67%!

This increment in misclassification is similar to the number of synthetic data points inserted into the training set. This may not be coincidental since SMOTE potentially distorts the underlying feature distributions of the minority class such that these synthetic datapoints no longer resemble the true data adequately for accurate classification.

#### Augmented Model: SMOTE + Tomek Linkages, Overlap Removal

One possibility of misclassifications may be due to overlapping observation points in the data as frequently observed from the plots shown earlier. Therefore, the Tomek Linkage undersampling process was next implemented after SMOTE to evaluate for classification improvements.


    

    Time elapsed to train:  62.56735682487488
    Confusion Matrix:  
     [[69206  1883]
     [   10   103]] 
    
                       pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.97      0.91      0.99      0.23      0.06     71089
              1       0.05      0.91      0.97      0.10      0.23      0.05       113
    
    avg / total       1.00      0.97      0.91      0.99      0.23      0.06     71202
    
    

Fortunately, results improved significantly when Tomek Linkages were used to remove any potential overlaps between fraud and non-fraud cases. In fact, the results are much better in comparison to the random uniform oversampling method. The costs now rest between 1.26-2.37% as a percentage of revenue. Nonetheless, this performance is not satisfactory since the gridsearched baseline still performs the best out of all the techniques explored at this point.

The changes in the top 6 pairs of correlated features were additionally plotted below to visualise changes to the data's characteristics.


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_33_0.png" width="100%" height="100%" />


From these diagrams, random uniform upsampling darkens spots with existing data while SMOTE techniques appear to generate additional noises. Resultantly, the scatter patterns are sharply altered relative to random upsampling. Unsurprisingly, data augmented with SMOTE fared the worst relative to the regular baseline while random oversampling being the second.

These diagrams thus illustrate the points aforementioned. It is likely random uniform oversampling caused overfitting of the classifier onto the training dataset such that it fares worse when validating its performance with unseen data. On SMOTE, they introduce unwanted noises that skew the underlying distributions of the features to the point it 'confuses' the classifier, hence resulting in its poorer performance.

_Garbage In, Garbage Out..._

#### General conclusions about random oversampling & SMOTE

It may be argued that minimal efforts were put into gridsearching for the optimal parameters. However, notice that the SMOTE-TL took approximately 1 minute. Even earlier, the gridsearched baseline required almost half an hour of gridsearching for that few hyperparameter options specified. Had gridsearch been implemented into the sampling and classification process, the amount of time taken to produce results potentially scales. With more hyperparameters to search for, the likelihood of encountering a combinatorial explosion!

Moreover, gridsearching is akin to brute forcing out optimal hyperparameter values. In this case, it is not unreasonable to postulate that the classifier would have been overfitted to the test set as well. When unseen fraudulent transactions arrive, the classification models may be incapable of identifying them at all. 

To address this issue, the Generative Adversarial Networks (GAN) is proposed here to augment minority class(es) within a dataset instead. The advantage of GANs is that the underlying feature distributions are learned by the network to produce realistic data.

### Generative Adversarial Networks
_**Why GANs in data augmentation**_<br>

First proposed by [Ian Goodfellow and colleagues](https://arxiv.org/abs/1406.2661), the vanilla GANs' implementation comprises of two networks, the generator and discriminator, which compete against each other in a min-max optimisation process. The highly used case example would be counterfeiting banknotes. The generator is the counterfeiter while the discriminator is the investigator trying to figure out whether the banknotes synthesized by the generator are authentic. Both network receives feedback from each other and improve in reproducing near genuine banknotes and being a better investigator over time.

At each particular epoch of neural net training, the generator draws 'noises' from a prior distribution. As it receives feedback from the discriminator in progressive epochs, the generator learns to map this prior distribution to the true (if not the posterior) distribution of the data inputted better.

Eventually, the generator learns to adjust the synthetic data's distribution to resemble more like the real data's distribution while the discriminator becomes more adept at adjusting the boundary which separates the differences between the generated data and the real data's distribution. 


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_gan5.png" width="100%" height="100%" />
Source: https://ireneli.eu/2016/11/17/deep-learning-13-understanding-generative-adversarial-network/_

The figure above illustrates how the adversarial network functions. The discriminator's decision boundary function is indicated with blue dotted lines, black dotted lines for the real data's probability density function _(pdf)_ and the generated data's pdf by green lines.

From panel (a) to (b), the discriminator learns to differentiate real and generated data and adjusts the boundary function accordingly. Anything beyond/to the right of the blue dotted line would consequently be identified as generated data.

From panel (b) to (c), upon receiving this feedback, the generator then knows that it has to adjust the generated data's distribution such that they lie as far to the left of the blue sigmoid curve. The generator next aims to map the prior values closer to the left of the decision boundary, approximating to the real data's distribution (black dotted) during this process.

After multiple epochs, the theoretical optimum is reached when both the generated and real data's distributions are identical, depicted in panel (d). At this juncture, the blue dotted curve is now a horizontal line (which is 0.5), which shows that the discriminator can no longer differentiate the authenticity of the data. For some data drawn, the discriminator has a probability of 0.5 to identify the data incorrectly. That is, there is no difference from a guess whether the data is real or generated.

#### Caveats of Training GANs

The only drawback of using classical GANs is that they tend to be delicate to train. During a training epoch, either the discriminator or the generator have the ability to overpower the other which could have rendered either network incapable of learning to generate better synthetic data or adjust the decision boundary.

Further on that note, mode collapses happen when the generator is capable of fooling the discriminator by generating data that falls within a small segment of the real data's distribution. With reference to the diagrams above, the generator repeatedly generates data points to the left of the sigmoid curve only.

When that happens, the generator is no longer capable of generating synthetic data close to the real data distribution since creating samples that resemble a small segment of it is sufficient to fool the discriminator. The generator no longer explores the full possibilities of generated data variants and hence 'overfits' to a segment of the real (posterior) distribution.

#### Altogether, why GANs make sense now

Using GANs, data generated from these networks theoretically have the properties of the real data's characteristics. This provides a more robust method of generating artificial samples than to add random noises (SMOTE) which there is less control over the changes in the features' distributions as well as preventing an overfit to the training set (random uniform oversampling).

The important thing is that if the GAN is capable of learning the posterior distribution of the real data, then the likelihood of overfitting is reduced. Augmenting the minority class(es) of imbalanced datasets will be a proportional scaling of the features' distributions.

### GANs, a practical showcase

The classical GAN was then implemented and to augment the training set's fraud observations with the aim of improving the classifier's (SGD classifier) ability to discern fraudulent transactions from normal ones.

Point to note:
Since GANs are notoriously difficult to train, a few pointers were integrated to help assist the networks' functionalities. More recommendations may be found [here](https://github.com/soumith/ganhacks).
    
Since there were 6300 fits in the gridsearched baseline, the GAN was trained for 6300 epochs correspondingly. At each epoch, 5000 synthetic datapoints were generated. However, not all generated data is used, only the epoch deemed suitable has its generated data used in augmentation.


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_39_0.png" width="100%" height="100%" />


Interestingly, the losses of the generator rises over epochs while decreases in the discriminator. This could suggest that the discriminator increasingly overpowers the generator over epochs. However, this is a good problem in the sense where the generator is forced to repeatedly generate new variants of synthetic data (thereby exploring more thoroughly the characteristics of the real data's distributions) to fool the discriminator.

Towards the final epoch, the training of the GAN on fraudulent data have losses approaching stabilisation. Data near this epoch could have been used to augment the training set. Nonetheless to set some standards, the GAN optimisation is declared to reach an optimum (or minimally stabilisation) when all losses in a 5% frame of the total epochs executed fluctuate within +/-75% of 1 standard deviation from the median loss recorded.

As the generator and the discriminator each have their own loss values (thus having their own stable frame of epochs), generated data from the latest epoch of the two networks was used.



    Steady epoch frame found at epoch 1520 as final
    Steady epoch frame found at epoch 1539 as final
    Fraud steady epoch: 1539
    

From above, the steady epoch was found to reside at the 1539th.  For a thorough investigation, generated data from the final epoch was also used for performance comparisons.



    
     ############################################# STEADY EPOCH #############################################
    Prediction Score (w/o standardisation) / Steady Epoch:  0.997078733743 
    
    Confusion Matrix (w/o standardisation) / Steady Epoch:  
     [[70896   193]
     [   15    98]] 
    
    Classification Report (w/o standardisation) / Steady Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      1.00      0.87      1.00      0.58      0.36     71089
              1       0.34      0.87      1.00      0.49      0.58      0.31       113
    
    avg / total       1.00      1.00      0.87      1.00      0.58      0.36     71202
    
    ==========================================================================================================
    Prediction Score (with standardisation) / Steady Epoch:  0.991109800287 
    
    Confusion Matrix (with standardisation) / Steady Epoch:  
     [[70468   621]
     [   12   101]] 
    
    Classification Report (with standardisation) / Steady Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.99      0.89      1.00      0.37      0.15     71089
              1       0.14      0.89      0.99      0.24      0.37      0.13       113
    
    avg / total       1.00      0.99      0.89      0.99      0.37      0.15     71202
    
    

The outcomes above reveal that although the recall on the minority class improves with the re-standardisation of the augmented training set (fitted on the pre-augmented training set), the cost of false negatives rises sharply which may not be ideal. Without standardisation, the costs lie between 0.32-0.49% but rises to 0.54-0.95% when standardised. 

Though still below the benchmarked 1.47% cost of frauds (as a percentage of revenue), the standardised model provides less room for cost accommodation to other fraud mitigation and investigation methods which were not factored into this study. Moreover, the real data features were already standardised due to PCA-transformation and the generator would have produced synthetically standardised observations. It is not surprising to see that the standardised augmentation fared worse.



    
     ############################################# FINAL EPOCH #############################################
    Prediction Score (w/o standardisation) / Final Epoch:  0.997078733743 
    
    Confusion Matrix (w/o standardisation) / Final Epoch:  
     [[70895   194]
     [   14    99]] 
    
    Classification Report (w/o standardisation) / Final Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      1.00      0.88      1.00      0.58      0.36     71089
              1       0.34      0.88      1.00      0.49      0.58      0.32       113
    
    avg / total       1.00      1.00      0.88      1.00      0.58      0.36     71202
    
    ==========================================================================================================
    Prediction Score (with standardisation) / Final Epoch:  0.978834864189 
    
    Confusion Matrix (with standardisation) / Final Epoch:  
     [[69596  1493]
     [   14    99]] 
    
    Classification Report (with standardisation) / Final Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.98      0.88      0.99      0.25      0.07     71089
              1       0.06      0.88      0.98      0.12      0.25      0.06       113
    
    avg / total       1.00      0.98      0.88      0.99      0.25      0.07     71202
    
    

Generally, a training set augmented with GAN generated data produced substantial improvements in the result. Essentially, the classifier is able to detect fraudulent transactions better. Though the precision in detecting frauds suffers, it does not degrade dramatically. A misclassification of approximately 200 benign transactions is decent.

Using the best performing model (data augmentation using generated samples from the final epoch), the cost computes to between 0.31-0.48%. While not as well as the gridsearched baseline (0.35-0.40%) at the upper cost boundary, the model outperforms in terms of the recall on fraudulent cases at 0.88 compared to the gridsearched's recall of 0.75.

Suppose sieving out for fraudulent cases is a priority, the GAN architecture is a viable option to consider, offering increased recall rate on the minority class without the escalation of manual review and false negative costs by too much.

The scatter relations of the top 6 correlated features were next plotted to visualise changes.


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_47_0.png" width="100%" height="100%" />


Effectively, there is little distinction between the pre-augmented training data and the various augmented training set types. Observing close enough, certain areas in the augmented data appears more concentrated but not substantially. This could also be due to augmenting the minority class with just 5000 additional generated datapoints against a total of 213,605 transaction observations in the training set.

Nevertheless, recall that the patterns were distinctively different in the random oversampled and SMOTE scatter diagrams. This improvement in the classifier's performance is substantial enough to be a viable option for the purpose of minimising the cost of fraud if this methodology is ever deployed.

#### Advancements in GAN - Wasserstein

One of the prominent solution to training GANs proposed is the Wasserstein GAN (WGAN) by Arjovsky, Chintala & Bottou (2017). In it, the discriminator is modified into a critique of the generated data's distribution. Rather than optimising the decision boundary to decide whether the data is generated or drawn from the real data's distribution, it is modified to critique the distance (known as the Earth Mover (EM) - Wasserstein distance) between the real data and the synthetic data's distribution by imposing a Lipschitz constraint. The generator receives this feedback to generate synthetic data that approximates closer to the actual data's distribution at each training epoch.

To impose the Lipschitz constraint onto the discriminator, the weights of this network during backpropagation are 'clipped' from changing too rapidly to allow room for learning by the generator about the real data's distributions. For greater technicalities into WGAN, refer to the original paper by [Arjovsky et. al. (2017)](https://arxiv.org/abs/1701.07875).

In the following, the Wasserstein GAN was implemented with slight modifications. In the original implementation, the loss function utilises a root mean-squared error optimisation to both the generator and discriminator's loss functions. However, the adam and stochastic gradient descent optimiser were respectively implemented instead. In addition, the clipping range was modified to between +/-0.05 as opposed to the original +/-0.01 used. Results showed improvements in misclassifications relative to the classical GAN although not dramatically.


<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_49_0.png" width="100%" height="100%" />


When the Lipschitz constraint was imposed, the generator's losses became significantly stable over epochs. This is good as it potentially signals the generator being given the opportunity to learn the real data's distribution more thoroughly and continuously, avoiding a case of mode collapse. Had mode collapse occurred, the generator's loss would have plunged significantly downwards, or worse, occurring at a very early epoch.

Like the standards set in the GAN implementation, the stable frame is found when 5% of the epochs' losses stabilise within +/-75% of 1 standard deviation from the median loss value.



    Steady epoch frame found at epoch 1665 as final
    Steady epoch frame found at epoch 440 as final
    Fraud steady epoch: 1665
    

It appears that the discriminator stabilises around the 1665th epoch while the generator at a very early period. This shows that the losses of the generator is repeatedly difficult to minimise, encouraging it to explore the full spectrum of the real data's characteristics to reproduce better generated samples.



    
     ############################################# STEADY EPOCH #############################################
    Wasserstein Prediction Score (w/o standardisation) / Steady Epoch:  0.996643352715 
    
    Wasserstein Confusion Matrix (w/o standardisation) / Steady Epoch:  
     [[70864   225]
     [   14    99]] 
    
    Wasserstein Classification Report (w/o standardisation) / Steady Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      1.00      0.88      1.00      0.55      0.33     71089
              1       0.31      0.88      1.00      0.45      0.55      0.28       113
    
    avg / total       1.00      1.00      0.88      1.00      0.55      0.33     71202
    
    ==========================================================================================================
    Wasserstein Prediction Score (with standardisation) / Steady Epoch:  0.991025532991 
    
    Wasserstein Confusion Matrix (with standardisation) / Steady Epoch:  
     [[70464   625]
     [   14    99]] 
    
    Wasserstein Classification Report (with standardisation) / Steady Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.99      0.88      1.00      0.37      0.15     71089
              1       0.14      0.88      0.99      0.24      0.37      0.12       113
    
    avg / total       1.00      0.99      0.88      0.99      0.37      0.15     71202
    
    
     ############################################# FINAL EPOCH #############################################
    Wasserstein Prediction Score (w/o standardisation) / Final Epoch:  0.997542203871 
    
    Wasserstein Confusion Matrix (w/o standardisation) / Final Epoch:  
     [[70928   161]
     [   14    99]] 
    
    Wasserstein Classification Report (w/o standardisation) / Final Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      1.00      0.88      1.00      0.62      0.40     71089
              1       0.38      0.88      1.00      0.53      0.62      0.36       113
    
    avg / total       1.00      1.00      0.88      1.00      0.62      0.40     71202
    
    ==========================================================================================================
    Wasserstein Prediction Score (with standardisation) / Final Epoch:  0.99245807702 
    
    Wasserstein Confusion Matrix (with standardisation) / Final Epoch:  
     [[70565   524]
     [   13   100]] 
    
    Wasserstein Classification Report (with standardisation) / Final Epoch: 
                        pre       rec       spe        f1       geo       iba       sup
    
              0       1.00      0.99      0.88      1.00      0.40      0.17     71089
              1       0.16      0.88      0.99      0.27      0.40      0.15       113
    
    avg / total       1.00      0.99      0.89      1.00      0.40      0.17     71202
    
    

With the results from augmenting data created by WGANs, the steady unstandardized WGAN yield a cost range between 0.33-0.51% while at the final unstandardized epoch, this cost ranges between 0.29-0.44%. Replicating this motif in the vanilla GAN unstandardized models, the steady and final epoch cost ranges are between 0.32-0.49% and 0.31-0.48%.

Through these, results suggest that the classical GAN models possibly require more training epochs before reaching the optimum. Comparing against WGAN, the results substantially improves from the steady to the final epoch by the time the final training epoch is ran. In addition, the precision of WGAN augmented data models appear to be higher than GAN models where the lowest recorded precision for WGAN was at 0.14 while 0.04 in GAN. This serves as an additional piece of evidence suggesting that WGAN would generally be a better architecture over its predecessor.



<img src="https://cheorm.github.io/accompanying_blog_notes/GAN_Credit_Card_Fraud/GAN-Credit-Card-Fraud_output_56_0.png" width="100%" height="100%" />


Like the data augmentation patterns found in GAN, there is little or near insignificant differences from the pre-augmented training set. Both the GAN and WGAN architectures prove capable of learning the real data's underlying distributions well or do not distort real data distributions when augmented with generated data at the very least.

### Summary of Findings

#### The impact of augmenting data from GANs on the cost of frauds


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
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Restd.</th>
            <th>Precision_fraud</th>
            <th>Recall_fraud</th>
            <th>FP</th>
            <th>TP</th>
            <th>FN</th>
            <th>TN</th>
            <th>Cost_L</th>
            <th>Cost_U </th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Baseline</td>
            <td>No</td>
            <td>0.06</td>
            <td>0.91</td>
            <td>1584</td>
            <td>103</td>
            <td>10</td>
            <td>69505</td>
            <td>1.09%</td>
            <td>2.03%</td>
        </tr>
        <tr>
            <td>Grid. Baseline</td>
            <td>No</td>
            <td>0.83</td>
            <td>0.75</td>
            <td>18</td>
            <td>85</td>
            <td>28</td>
            <td>71071</td>
            <td>0.35%</td>
            <td>0.40%</td>
        </tr>
        <tr>
            <td>Rand. Ovrsmpl</td>
            <td>No</td>
            <td>0.04</td>
            <td>0.91</td>
            <td>2546</td>
            <td>103</td>
            <td>10</td>
            <td>68543</td>
            <td>1.65%</td>
            <td>3.13%</td>
        </tr>
        <tr>
            <td>SMOTE</td>
            <td>No</td>
            <td>0.01</td>
            <td>0.92</td>
            <td>7398</td>
            <td>104</td>
            <td>9</td>
            <td>63691</td>
            <td>4.47%</td>
            <td>8.67%</td>
        </tr>
        <tr>
            <td>SMOTE-TL</td>
            <td>No</td>
            <td>0.05</td>
            <td>0.91</td>
            <td>1883</td>
            <td>103</td>
            <td>10</td>
            <td>69206</td>
            <td>1.26%</td>
            <td>2.37%</td>
        </tr>
        <tr>
            <td>GAN, Steady</td>
            <td>No</td>
            <td>0.34</td>
            <td>0.87</td>
            <td>193</td>
            <td>98</td>
            <td>15</td>
            <td>70896</td>
            <td>0.32%</td>
            <td>0.49%</td>
        </tr>
        <tr>
            <td>GAN, Steady</td>
            <td>Yes</td>
            <td>0.14</td>
            <td>0.89</td>
            <td>621</td>
            <td>101</td>
            <td>12</td>
            <td>70468</td>
            <td>0.54%</td>
            <td>0.95%</td>
        </tr>
        <tr>
            <td>GAN, Final</td>
            <td>No</td>
            <td>0.34</td>
            <td>0.88</td>
            <td>194</td>
            <td>99</td>
            <td>14</td>
            <td>70895</td>
            <td>0.31%</td>
            <td>0.48%</td>
        </tr>
        <tr>
            <td>GAN, Final</td>
            <td>Yes</td>
            <td>0.06</td>
            <td>0.88</td>
            <td>1493</td>
            <td>99</td>
            <td>14</td>
            <td>69596</td>
            <td>1.07%</td>
            <td>1.96%</td>
        </tr>
        <tr>
            <td>WGAN, Steady</td>
            <td>No</td>
            <td>0.31</td>
            <td>0.88</td>
            <td>225</td>
            <td>99</td>
            <td>14</td>
            <td>70864</td>
            <td>0.33%</td>
            <td>0.51%</td>
        </tr>
        <tr>
            <td>WGAN, Steady</td>
            <td>Yes</td>
            <td>0.14</td>
            <td>0.88</td>
            <td>625</td>
            <td>99</td>
            <td>14</td>
            <td>70464</td>
            <td>0.57%</td>
            <td>0.97%</td>
        </tr>
        <tr>
            <td>WGAN, Final</td>
            <td>No</td>
            <td>0.38</td>
            <td>0.88</td>
            <td>161</td>
            <td>99</td>
            <td>14</td>
            <td>70928</td>
            <td>0.29%</td>
            <td>0.44%</td>
        </tr>
        <tr>
            <td>WGAN, Final</td>
            <td>Yes</td>
            <td>0.16</td>
            <td>0.88</td>
            <td>524</td>
            <td>100</td>
            <td>13</td>
            <td>70565</td>
            <td>0.50%</td>
            <td>0.85%</td>
        </tr>
    </tbody>
</table>
</div>



All in all, generative adversarial networks are very successful in learning the real data's distribution rather than duplicating the given data. While the recall on the minority-fraud class suffers relative to the random oversampling and SMOTE, this value is very small in absolute numbers. 

Further, the precision of the classifiers using GAN generated data perform much better than the vanilla methods. This supports the possible idea that the reason why random oversample and SMOTE augmented data possess greater recall on frauds is due to the fact that the classifier applies a 'blanket cover' and is likelier to identify data as frauds (hence the poorer precision). In a simplistic sense, this is akin to defensive strategies - better to err on the side of caution.

However, being defensive is not necessarily the optimal choice to businesses. The cost of managing frauds scales together with an increased amount of falsely identified transactions as frauds. In the accompanying [notes](https://github.com/CheoRM/cheorm.github.io/tree/master/accompanying_blog_notes/GAN_Credit_Card_Fraud), the cost calculations show that when methods with higher rates of misclassifications are implemented, the costs of manual reviews become correspondingly larger such that it becomes an irrational business decision to implement these models.

#### Concluding points

The purpose of choosing GANs to augment data was fundamentally borne out of the consideration on preventing overfitting to the training sets as well as avoiding overfitted classifiers on the test set via gridsearching. Moreover, while not an extensive array of non-GAN data augmentation methods were explored, the GAN/WGAN results above showed that time could potentially be saved from gridsearching as well as searching for the method that augments data well.

Since every dataset is unique, so will the sampling methods likely be changed. Instead of researching for the right classical approach all over, GANs reduce this search effort by primarily learning the distribution of the data and generate samples out of it without incurring the possibility of combinatorial explosion (when gridsearching) or misrepresenting the real minority class(es) feature distributions/scatter patterns during the sampling process. When considering a GAN architecture, the WGAN is the better choice among the two GAN systems explored here for its faster convergence to the global optimum while less susceptible to mode collapsing.

#### Future works...

A large repository of generative models exists. Even more, a wide variation of GANs exists and only the vanilla and Wasserstein versions were implemented in this study. One of the more interesting models that emerged is the hybridisation of autoencoders and generative adversarial networks - the Adversarial Autoencoder (AAE).

While not shown here, the performance of an autoencoder alone did not produce significant differences from non-GAN data augmentation methods employed here. Perhaps by integrating an adversarial factor into autoencoders (on the encoder end), results could be better than GANs implemented here. Find out more about AAE [here](https://arxiv.org/abs/1511.05644).

### References

_**GAN & WGANs**_ <br>
\[1] [Classical GAN Publication](https://arxiv.org/abs/1406.2661) <br>
\[2] [Classical GAN Implementation](https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py) <br>
\[3] [Classical GAN in Image Generation](https://github.com/ckmarkoh/GAN-tensorflow/blob/master/gan.py) <br>
\[4] [Wasserstein GAN Publication](https://arxiv.org/abs/1701.07875) <br>
\[5] [Wasserstein GAN Implementation](https://github.com/wiseodd/generative-models/blob/master/GAN/wasserstein_gan/wgan_tensorflow.py) <br>
\[6] [Tips to Training GANs](https://github.com/soumith/ganhacks) <br>
\[7] [Understanding GANs Conceptually](https://ireneli.eu/2016/11/17/deep-learning-13-understanding-generative-adversarial-network/) <br>


_**Autoencoder & Adversarial Autoencoders**_ <br>
\[8] [Adversarial Autoencoders Publication](https://arxiv.org/abs/1511.05644) <br>
\[9] [Classical Autoencoder Implementation](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/autoencoder.ipynb) <br>


_**Noteworthy Python Dataset, Library & Tips**_ <br>
\[10] [Imblearn Python Library](http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html) <br>
\[11] [Understanding Xavier Initialisation](https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/) <br>
\[12] [Source Credit Card Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) <br>
\[13] [Original Dataset Contributor & Publication](http://www.ulb.ac.be/di/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf) <br>


_**Information on Frauds**_ <br>
\[14] [Lexis Nexis Cost of Fraud Report](https://www.lexisnexis.com/risk/downloads/assets/true-cost-fraud-2016.pdf) <br>
\[15] [Estimating Cost of Manual Review](https://www.quora.com/How-much-does-it-cost-issuing-banks-to-investigate-a-credit-card-fraud-case) <br>
\[16] [Cost of Fraud Article](https://skodaminotti.com/blog/the-art-of-pricing-a-forensic-investigation-what-will-your-fraud-examination-cost/)
