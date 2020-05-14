#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 entro <entropy1208@yahoo.co.in>
#
# Distributed under terms of the MIT license.

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import category_encoders as ce
import matplotlib.pyplot as plt


df = pd.read_csv('train.csv', na_values="NA")
ordinal_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu',
                    'GarageQual', 'GarageCond', 'PoolQC', 'Fence']
nominal_features = ['MSZoning', 'Street', 'Alley', 'LotShape',
                    'LandContour', 'Utilities', 'LotConfig',
                    'LandSlope', 'Neighborhood', 'Condition1',
                    'Condition2', 'BldgType', 'HouseStyle',
                    'RoofStyle', 'RoofMatl', 'Exterior1st',
                    'Exterior2nd', 'MasVnrType', 'Foundation',
                    'Heating', 'CentralAir', 'Electrical',
                    'Functional', 'GarageType', 'GarageFinish',
                    'PavedDrive', 'MiscFeature',
                    'SaleType', 'SaleCondition']
df = df.drop(['Id'], axis=1)
# Get some data description and statistics
#print df.dtypes
#print df.shape
print df.head(5)
#print df['SalePrice'].describe()
#plt.scatter(range(1, 1461), df['SalePrice'])
#plt.show()
# Drop missing data
df.fillna(value=-99999, inplace=True)
df.dropna(inplace=True)
# Perform One Hot encoding on the nominal categorial features
df = pd.get_dummies(df, columns=nominal_features)
y = np.array(df['SalePrice'])
ordinal_mapping_1 = [('Ex', 5), ('Gd', 4), ('TA', 3),
                     ('Fa', 2), ('Po', 1), ('NA', 0)]
ordinal_mapping_2 = [('Gd', 4), ('Av', 3), ('Mn', 2),
                     ('No', 1), ('NA', 0)]
ordinal_mapping_3 = [('GLQ', 6), ('ALQ', 5), ('BLQ', 4),
                     ('Rec', 3), ('LwQ', 2), ('Unf', 1),
                     ('NA', 0)]
ordinal_mapping_4 = [('GdPrv', 4), ('MnPrv', 3), ('GdWo', 2),
                     ('MnWw', 1), ('NA', 0)]
binary_ordinal_mapping_1 = [('Ex', 11111), ('Gd', 11110),
                            ('TA', 11100), ('Fa', 11000),
                            ('Po', 10000), ('NA', 00000)]
binary_ordinal_mapping_2 = [('Gd', 1111), ('Av', 1110),
                            ('Mn', 1100), ('No', 1000),
                            ('NA', 0000)]
binary_ordinal_mapping_3 = [('GLQ', 111111), ('ALQ', 111110),
                            ('BLQ', 111100), ('Rec', 111000),
                            ('LwQ', 110000), ('Unf', 100000),
                            ('NA', 000000)]
binary_ordinal_mapping_4 = [('GdPrv', 1111), ('MnPrv', 1110),
                            ('GdWo', 1100), ('MnWw', 1000),
                            ('NA', 0000)]
encodings = {
    'OneHot': ce.OneHotEncoder(cols=ordinal_features),
    'Ordinal': ce.OrdinalEncoder(mapping=[
        {'col': 'ExterQual',
         'mapping': ordinal_mapping_1},
        {'col': 'ExterCond',
         'mapping': ordinal_mapping_1},
        {'col': 'BsmtQual',
         'mapping': ordinal_mapping_1},
        {'col': 'BsmtCond',
         'mapping': ordinal_mapping_1},
        {'col': 'BsmtExposure',
         'mapping': ordinal_mapping_2},
        {'col': 'BsmtFinType1',
         'mapping': ordinal_mapping_3},
        {'col': 'BsmtFinType2',
         'mapping': ordinal_mapping_3},
        {'col': 'HeatingQC',
         'mapping': ordinal_mapping_1},
        {'col': 'KitchenQual',
         'mapping': ordinal_mapping_1},
        {'col': 'FireplaceQu',
         'mapping': ordinal_mapping_1},
        {'col': 'GarageQual',
         'mapping': ordinal_mapping_1},
        {'col': 'GarageCond',
         'mapping': ordinal_mapping_1},
        {'col': 'PoolQC',
         'mapping': ordinal_mapping_1},
        {'col': 'Fence',
         'mapping': ordinal_mapping_4}],
        cols=ordinal_features),
    'Binary Ordinal': ce.OrdinalEncoder(mapping=[
        {'col': 'ExterQual',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'ExterCond',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'BsmtQual',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'BsmtCond',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'BsmtExposure',
         'mapping': binary_ordinal_mapping_2},
        {'col': 'BsmtFinType1',
         'mapping': binary_ordinal_mapping_3},
        {'col': 'BsmtFinType2',
         'mapping': binary_ordinal_mapping_3},
        {'col': 'HeatingQC',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'KitchenQual',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'FireplaceQu',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'GarageQual',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'GarageCond',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'PoolQC',
         'mapping': binary_ordinal_mapping_1},
        {'col': 'Fence',
         'mapping': binary_ordinal_mapping_4}],
        cols=ordinal_features)}
classifiers = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'SVR': LinearSVR()
    }
results = {
    'Ridge Regression': {},
    'Decision Tree': {},
    'Random Forest': {},
    'SVR': {}
    }
for clf in classifiers:
    for encoder in encodings:
        X = encodings[encoder].fit_transform(df.drop(['SalePrice'], 1))
        scores = cross_val_score(classifiers[clf], X, y, cv=10)
        results[clf][encoder] = []
        results[clf][encoder].extend(scores)
        print "Accuracy for %s encoder with %s model is : %0.2f (+/- %0.2f)" \
            % (encoder, clf, scores.mean(), scores.std() * 2)
# Draw graphs
x = map(abs, results['SVR']['OneHot'])
y = map(abs, results['SVR']['Ordinal'])
z = map(abs, results['SVR']['Binary Ordinal'])
plt.scatter(x, y, marker='x', color="green", s=30, label="Ordinal")
plt.scatter(x, z, marker='o', color="blue", s=30, label="Binary Ordinal")
x = np.linspace(0, 1, 1000)
plt.plot(x, x + 0, '-r')  # solid green
plt.title("OneHot vs Binary vs Ordinal Binary for SVR")
plt.xlabel("OneHot")
plt.ylabel("Prediction score")
plt.legend(loc="upper right")
plt.show()
