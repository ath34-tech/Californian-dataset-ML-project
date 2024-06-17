import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#all library for data manupulation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#all models we gonna use
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

#all performance measuring
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score

#all fine tuning library
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


#it will take a label column name , test size and random state
class trainTestSplitter(TransformerMixin, BaseEstimator):
    def __init__(self,test_size,random_state):
        self.test_size=test_size
        self.random_state=random_state
        
    def fit(self):
        return self
        
    def transform(self,df,label):
        X=df.drop(label,axis=1)
        Y=df[label]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test


'''
1. We will first make a diff num df and categorical df
2.we will apply imputer on num df
3. we will use scaler on num df
4. we will use hot encoder on cat df

strategy will be passed
'''
class DataProcessor(BaseEstimator,TransformerMixin):
    def __init__(self,imputer_strategy):
        self.imputer_strategy=imputer_strategy
        self.num_df=None
        self.cat_df=None
        self.num_pipeline=None
        self.cat_pipeline=None
        self.full_pipeline=None
    def fit(self,df):
        self.num_df=df.select_dtypes(include=["number"])
        self.cat_df=df.select_dtypes(include=["object"])

        self.num_pipeline=Pipeline([
            ("imputer",SimpleImputer(strategy=self.imputer_strategy)),
            ("scaler",MinMaxScaler())
        ])
        self.cat_pipeline=Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ("encoder",OneHotEncoder())
        ])

        self.full_pipeline=ColumnTransformer([
            ('num', self.num_pipeline, self.num_df.columns.tolist()),
            ('cat', self.cat_pipeline, self.cat_df.columns.tolist())
        ])

        self.full_pipeline.fit(df)
        return self

    def transform(self,df):
        data_prep=self.full_pipeline.transform(df)
        return data_prep
        


'''
we will give 4 options: linear, tree, random forest and SVM
'''
class PredictingTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,model_name):
        self.model_name=model_name
        self.model=None
        self.train_data=None
        self.train_label=None
    def fit(self,train_data,train_label):
        self.train_data=train_data
        self.train_label=train_label
        if self.model_name=="linear":
            self.model=LinearRegression()
        if self.model_name=="tree":
            self.model=DecisionTreeRegressor()
        if self.model_name=="forest":
            self.model=RandomForestRegressor()
        if self.model_name=="svm":
            self.model=SVR(kernel='rbf', C=1.0, epsilon=0.2)

        self.model.fit(train_data,train_label)
        return self

    def transform(self,test_data):
        train_pred=self.model.predict(self.train_data)
        test_pred=self.model.predict(test_data)
        return train_pred,test_pred


'''
1. cross validation
2. rmse calculation
3. fine tuning using grid search
'''
class FineTunerTransformer(BaseEstimator,TransformerMixin):
    def __init__(self,model,param,cv_finetune,prediction,labels,cv=10):
        self.param=param
        self.cv_finetune=cv_finetune
        self.cv=cv
        self.model=model
        self.prediction=prediction
        self.labels=labels
        self.gridsearch=None
        self.best_params=None
        self.cross_val=None
        self.mse_model=None
    def fit(self,df):
        self.mse_model=mse(self.labels,self.prediction)
        self.cross_val=cross_val_score(self.model,df,self.labels,scoring="neg_mean_squared_error",cv=self.cv)
        self.gridsearch=GridSearchCV(self.model,self.param,cv=self.cv_finetune,scoring='neg_mean_squared_error')
        return self
    def transform(self,df):
        self.gridsearch.fit(df,self.labels)
        return self.gridsearch
        
    def display_scores(self):
        rmse=np.sqrt(-self.cross_val)
        print("scores:",rmse)
        print("Mean:",rmse.mean())
        print("std deviation:",rmse.std())


class CompleteTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df, target, finetune_params=[], imputer_strategy="median", splitting=0.2, random_state=42, model_name="linear", cv=10, finetune_cv=5):
        self.df=df
        self.target=target
        self.splitting=splitting
        self.random_state=random_state
        self.imputer_strategy=imputer_strategy
        self.model_name=model_name
        self.finetune_params=finetune_params
        self.cv=cv
        self.finetune_cv=finetune_cv
        self.x_train=None
        self.x_test=None
        self.y_train=None
        self.y_test=None
        self.trained_processed_data=None
        self.test_processed_data=None
        self.predictor=None
        self.fine_tuner=None
        self.train_pred=None
        self.test_pred=None
    def fit(self):
        #splitting data
        spliter=trainTestSplitter(self.splitting,self.random_state)
        spliter.fit()
        self.x_train, self.x_test, self.y_train, self.y_test=spliter.transform(self.df,self.target)
        print("data splitting done \n")
        #data processing both train and test
        train_processor=DataProcessor(self.imputer_strategy)
        train_processor.fit(self.x_train)
        self.trained_processed_data=train_processor.transform(self.x_train)
        print("trained data processed\n")
        test_processor=DataProcessor(self.imputer_strategy)
        test_processor.fit(self.x_test)
        self.test_processed_data=test_processor.transform(self.x_test)
        print("test data processed\n")

        #model initialization
        self.predictor=PredictingTransformer(self.model_name)   
        self.predictor.fit(self.trained_processed_data,self.y_train)
        print("model trained\n")
        return self

    def transform(self,X):
        self.train_pred,self.test_pred=self.predictor.transform(X)
        print("prediction is done\n")
        self.fine_tuner=FineTunerTransformer(self.predictor.model,self.finetune_params,self.finetune_cv,self.train_pred,self.y_train,self.cv)
        self.fine_tuner.fit(self.trained_processed_data)
        self.grid_search=self.fine_tuner.transform(self.trained_processed_data)

        print("grid search done\n")

        return self.train_pred,self.test_pred,self.grid_search,self.predictor.model
        
        