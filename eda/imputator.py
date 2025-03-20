import streamlit as st
import pandas as pd

class Imputator:
    '''
        Class to perform data imputation on the missing values.
    '''
    def __init__(self, selector=None, dao=None, strategy='mean'):
        '''
            Initialize the Imputator class.
        '''
        self.selector = selector
        self.strategy = strategy
        self.dao = dao

    def fit_transform(self, X):
        '''
            Fit and transform the dataset.
        '''
        numeric_cols = X.select_dtypes(include=['number']).columns
        object_cols = X.select_dtypes(include=['object']).columns
        imputed_cols = []

        if self.strategy == 'mean':
            for col in numeric_cols:
              if X[col].isnull().sum() > 0:
                self.fill_value = X[col].mean()
                X[col] = X[col].fillna(self.fill_value)
                imputed_cols.append(col)
        elif self.strategy == 'median':
            for col in numeric_cols:
              if X[col].isnull().sum() > 0:
                self.fill_value = X[col].median()
                X[col] = X[col].fillna(self.fill_value)
                imputed_cols.append(col)
        elif self.strategy == 'mode':
            for col in object_cols:
              if X[col].isna().sum() > 0:
                self.fill_value = X[col].mode().iloc[0]
                X[col] = X[col].fillna(self.fill_value)
                imputed_cols.append(col)    
        else:
            st.error('Please select an imputation techinque.')
            st.stop()

        return X, imputed_cols

    def data_imputer(self, df):
      '''
          Perform data imputation on the missing values.
      '''
      missing_values = self.dao.validator.has_missing_values(df)
      if missing_values is not None:
          st.header('Data Imputation')
          imputation_options = self.selector.get_imputation_options()
          chosen_imputer_key = st.selectbox('Select imputation strategy', imputation_options.keys())
          self.strategy = imputation_options[chosen_imputer_key]
          
          df, imputed_cols = self.fit_transform(df[missing_values.index])

          if len(imputed_cols) == 0:
            st.warning(f'Imputation not applicable')
          else: 
            st.write('Imputation successfully applied.')
            st.write(df[imputed_cols].head())
      return df