import pandas as pd
import numpy as np
import streamlit as st

class Validator:
    '''
        Class to validate the CSV file.
    '''      
    def __init__(self):
      self.missing_values_df = None
      self.df = None

    def validate_file_format(self, file_path):
      if not file_path.endswith('.csv'):
        st.error('❌ Please provide a valid CSV file.')
        st.stop()

    def has_missing_values(self, df):
      '''
          Validate if there are any missing values in the dataset.
      '''
      missing_values_df = find_missing_values_percentage(df)
      if missing_values_df.shape[0] > 0:
        st.error('❌ There are missing values in the dataset.')
        st.dataframe(missing_values_df)
        return missing_values_df

      return None

    def has_duplicate_rows(self, df):
      '''
          Validate if there are any duplicate rows in the dataset.
      '''
      duplicate_rows = df.duplicated()
      if duplicate_rows.sum() > 0:
        st.error('❌ There are duplicate rows in the dataset.')
        return True

      return False

    def validate_empty_file(self, df):
      '''
          Validate if the csv file is empty.
      '''
      if df.empty:
        st.error('❌ The csv file is empty.')
        return True
      return False

def find_missing_values_percentage(df):
  '''
      Find the percentage of missing values for the columns.
  '''
  missing_values_all = df.isna().sum()
  missing_values = missing_values_all[missing_values_all > 0]
  missing_values_percentage = np.round((missing_values / df.shape[0]) * 100, 2)
  missing_values_percentage = missing_values_percentage.sort_values(ascending=False)
  missing_values_df = pd.DataFrame(missing_values_percentage, columns=['Missing Values Percentage'])
  return missing_values_df