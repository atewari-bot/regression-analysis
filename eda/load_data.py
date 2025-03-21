import os
import streamlit as st
import pandas as pd

class DAO:
  '''
  Data Access Object
  '''
  def __init__(self, validator=None):
    self.file_path = None
    self.validator = validator

  def load_csv_file(self, selected_file):
    '''
        Load a CSV file from the local directory.
    '''
    self.file_path = 'data/' + selected_file
    self.validator.validate_file_format(self.file_path)
    return self.file_path
  
  # @st.cache_data
  def load_data(self, file_path):
    '''
        Load the dataset to DataFrame from the provided file path.
    '''
    st.write('Loaded Dataset: ', file_path.split('/')[-1])
    df = pd.read_csv(file_path)
    self.validator.df = df

    if self.validator.validate_empty_file(df):
        st.stop()
    return df
  
  def list_files(self):
      '''
          List all the files in the data directory.
      '''
      directory_path = os.getcwd()
      target_dir = os.path.join(directory_path, "data")
      files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
      return files