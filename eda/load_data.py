import os
import streamlit as st
import pandas as pd
from utils.validator import Validator

class DAO:
  '''
  Data Access Object
  '''
  def __init__(self):
    self.file_path = None
    self.validator = Validator()

  def load_csv_file(self):
    '''
        Load a CSV file from the local directory.
    '''
    directory_path = os.getcwd()
    target_dir = os.path.join(directory_path, "data")
    files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    default_file_selection = ['Advertising.csv']

    selected_file = st.selectbox('Choose a Dataset (.csv)', files, index=files.index(default_file_selection[0]))

    self.file_path = 'data/' + selected_file
    self.validator.validate_file_format(self.file_path)
  
  # @st.cache_data
  def load_data(self):
    '''
        Load the dataset to DataFrame from the provided file path.
    '''
    st.write('Loaded Dataset: ', self.file_path.split('/')[-1])
    df = pd.read_csv(self.file_path)
    self.validator.df = df

    if self.validator.validate_empty_file(df):
        st.stop()
    return df