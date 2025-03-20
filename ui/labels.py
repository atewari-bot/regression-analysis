import streamlit as st

class UILabels:
  '''
  Class containing all labels used in the Streamlit App.
  '''
  def __init__(self):
    self.app_title = 'Data Analysis'
    self.label_encoded_feature = 'Label Encoded Features'
    self.select_dep_var_error = 'Please select a dependent variable.'
    self.select_dep_var_label = 'Select a dependent variable'
    self.empty_file_error = 'The uploaded file is empty. Please upload a non-empty CSV file.'
    self.missing_values_header = 'Missing Values'
    self.data_types_header = 'Data Types'
    self.summary_statistics_header = 'Summary Statistics'
    self.visualization_header = 'Data Visualization'
    self.correlation_matrix_header = 'Correlation Matrix'
    self.imputation_header = 'Data Imputation'

  def set_app_title(self):
    '''
    Set app title.
    '''
    st.title(self.app_title)

  def label_encoded(self):
    '''
    Set label for encoded data.
    '''
    st.write(self.label_encoded_feature)
      