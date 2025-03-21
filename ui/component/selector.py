import streamlit as st
import pandas.api.types as ptypes
from sklearn.linear_model import LinearRegression, LogisticRegression

class UISelector:
    '''
    Class to select the UI components.
    '''
    def __init__(self, name='UISelector', validator=None):
        self.name = name
        self.validator = validator

    def variable_selector(self, df):
        '''
        Select dependent variable from the given options.
        '''
        # Create 2 column layout
        dependent_var_col, independent_var_col = st.columns(2)

        # Dependent variable selector
        chosen_y = dependent_var_col.selectbox('Select a dependent variable', df.columns)
        if not chosen_y:
            st.error('Please select a dependent variable.')
            st.stop()
        if not ptypes.is_numeric_dtype(df[chosen_y]):
            st.error('Please select a numeric column for dependent variable.')
            st.stop()

        # Independent variable selector
        chosen_x = independent_var_col.multiselect('Select independent variables', df.drop(chosen_y, axis=1).columns)
        if not chosen_x:
            st.error('Please select at least one independent variable.')
            st.stop()

        return chosen_y, chosen_x, dependent_var_col, independent_var_col
    
    def model_algo_selector(self, selector, model_algo_type):
        '''
        Returns the model algorithm key.
        '''
        model_options = selector.get_model_options()
        model_algo_key = model_algo_type.selectbox('Select an algorithm', model_options.keys())
        lr = LinearRegression() if model_options[model_algo_key] == 'lr' else LogisticRegression()
        if not lr:
            st.error('Please select an Algorithm.')
            st.stop()
        if type(lr) != LinearRegression:
            st.write('Only Linear Regression model supported at this moment. Please try again later!')
            st.stop()
        return lr
    
    def residual_metric_selector(self, lr, selector_options, error_metric_type):
        '''
        Returns the residual metric options and key.
        '''
        error_metric_key = None
        error_metric_options = None
        if type(lr) == LinearRegression:
            error_metric_options = selector_options.get_lr_error_metrics_options()
            error_metric_key = error_metric_type.selectbox('Select Residual Metric', error_metric_options.keys())
        elif type(lr) == LogisticRegression:
            error_metric_options = selector_options.get_lg_error_metrics_options()
            error_metric_key = error_metric_type.selectbox('Select Residual Metric', error_metric_options.keys())
        
        if not error_metric_key:
            st.error('Please select an error metric.')
            st.stop()
        return error_metric_options, error_metric_key
    
    def get_slider_value_selector(self, X, features):
        '''
            Select the number of features to select.
        '''
        return features.slider('Select number of features', min_value=0, max_value=X.shape[1], value=X.shape[1])
    
    def sfs_metrics_selector(self, selector_options):
        '''
            Select the Sequential Forward Selection (SFS) metrics.
        '''
        sfs_metrics_options = selector_options.get_sfs_metrics_options()
        selected_sfs_metrics_key = st.selectbox('Choose SFS Metrics', sfs_metrics_options.keys())
        selected_sfs_metrics = sfs_metrics_options[selected_sfs_metrics_key]

        if not selected_sfs_metrics:
            st.error('Please select a valid SFS metrics.')
            st.stop()
        return selected_sfs_metrics