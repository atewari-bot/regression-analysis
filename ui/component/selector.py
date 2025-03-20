import streamlit as st
import pandas.api.types as ptypes

class UISelector:
    def __init__(self, name='UISelector'):
        self.name = name

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