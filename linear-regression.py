import os
import pandas as pd
import pandas.api.types as ptypes
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

st.title('Linear Regression Analysis')

def load_csv_file():
    '''
        Load a CSV file from the local directory.
    '''
    directory_path = os.getcwd()
    target_dir = os.path.join(directory_path, "data")
    files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    default_file_selection = ['Advertising.csv']

    # Store the previous selected options (in a session state)
    # if 'selected_options' not in st.session_state:  
    #     st.session_state.selected_options = []

    selected_file_url_option = st.selectbox('Choose a Dataset:', files, index=files.index(default_file_selection[0]))

    # Detect if a selection has been removed
    # removed_option = None
    # if len(st.session_state.selected_options) > len(selected_file_url_option):
    #     # Compare the previous and current selections to find the removed option
    #     removed_option = list(set(st.session_state.selected_options) - set(selected_file_url_option))[0]
    # Update the session state with the new selections
    # st.session_state.selected_options = selected_file_url_option

    if '.csv' not in selected_file_url_option:
        st.error('Please provide a valid CSV file path.')
        st.stop()
    return 'data/' + selected_file_url_option
  
@st.cache_data
def load_data(file_url):
    '''
        Load the dataset to DataFrame from the provided file path.
    '''
    st.write('Loaded Dataset: ', file_url.split('/')[-1])
    df = pd.read_csv(file_url)
    return df

def data_imputer(df):
    '''
        Perform data imputation on the missing values.
    '''
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write('Missing Values in the dataset')
        st.write(missing_values)

        chosen_imputer = st.selectbox('Select imputation technique:', ['mean', 'median', 'mode'])
        if not chosen_imputer:
            st.error('Please select an imputation techinque.')
            st.stop()

        numeric_cols = df.select_dtypes(include=['number']).columns
        object_cols = df.select_dtypes(include=['object']).columns
        imputation_count = 0

        if chosen_imputer == 'mean':
            st.header('Imputing missing values with mean...')
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
                imputation_count += 1
        elif chosen_imputer == 'median':
            st.header('Imputing missing values with median...')
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
                imputation_count += 1
        elif chosen_imputer == 'mode':
            st.header('Imputing missing values with mode...')
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                imputation_count += 1

        imputation_success, imputation_failure = st.columns(2)

        with imputation_success:
            st.write('Missing values imputed successfully for columns')
            st.write(df[numeric_cols].head())

        if len(df.columns) != imputation_count:
            with imputation_failure:
                st.write('Missing values imputation failed for columns')
                st.write(df[object_cols].head())
    return df

def variables_selector(df):
    '''
        Define multi selector for  dependent and independent variables.
    '''
    # Create 2 column layout
    dependent_var_col, independent_var_col = st.columns(2)

    # Dependent variable selector
    chosen_y = dependent_var_col.selectbox('Select a dependent variable:', df.columns)
    if not chosen_y:
        st.error('Please select a dependent variable.')
        st.stop()
    if not ptypes.is_numeric_dtype(df[chosen_y]):
        st.error('Please select a numeric column for dependent variable.')
        st.stop()
    
    y = df.loc[:, chosen_y]
    dependent_var_col.write(y.head())

    # Independent variable selector
    chosen_x = independent_var_col.multiselect('Select independent variables:', df.drop(chosen_y, axis=1).columns)
    if not chosen_x:
        st.error('Please select at least one independent variable.')
        st.stop()
    Z = df.loc[:, chosen_x]
    independent_var_col.write(Z.head())

    df = label_encoder(df.loc[:, chosen_x])
    X = df.loc[:, chosen_x]

    return y, X, df

def label_encoder(df):
    '''
        Perform Label Encoding on the categorical columns.
    '''
    # Initialize LabelEncoder
    encoder = LabelEncoder()
    # Apply to each object column
    cat_cols = df.select_dtypes(include=['object'])
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])
    
    if not cat_cols.empty:
        st.write('Label Encoded Features')
        st.write(df.head())
    return df

def feature_lr_selector(X):
    '''
        Select the number of features, linear regression algorithm and error metric.
    '''
    features, linear_algo_type, error_metric_type = st.columns(3)
    k = features.slider('Select number of features:', min_value=0, max_value=X.shape[1], value=X.shape[1])
    
    linear_algo = linear_algo_type.selectbox('Select a linear algorithm:', ['regression', 'classification'])
    lr = LinearRegression() if linear_algo == 'regression' else LogisticRegression()
    if not lr:
        st.error('Please select a Linear Algorithm.')
        st.stop()
    if type(lr) != LinearRegression:
        st.write('Only Linear Regression model supported at this moment. Please try again later!')
        st.stop()

    error_metric = None
    if type(lr) == LinearRegression:
      error_metric = error_metric_type.selectbox('Select Residual Metric:', ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
      if not error_metric:
          st.error('Please select an error metric.')
          st.stop()
    elif type(lr) == LogisticRegression:
      error_metric = error_metric_type.selectbox('Select Residual Metric:', ['accuracy', 'precision', 'recall', 'f1'])
      if not error_metric:
          st.error('Please select an error metric.')

    return lr, error_metric, k

def compute_sfs(lr, error_metric, k, X, y):
    '''
        Compute Sequential Forward Selection (SFS) metrics.
    '''
    sfs = SFS(lr, k_features=k, forward=True, scoring=error_metric, cv=2)
    sfs = sfs.fit(X, y)

    # SFS metrics 
    sfs_metrics = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

    return sfs_metrics
  
def plot_sfs_metrics(sfs_metrics):
    '''
        Plot the Sequential Forward Selection (SFS) metrics.
    '''
    st.header('Relevant Sequential Forward Selection (SFS) metrics:')
    st.write(sfs_metrics.iloc[-1, 3])
    st.line_chart(data=sfs_metrics, y='avg_score')
    st.write(sfs_metrics)

def linear_regression(df, independent_columns, dependent_column):
    '''
        Perform Linear Regression and return the test residuals.
    '''
    X = df[independent_columns]
    y = dependent_column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    model = LinearRegression()
    model.fit(X_train, y_train)
    tests_predictions = model.predict(X_test)
    test_residuals = y_test - tests_predictions
    return y_test, test_residuals

def scatter_plot(df, independent_columns, dependent_column):
    '''
        Plot the scatter plot of selected features and target.
    '''
    st.header('Scatter plot of selected features and target:')
    plt.figure(figsize=(12, 6))
    fig = px.scatter(df, x=independent_columns, y=dependent_column, trendline='ols')

    # Update the layout
    fig.update_layout(xaxis_title="Independent Variables")
    fig.update_layout(yaxis_title=dependent_column.name)
    st.plotly_chart(fig, use_container_width=True)

def residual_distribution(xlabel, test_residuals):
    '''
        Plot the residual distribution.
    '''
    
    st.header('Residual Distribution Plot:')
    plt.figure(figsize=(12, 6))
    fig1, axes = plt.subplots()
    sns.histplot(test_residuals, bins=20, kde=True, ax=axes)
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    plt.title('Residual Distribution')
    st.pyplot(fig1)

def residual_plot(df, xlabel, y_test, test_residuals):
    '''
        Plot the residual plot.
    '''
    get_residuals_md()

    plt.figure(figsize=(12, 6))
    fig2, axes = plt.subplots()
    sns.scatterplot(data=df, x=y_test, y=test_residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    plt.title('Residual plot')
    st.pyplot(fig2)

def get_residuals_md():
    '''
        Get the residuals plot markdown.
    '''
    st.header('Residual plot:')
    st.subheader("What is a Residual Plot?")
    st.markdown("""
    - A residual plot is a graphical representation that shows the difference (residuals) between the observed values and the predicted values in a regression model. 
    - Residual = Observed Value - Predicted Value
        - ✔ If residuals are randomly spread around zero, the regression model is likely valid.
      - ❌ If residuals show a clear pattern, the model may need a transformation or non-linear approach.
    """)

    st.subheader("How to Interpret a Residual Plot?")
    st.markdown("""
        - ✅ Good Model (Random Residuals)
            - Residuals are randomly scattered around zero.
            - No visible pattern → Linear regression is appropriate.

        - ⚠️ Bad Model (Patterned Residuals)
            - Curved Shape → The relationship might be non-linear.
            - Funnel Shape (Heteroscedasticity) → Variance of residuals increases/decreases.
            - Clusters or Outliers → Model may be biased or missing key predictors.
    """)

def probability_plot(test_residuals):
    '''
        Plot the probability plot for residuals.
    '''
    get_probability_plot_md()

    plt.figure(figsize=(12, 6))
    fig3, axes = plt.subplots(figsize=(6, 8), dpi=100)
    # probplot returns the raw values if needed
    _ = sp.stats.probplot(test_residuals, plot=axes)
    plt.title('Probability Plot')
    st.pyplot(fig3)

def get_probability_plot_md():
    '''
        Get the probability plot markdown.
    '''
    st.header('Probability Plot for Residuals (Q-Q Plot):')
    st.subheader("Why Use a Probability Plot for Residuals?")
    st.markdown("""
    - Identifies normality: If residuals are normally distributed, they should lie along a straight diagonal line in the Q-Q plot.
    - Detects outliers: Deviations from the line suggest skewness or heavy tails.
    - Validates regression assumptions: Helps ensure that the errors are independent and normally distributed (important for linear regression).
    """)

    st.subheader("Interpreting the Q-Q Plot")
    st.markdown("""
    - Straight Line → Residuals are normally distributed. ✅
    - S-Shaped Curve → Skewed residuals (left or right). ❌
    - Extreme Deviations at Ends → Heavy tails (outliers). ❌
    """)

def make_plots(df, X, y, y_test, sfs_metrics, test_residuals):
    '''
        Make all the plots.
    '''
    plot_sfs_metrics(sfs_metrics)
    scatter_plot(df, X.columns, y)
    residual_distribution(y.name, test_residuals)
    residual_plot(df, y.name, y_test, test_residuals)
    probability_plot(test_residuals)

def main():
    '''
        Main function to run the linear regression analysis.
    '''
    file_url = load_csv_file()
    df = load_data(file_url)
    df = data_imputer(df)
    y, X, df = variables_selector(df)
    lr, error_metric, k = feature_lr_selector(X)
    sfs_metrics = compute_sfs(lr, error_metric, k, X, y)
    y_test, test_residuals = linear_regression(df, X.columns, y)
    make_plots(df, X, y, y_test, sfs_metrics, test_residuals)

if __name__ == '__main__':
    main()