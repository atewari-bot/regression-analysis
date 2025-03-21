import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from utils.selector_options import SelectorOptions
from eda.load_data import DAO
from eda.imputator import Imputator
from ui.labels import UILabels
from ui.component.selector import UISelector
from ui.markdown.lr_md import LRMarkdown
from ui.component.sidebar import SideBar
from ui.charts.residual_plot import ResidualPlot
from ui.charts.line_chart import LineChart
from ui.charts.scatter_plot import ScatterPlot
from ui.charts.qq_plot import QQPlot


class LinearRegressionAnalysis:
    '''
        Class to perform Linear Regression Analysis.
    '''
    def __init__(self):
        self.ui_labels = UILabels()
        self.dao = DAO()
        self.selector_options = SelectorOptions()
        self.ui_selector = UISelector(validator=self.dao.validator)
        self.encoder = LabelEncoder()
        self.lr_md = LRMarkdown()
        self.sidebar_ui = SideBar()
        self.imputator = Imputator(selector=self.selector_options, dao=self.dao)
        self.residual_plot = ResidualPlot(md=self.lr_md)
        self.line_chart = LineChart(md=self.lr_md)
        self.scatter_plot = ScatterPlot(md=self.lr_md)
        self.qq_plot = QQPlot(md=self.lr_md)

        self.ui_labels.set_app_title()

    def feature_selection(self, df):
        '''
            Define multi selector for  dependent and independent variables.
        '''
        chosen_y, chosen_x, dependent_var_col, independent_var_col = self.ui_selector.variable_selector(df)
        y = df.loc[:, chosen_y]
        dependent_var_col.write(y.head())
        Z = df.loc[:, chosen_x]
        independent_var_col.write(Z.head())
        df = self.label_encoder(df.loc[:, chosen_x])
        X = df.loc[:, chosen_x]

        return y, X, df

    def label_encoder(self, df):
        '''
            Perform Label Encoding on the categorical columns.
        '''
        # Apply to each object column
        cat_cols = df.select_dtypes(include=['object'])
        for col in cat_cols:
            df[col] = self.encoder.fit_transform(df[col])
        
        if not cat_cols.empty:
            self.ui_labels.label_encoded()
            st.write(df.head())
        return df

    def feature_lr_selector(self, X):
        '''
            Select the number of features, linear regression algorithm and error metric.
        '''
        features, model_algo_type, error_metric_type = st.columns(3)
        lr = self.ui_selector.model_algo_selector(self.selector_options, model_algo_type)
        error_metric_options, error_metric_key = self.ui_selector.residual_metric_selector(lr, self.selector_options, error_metric_type)
        error_metric_value = error_metric_options[error_metric_key]

        return lr, error_metric_value, self.ui_selector.get_slider_value_selector(X, features)

    def compute_sfs(self, lr, cv, error_metric, k, X, y):
        '''
            Compute Sequential Forward Selection (SFS) metrics.
        '''
        sfs = SFS(lr, k_features=k, forward=True, scoring=error_metric, cv=cv)
        sfs = sfs.fit(X, y)
        # SFS metrics 
        sfs_metrics = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
        return sfs_metrics

    def regression_analysis(self, df, independent_columns, dependent_column):
        '''
            Perform Regression analysis and return the test residuals.
        '''
        X = df[independent_columns]
        y = dependent_column
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        model = LinearRegression()
        model.fit(X_train, y_train)
        tests_predictions = model.predict(X_test)
        test_residuals = y_test - tests_predictions
        return y_test, test_residuals

    def make_plots(self, df, X, y, y_test, sfs_metrics, selected_sfs_metric, test_residuals, plot_flags):
        '''
            Make all the plots.
        '''
        self.line_chart.sfs_metrics_chart(sfs_metrics, selected_sfs_metric)
        self.scatter_plot.selected_features_plot(df, X.columns, y)

        if plot_flags[2]:
            self.residual_plot.residual_distribution(y.name, test_residuals)
        if plot_flags[0]:
            self.residual_plot.draw_residual_plot(df, y.name, y_test, test_residuals)
        if plot_flags[1]:
            self.qq_plot.probability_plot(test_residuals)

    def sidebar(self, files):
        '''
            Sidebar for the application.
        '''
        sidebar_config = self.sidebar_ui.build_sidebar(files, self.ui_selector, self.selector_options)
        return sidebar_config['selected_sfs_metrics'], sidebar_config['cv'], sidebar_config['selected_file'], sidebar_config['plot_flags']
    
def main():
    '''
        Main function to run the regression analysis.
    '''
    lra = LinearRegressionAnalysis()

    files = lra.dao.list_files()
    selected_sfs_metric, cv, selected_file, plot_flags = lra.sidebar(files)
    file_path = lra.dao.load_csv_file(selected_file)
    df = lra.dao.load_data(file_path)
    df = lra.imputator.data_imputer(df)

    y, X, df = lra.feature_selection(df)
    lr, error_metric, k = lra.feature_lr_selector(X)
    sfs_metrics = lra.compute_sfs(lr, cv, error_metric, k, X, y)
    y_test, test_residuals = lra.regression_analysis(df, X.columns, y)
    lra.make_plots(df, X, y, y_test, sfs_metrics, selected_sfs_metric, test_residuals, plot_flags)

if __name__ == '__main__':
    main()