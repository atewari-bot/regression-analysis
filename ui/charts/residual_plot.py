import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class ResidualPlot:
  '''
      Class to plot residual plot.
  '''
  def __init__(self, md=None):
    self.md = md
    
  def draw_residual_plot(self, df, xlabel, y_test, test_residuals):
    '''
        Plot the residual plot.
    '''
    self.md.get_residuals_md()

    plt.figure(figsize=(12, 6))
    fig2, axes = plt.subplots()
    sns.scatterplot(data=df, x=y_test, y=test_residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    plt.title('Residual plot')
    st.pyplot(fig2)

  def residual_distribution(self, xlabel, test_residuals):
    '''
        Plot the residual distribution.
    '''
    st.header('Residual Distribution Plot')
    plt.figure(figsize=(12, 6))
    fig1, axes = plt.subplots()
    sns.histplot(test_residuals, bins=20, kde=True, ax=axes)
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    plt.title('Residual Distribution')
    st.pyplot(fig1)