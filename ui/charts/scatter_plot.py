import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

class ScatterPlot:
  '''
  Class for Scatter Plot
  '''
  def __init__(self, md=None):
    self.md = md

  def selected_features_plot(self, df, independent_columns, dependent_column):
    '''
        Plot the scatter plot of selected features and target.
    '''
    st.header('Scatter plot of selected features and target')
    plt.figure(figsize=(12, 6))
    fig = px.scatter(df, x=independent_columns, y=dependent_column, trendline='ols')

    # Update the layout
    fig.update_layout(xaxis_title="Feature Variables")
    fig.update_layout(yaxis_title=dependent_column.name)
    fig.update_layout(legend=dict(
        title="Features"
    ))
    st.plotly_chart(fig, use_container_width=True)