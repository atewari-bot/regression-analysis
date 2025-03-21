import matplotlib.pyplot as plt
import streamlit as st

class LineChart:
  '''
      Class to plot line chart.
  '''
  def __init__(self, md=None):
    self.md = md

  def sfs_metrics_chart(self, sfs_metrics, selected_sfs_metric):
    '''
        Plot the Sequential Forward Selection (SFS) metrics.
    '''
    st.header('Relevant Sequential Forward Selection (SFS) metrics')
    st.write(sfs_metrics.iloc[-1, 3])
    plt.figure(figsize=(12, 8))
    st.line_chart(data=sfs_metrics, y=selected_sfs_metric)
    st.write(sfs_metrics)