import matplotlib.pyplot as plt
import streamlit as st
import scipy as sp

class QQPlot:
    '''
    Class for creating QQ plots
    '''
    def __init__(self, md=None):
        self.md = md

    def probability_plot(self, test_residuals):
        '''
            Plot the probability plot for residuals.
        '''
        self.md.get_probability_plot_md()

        plt.figure(figsize=(12, 6))
        fig3, axes = plt.subplots(figsize=(6, 8), dpi=100)
        # probplot returns the raw values if needed
        _ = sp.stats.probplot(test_residuals, plot=axes)
        plt.title('Probability Plot')
        st.pyplot(fig3)