import streamlit as st

class LRMarkdown:
    """
    A class to represent a markdown file for Linear Regression Analysis.
    """

    def __init__(self):
        pass

    def get_residuals_md(self):
      '''
          Get the residuals plot markdown.
      '''
      st.header('Residual plot')
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
    
    def get_probability_plot_md(self):
      '''
          Get the probability plot markdown.
      '''
      st.header('Probability Plot for Residuals (Q-Q Plot)')
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