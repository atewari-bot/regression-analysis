class Selector:
    '''
    A class to select the model, error metrics, and other options.
    '''
    def __init__(self):
      self.model = None
      self.lr_error_metrics_options = None
      self.lg_error_metrics_options = None
      self.sfs_metrics_options = None
      self.imputation_strategy = None

    def get_lr_error_metrics_options():
      '''
      Returns the list of error metrics for Linear Regression model.
      '''
      lr_error_metrics_options = {
          'MAE': 'neg_mean_absolute_error',
          'MSE': 'neg_mean_squared_error',
          'RMSE': 'neg_root_mean_squared_error',
          'R2 Score': 'r2'
      }
      return lr_error_metrics_options

    def get_lg_error_metrics_options():
      '''
      Returns the list of error metrics for Logistic Regression model.
      '''
      lg_error_metrics_options = {
          'Accuracy': 'accuracy', 
          'Precision': 'precision', 
          'Recall': 'recall', 
          'F1': 'f1'
      }
      return lg_error_metrics_options

    def get_model_options():
      '''
      Returns the list of models.
      '''
      model = {
          'Linear Regression': 'lr',
          'Polynomial Regression': 'pr',
          'Logistic Regression': 'lg'
      }
      return model

    def get_sfs_metrics_options():
      '''
      Returns the list of metrics for Sequential Forward Selection.
      '''
      sfs_metrics_options = {
          'Average Score': 'avg_score',
          'CI Bound': 'ci_bound',
          'Standard Deviation': 'std_dev',
          'Standard Error': 'std_err'
      }
      return sfs_metrics_options

    def get_imputation_options():
      '''
      Returns the list of imputation strategies.
      '''
      imputation_strategy = {
          'Mean': 'mean',
          'Median': 'median',
          'Mode': 'mode'
      }
      return imputation_strategy