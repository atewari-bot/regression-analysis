def get_lr_error_metrics_options():
    '''
    Returns the list of error metrics for Linear Regression model.
    '''
    return {
        'MAE': 'neg_mean_absolute_error',
        'MSE': 'neg_mean_squared_error',
        'RMSE': 'neg_root_mean_squared_error',
        'R2 Score': 'r2'
    }

def get_lg_error_metrics_options():
    '''
    Returns the list of error metrics for Logistic Regression model.
    '''
    return {
        'Accuracy': 'accuracy', 
        'Precision': 'precision', 
        'Recall': 'recall', 
        'F1': 'f1'
    }

def get_model_options():
    '''
    Returns the list of models.
    '''
    return {
        'Linear Regression': 'lr',
        'Polynomial Regression': 'pr',
        'Logistic Regression': 'lg'
    }

def get_sfs_metrics_options():
    '''
    Returns the list of metrics for Sequential Forward Selection.
    '''
    return {
        'Average Score': 'avg_score',
        'CI Bound': 'ci_bound',
        'Standard Deviation': 'std_dev',
        'Standard Error': 'std_err'
    }

def get_imputation_options():
    '''
    Returns the list of imputation strategies.
    '''
    return {
        'Mean': 'mean',
        'Median': 'median',
        'Mode': 'mode'
    }