import streamlit as st

class SideBar:
  '''
      Class to build the sidebar for the application.
  '''
  def __init__(self):
    pass

  def build_sidebar(self, files, ui_selector, selector_options):
    '''
        Sidebar for the application.
    '''
    st.sidebar.title('Regression Analysis')
    st.sidebar.subheader('About')
    st.sidebar.markdown("""
        This application performs Regression analysis on the selected dataset.
    """)
    
    with st.sidebar:
      st.header("Customize Regression Analysis")
      default_file_selection = ['Advertising.csv']
      selected_file = st.selectbox('Choose a Dataset (.csv)', files, index=files.index(default_file_selection[0]))
      cv = st.number_input("Enter cross validation value", min_value=0, max_value=10, value=2, step=1)
      if cv == 1:
          st.error('Cross validation value should not be 1.')

      show_residual_plot = st.checkbox("Show Residual Plot", value=False)
      show_prob_plot = st.checkbox("Show Probability Plot", value=False)
      show_res_dist_plot = st.checkbox("Show Residual Distribution Plot", value=False)

      selected_sfs_metrics = ui_selector.sfs_metrics_selector(selector_options)
      return {
        'cv': cv, 
        'selected_file': selected_file, 
        'selected_sfs_metrics': selected_sfs_metrics, 
        'plot_flags': [show_residual_plot, show_prob_plot, show_res_dist_plot]
        }