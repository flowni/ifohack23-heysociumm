import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xgboost as xgb
import altair as alt
import shap
from streamlit_shap import st_shap
from utils import *
from constants import *
from data_explorer import *

# Load the data for Cologne
zensus_data_by_neighborhood_grouped = gpd.read_file('../data_example/bremen_merged_data.gpkg')
# Load the predictor model
predictor_model = xgb.XGBRegressor()
predictor_model.load_model("xgboost_model.json")
# Build the explanations
explainer = shap.Explainer(predictor_model)
X_data = zensus_data_by_neighborhood_grouped.drop(columns=['geometry', 'Area Type', 'Euros per square meter', 'Neighborhood Name'])
clean_rows(X_data)
shap_values = explainer(X_data)

def run_UI():
    st.title("What determines the land prices in Bremen?")
    render_views()

def render_views():
    df = zensus_data_by_neighborhood_grouped
    if df is not None:
        st.write('''### Check the spatial distribution of zensus features''')
        temp = df.copy()
        temp.reset_index(inplace=True)
        feature_labels = list(
            set(temp.columns) - {'Area Type', 'Euros per square meter', 'geometry', 'index', 'Neighborhood ID', 'Neighborhood Name'})
        # Add unselected option to neighborhoods
        neighborhoods = [''] + list(temp["Neighborhood Name"])
        feature_labels.sort()
        single_feature = st.selectbox('Feature', feature_labels, 0)
        make_map(temp, single_feature)
        make_chart(temp, single_feature)
        st.write('''
            ### Looking at key price contributors for a single neighborhood
            Select a neighborhood to see the features that contribute most to its price.
            ''')
        neighborhood = st.selectbox('Neighborhood', neighborhoods, 0)
        if neighborhood != '':
            neighborhood_index = np.where(temp['Neighborhood Name'] == neighborhood)[0][0]
            make_force_plot(temp, neighborhood_index, shap_values)
        st.write('''
            ### Compare Features
            Select two features to compare on the X and Y axes. Only numerical data can be compared.
            ''')
        col1, col2, col3 = st.columns(3)
        with col1:
            feature_1 = st.selectbox('X Feature', feature_labels, 0)
        with col2:
            feature_2 = st.selectbox('Y Feature', feature_labels, 1)
        feature_comparison_charts(feature_1, feature_2, shap_values)
        feature_exploration_charts(shap_values)

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    run_UI()
