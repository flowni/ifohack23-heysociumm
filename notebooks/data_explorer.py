import pandas as pd
import geopandas as gpd
import folium 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import linregress
import folium
from scipy.stats import pearsonr
import streamlit as st
from streamlit_folium import folium_static, st_folium
from pysal.lib import weights  
import segregation as seg
import pydeck as pdk
import altair as alt

def county_data_explorer(df):
    df = zensus_data_by_neighborhood_grouped
    if df is not None:
        st.write('''### View Feature''')
        temp = df.copy()
        temp.reset_index(inplace=True)
        feature_labels = list(
            set(temp.columns) - {'County Name', 'State', 'county_id', 'state_id', 'pop10_sqmi', 'pop2010','fips','cnty_fips','state_fips'})
        feature_labels.sort()
        single_feature = st.selectbox('Feature', feature_labels, 0)
        make_map(temp, temp, single_feature)
        make_chart(temp, single_feature, st.session_state.data_format)
        
        st.write('''
            ### Compare Features
            Select two features to compare on the X and Y axes. Only numerical data can be compared.
            ''')
        col1, col2, col3 = st.columns(3)
        with col1:
            feature_1 = st.selectbox('X Feature', feature_labels, 0)
        with col2:
            feature_2 = st.selectbox('Y Feature', feature_labels, 1)
        with col3:
            scaling_feature = st.selectbox('Scaling Feature', feature_labels, len(feature_labels) - 1)
        #if feature_1 and feature_2 and scaling_feature:
        #    visualization.make_scatter_plot_counties(temp, feature_1, feature_2, scaling_feature, st.session_state.data_format)
        #temp.drop(['State', 'County Name', 'county_id'], inplace=True, axis=1)
        #visualization.make_correlation_plot(temp, feature_labels)