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
from streamlit_shap import st_shap
import shap
from utils import color_scale, compute_view

def make_map(geo_df: pd.DataFrame, map_feature: str):
    geo_df_copy = geo_df.copy()

    label = map_feature
    geo_df_copy.dropna(inplace=True)
    feat_series = geo_df_copy[label]
    feat_type = None

    # Assign colors to categorical / numerical values
    if feat_series.dtype == 'object':
        try:
            color_lookup = pdk.data_utils.assign_random_colors(geo_df_copy[map_feature])
            geo_df_copy['fill_color'] = geo_df_copy.apply(lambda row: color_lookup.get(row[map_feature]), axis=1)
        except TypeError:
            normalized_vals = feat_series
            colors = list(map(color_scale, normalized_vals))
            geo_df_copy['fill_color'] = colors
            geo_df_copy.fillna(0, inplace=True)
            geo_df_copy = geo_df_copy.astype({label: 'float64'})
    else:
        normalized_vals = feat_series
        colors = list(map(color_scale, normalized_vals))
        geo_df_copy['fill_color'] = colors
        geo_df_copy.fillna(0, inplace=True)
        geo_df_copy = geo_df_copy.astype({label: 'float64'})

    
    geo_df_copy = geo_df_copy.set_crs(epsg=3035, allow_override=True)
    geo_df_copy = geo_df_copy.to_crs(epsg=4326)    
    tooltip = {"html": ""}
    geo_df_copy = geo_df_copy.to_crs(epsg=4326)
    m = folium.Map(**compute_view(geo_df_copy.geometry))
    geo_df_copy.explore(m=m, name="Neighborhoods", color="red", column=map_feature, tooltip=[map_feature, 'Euros per square meter', 'Neighborhood ID'])
    folium.LayerControl().add_to(m)
    folium_static(m)

def make_chart(df: pd.DataFrame, feature: str):
    data_df = pd.DataFrame(df[[feature, 'Neighborhood ID']])
    data_df2 = df[[feature, 'Euros per square meter']]
    # feat_type = 'category' if data_df[feature].dtype == 'object' else 'numerical'
    # if feat_type == 'category':
    #     print("Categorical Data called on make_chart")
    # else:
    label = feature
    data_df = data_df.round(3) 
    data_df2 = data_df2.round(3)
    data_df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    data_df2 = data_df2.dropna() 

    bar = alt.Chart(data_df) \
        .mark_bar() \
        .encode(x='Neighborhood ID',
                y=label + ':Q',
                tooltip=['Neighborhood ID', label])
    st.altair_chart(bar, use_container_width=True)
    corr_coef = data_df2.corr().iloc[0,1]
    scatter_plot = alt.Chart(data_df2) \
        .mark_point() \
        .encode(x=label + ':Q',
                y='Euros per square meter:Q',
                tooltip=[label, 'Euros per square meter']) 
    reg_line = scatter_plot.transform_regression(label, 'Euros per square meter').mark_line(color='red')
    corr_text = scatter_plot.transform_regression(label, 'Euros per square meter').mark_text(
        align='left', baseline='top', dx=5, dy=-5
    ).encode(
        text=alt.Text(corr_coef)
    )
    corr_text = alt.Chart({'values':[{}]}).mark_text(
    align="left", baseline="top"
).encode(
    x=alt.value(5),  # pixels from left
    y=alt.value(5),  # pixels from top
    text=alt.value(f"r: {corr_coef:.3f}"),
)

    # combine the scatter plot, regression line, and correlation coefficient text
    chart = scatter_plot + reg_line
    st.altair_chart(chart, use_container_width=True)
    st.write("Correlation coefficient:", corr_coef)

def feature_exploration_charts(shap_values: np.array):
    st.write('''
            ### Global feature importance
            Here you can see the most important features that contribute to the housing price, according to our prediction model.
            ''')

    st_shap(shap.plots.beeswarm(shap_values), height=500, width=1200)

def feature_comparison_charts(feature_x: str, feature_color: str, shap_values: np.array):
    st_shap(shap.plots.scatter(shap_values[:,feature_x], color=shap_values[:,feature_color]), height=500, width=1200)
    
def make_force_plot(df: pd.DataFrame, row: int, shap_values: np.array):
    st_shap(shap.plots.waterfall(shap_values[row]), height=500, width=1200)