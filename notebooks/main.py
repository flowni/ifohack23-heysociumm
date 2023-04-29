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

# Load the data for Cologne
filtered_zensus_data = gpd.read_file('../data_example/bremen_merged_data.gpkg')
zensus_data_by_neighborhood_grouped = filtered_zensus_data.groupby('Neighborhood_FID')

# Create a map centered on Cologne
map = folium.Map(location=[50.938, 6.959], zoom_start=12)
# Group the data by geometry and calculate the mean of the BRW column for each group
zensus_data_by_geometry = gpd.GeoDataFrame(zensus_data_by_neighborhood_grouped).dissolve(by='Neighborhood_FID', aggfunc='sum').reset_index()
# Define a function that creates the sidebar
def create_sidebar(selected_neighborhood):
    neighborhood_data = zensus_data_by_geometry[zensus_data_by_geometry['Neighborhood_FID'] == selected_neighborhood]
    st.sidebar.markdown(f"### {selected_neighborhood}")
    st.sidebar.write(neighborhood_data)

# Define a function that will be called when a neighborhood is clicked on the map
def on_click(feature, **kwargs):
    selected_neighborhood = feature['id']
    create_sidebar(selected_neighborhood)

# Add the neighborhoods as polygons to the map
folium.GeoJson(zensus_data_by_geometry[['Land_Value', 'geometry']].to_json(),
                name='choropleth',
                on_click=on_click,
                highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.6},
                smooth_factor=2.0).add_to(map)

st_folium()