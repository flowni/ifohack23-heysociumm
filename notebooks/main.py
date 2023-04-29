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


# Load the data for Cologne
filtered_zensus_data = gpd.read_file('../data_example/bremen_merged_data.gpkg')
zensus_data_by_neighborhood_grouped = filtered_zensus_data

COLOR_RANGE = [
    [65, 182, 196],
    [127, 205, 187],
    [199, 233, 180],
    [237, 248, 177],
    [255, 255, 204],
    [255, 237, 160],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
]

COLOR_VALUES=[
    [13, 59, 102],
    [238, 150, 75],
    [249, 87, 56],
    [0,0,128],
    [210,105,30],
    [220,20,60],
    [250,128,114],
    [0,100,0],
    [255,215,0],
    [50,205,50],
    [47,79,79],
    [0,255,255],
    [0,0,255],
    [75,0,130],
    [255,0,255],
    [255,192,203],
]

BREAKS = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

def color_scale(val: float) -> list:
    for i, b in enumerate(BREAKS):
        if val <= b:
            return COLOR_RANGE[i]
    return COLOR_RANGE[i]

def make_map(geo_df: pd.DataFrame, df: pd.DataFrame, map_feature: str, data_format: str = 'Raw Values',
             show_transit: bool = False):
    geo_df_copy = geo_df.copy()

    label = map_feature
    geo_df_copy.dropna(inplace=True)
    feat_series = geo_df_copy[label]
    feat_type = None

    if feat_series.dtype == 'object':
        try:
            # feat_dict = {k: (i % 10) / 10 for i, k in enumerate(
            #     feat_series.unique())}  # max 10 categories, following from constants.BREAK, enumerated rather than encoded
            # normalized_vals = feat_series.apply(lambda x: feat_dict[x])  # getting normalized vals, manually.
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
    def compute_view2(polygons):
        bounds = np.array([list(p.bounds) for p in polygons])
        min_lon, min_lat = bounds[:, 0].astype(float).min(), bounds[:, 1].astype(float).min()
        max_lon, max_lat = bounds[:, 2].astype(float).max(), bounds[:, 3].astype(float).max()
        center_lon, center_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
        zoom = 10
        return {
            "min_lat": min_lat,
            "min_lat": max_lat,
            "min_lon": min_lon,
            "max_lon": max_lon,
            "location": [center_lat, center_lon]
        }
    geo_df_copy = geo_df_copy.set_crs(epsg=3035, allow_override=True)
    geo_df_copy = geo_df_copy.to_crs(epsg=4326)    
    tooltip = {"html": ""}
    geo_df_copy = geo_df_copy.to_crs(epsg=4326)
    print(geo_df_copy.columns)
    m = folium.Map(**compute_view2(geo_df_copy.geometry))
    geo_df_copy.explore(m=m, name="Neighborhoods", color="red", column=map_feature, tooltip=[map_feature, 'Euros per square meter', 'Neighborhood ID'])
    folium.LayerControl().add_to(m)
    folium_static(m)

def run_UI():
    st.title("Data Explorer")
    subcol_1, subcol_2 = st.columns(2)
    with subcol_1:
        st.session_state.data_type = st.radio("Data resolution:", ('County Level', 'Census Tracts'), index=0)
    with subcol_2:
        # Todo: implement for census level too
        if st.session_state.data_type =='County Level':
            st.session_state.data_format = st.radio('Data format', ['Raw Values', 'Per Capita', 'Per Square Mile'], 0)
    county_data_explorer()

def make_chart(df: pd.DataFrame, feature: str, data_format: str = 'Raw Values'):
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

def county_data_explorer():
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
if __name__ == '__main__':
    run_UI()
