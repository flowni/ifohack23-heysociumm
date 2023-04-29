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

    print(geo_df_copy.head())
    feat_series = df[label]
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
    # view_state = pdk.ViewState(
    #         latitude=geo_df_copy['geometry'].centroid.y[0], longitude=geo_df_copy['geometry'].centroid.x[0],
    #            zoom=5, maxZoom=16)
    def compute_view(polygons):
        bounds = np.array([list(p.bounds) for p in polygons])
        min_lon, min_lat = bounds[:, 0].astype(float).min(), bounds[:, 1].astype(float).min()
        max_lon, max_lat = bounds[:, 2].astype(float).max(), bounds[:, 3].astype(float).max()
        center_lon, center_lat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
        zoom = 10
        return pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=zoom
        )
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
    view_state = compute_view(
    geo_df_copy.geometry,  # explode the geometry column to obtain all the polygons
    )
    view_state = compute_view(geo_df_copy.geometry)
    view_state = pdk.ViewState(
        latitude=53.084755,
        longitude=8.82079,
        zoom=10
    )
    tooltip = {"html": ""}
    geo_df_copy = geo_df_copy.to_crs(epsg=4326)
    print(geo_df_copy.columns)
    # polygon_layer = pdk.Layer(
    #     "PolygonLayer",
    #     geo_df_copy[['geometry']],
    #     get_polygon="geometry.coordinates",
    #     filled=True,
    #     stroked=True,
    #     get_fill_color=[255, 255, 0],
    #     opacity=1,
    #     pickable=True,
    #     auto_highlight=True
    # )
    # layers = [polygon_layer]
    # if show_transit:
    #     transit_layers = make_transit_layers(tract_df=df, pickable=False)
    #     layers += transit_layers
    # r = pdk.Deck(
    #     layers=layers,
    #     initial_view_state=view_state,
    #     map_style=pdk.map_styles.LIGHT,
    #     tooltip=tooltip
    # )
    # st.pydeck_chart(r)
    m = folium.Map(**compute_view2(geo_df_copy.geometry))
    geo_df_copy.explore(m=m, name="Neighborhoods", color="red", column=map_feature, tooltip=[map_feature, 'Euros per square meter', 'Neighborhood ID'])
    folium.LayerControl().add_to(m)
    folium_static(m)

def make_transit_layers(tract_df: pd.DataFrame, pickable: bool = True):
    tracts = tract_df['Census Tract'].to_list()
    tracts_str = str(tuple(tracts)).replace(',)', ')')

    NTM_shapes = queries.get_transit_shapes_geoms(
        columns=['route_desc', 'route_type_text', 'length', 'geom', 'tract_id', 'route_long_name'],
        where=f" tract_id IN {tracts_str}")

    tolerance = 0.0000750
    NTM_shapes['geom'] = NTM_shapes['geom'].apply(lambda x: x.simplify(tolerance, preserve_topology=False))

    NTM_stops = queries.get_transit_stops_geoms(columns=['stop_name', 'stop_lat', 'stop_lon', 'geom'],
                                                where=f" tract_id IN {tracts_str}")

    NTM_shapes.drop_duplicates(subset=['geom'])
    NTM_stops.drop_duplicates(subset=['geom'])

    if NTM_shapes.empty:
        st.write("Transit lines have not been identified for Equity Geographies in this region.")
        line_layer = None
    else:
        NTM_shapes['path'] = NTM_shapes['geom'].apply(utils.coord_extractor)
        NTM_shapes.fillna("N/A", inplace=True)

        route_colors = {}
        for count, value in enumerate(NTM_shapes['route_type_text'].unique()):
            route_colors[value] = COLOR_VALUES[count]
        NTM_shapes['color'] = NTM_shapes['route_type_text'].apply(lambda x: route_colors[x])
        NTM_shapes['alt_color'] = NTM_shapes['color'].apply(lambda x: "#%02x%02x%02x" % (x[0], x[1], x[2]))

        # REMOVED LEGEND BECAUSE IT LOOKED BUSY
        # bar = alt.Chart(
        #     NTM_shapes[['length', 'route_type_text', 'alt_color', 'tract_id', 'route_long_name']]).mark_bar().encode(
        #     y=alt.Y('route_type_text:O', title=None, axis=alt.Axis(labelFontWeight='bolder')),
        #     # column=alt.Column('count(length):Q', title=None, bin=None), 
        #     x=alt.X('tract_id:N', title='Census Tracts', axis=alt.Axis(orient='top', labelAngle=0)),
        #     color=alt.Color('alt_color', scale=None),
        #     tooltip=['tract_id']) \
        #     .interactive()

        # st.altair_chart(bar, use_container_width=True)

        line_layer = pdk.Layer(
            "PathLayer",
            NTM_shapes,
            get_color='color',
            get_width=12,
            # highlight_color=[176, 203, 156],
            picking_radius=6,
            auto_highlight=pickable,
            pickable=pickable,
            width_min_pixels=2,
            get_path="path"
        )

    if NTM_stops.empty:
        st.write("Transit stops have not been identified for Equity Geographies in this region.")
        stop_layer = None
    else:
        stop_layer = pdk.Layer(
            'ScatterplotLayer',
            NTM_stops,
            get_position=['stop_lon', 'stop_lat'],
            auto_highlight=pickable,
            pickable=False,
            get_radius=36,
            get_fill_color=[255, 140, 0],
        )

    return [
        line_layer,
        stop_layer
    ]

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
