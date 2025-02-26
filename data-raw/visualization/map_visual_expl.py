# This file loads spatial files and creates and interactive map
# For the South Delta Gates 

import geopandas as gpd
import folium
from IPython.display import display
import pandas as pd

import os
print(os.getcwd())


shapefile_paths = [
    "../MSS_nodes/dsm2_nodes_newcs_extranodes.shp",
    "../fc2024.01_chan/FC2024.01_channels_centerlines.shp"
]

nodes_to_highlight = [112, 176, 69]

nodes = gpd.read_file(shapefile_paths[0])
channels = gpd.read_file(shapefile_paths[1])

nodes_filter = nodes[nodes['id'].isin(nodes_to_highlight)]

channels_with_numbers = pd.read_csv('../channel_names_from_h5.csv')
channels_with_numbers = channels_with_numbers.rename(columns={'chan_no': 'id'})

channels_merge = pd.merge(
    channels,
    channels_with_numbers,
    how='left',
    left_on='id',
    right_on='id'
)

filtered_channels = channels_merge[channels_merge['id'].isin([211, 79, 134])]


def create_multi_layer_map(shapefile_paths, filtered_gdf=None, filtered_polylines=None):
    """
    Creates an interactive map with multiple shapefile layers, optional filtered points in red, 
    and optional filtered polylines with a distinct layer.
    
    Parameters:
    - shapefile_paths: List of file paths to the shapefiles.
    - filtered_gdf: A GeoDataFrame with filtered points to display in red (optional).
    - filtered_polylines: A GeoDataFrame with filtered polylines to display as a separate layer (optional).
    """
    # Initialize a list to store valid GeoDataFrames and their centroids
    gdfs = []
    all_centroids = []
    
    for path in shapefile_paths:
        # Read shapefile
        gdf = gpd.read_file(path)
        
        # Ensure the CRS is EPSG:4326
        if gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        
        # Remove rows with invalid or missing geometries
        gdf = gdf[~gdf.geometry.isna()]
        
        # Compute the centroid in EPSG:4326 (no need for reprojection)
        centroid = gdf.geometry.centroid
        center = [centroid.y.mean(), centroid.x.mean()]
        
        # Append to the list
        gdfs.append((gdf, center))
        all_centroids.append(center)
    
    # Check and transform filtered_gdf to EPSG:4326 if provided
    if filtered_gdf is not None:
        if filtered_gdf.crs != "EPSG:4326":
            filtered_gdf = filtered_gdf.to_crs("EPSG:4326")
        filtered_gdf = filtered_gdf[~filtered_gdf.geometry.isna()]  # Remove invalid geometries

    # Check and transform filtered_polylines to EPSG:4326 if provided
    if filtered_polylines is not None:
        if filtered_polylines.crs != "EPSG:4326":
            filtered_polylines = filtered_polylines.to_crs("EPSG:4326")
        filtered_polylines = filtered_polylines[~filtered_polylines.geometry.isna()]  # Remove invalid geometries
    
    # Calculate map center based on all layers
    avg_lat = sum([c[0] for c in all_centroids]) / len(all_centroids)
    avg_lon = sum([c[1] for c in all_centroids]) / len(all_centroids)
    
    # Create a Folium map centered on the average centroid
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=10)
    
    # Add each shapefile as a separate layer to the map
    for idx, (gdf, _) in enumerate(gdfs):
        feature_group = folium.FeatureGroup(name=f"Layer {idx + 1}: {shapefile_paths[idx].split('/')[-1]}")
        for _, row in gdf.iterrows():
            if row.geometry.type == 'Point':
                # Add points as markers
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=f"ID: {row.get('id', 'N/A')}<br>Name: {row.get('name', 'N/A')}",
                    icon=folium.Icon(color='blue', icon="circle")
                ).add_to(feature_group)
            elif row.geometry.type == 'LineString':
                # Add polylines
                coordinates = [(point[1], point[0]) for point in row.geometry.coords]  # Convert to (lat, lon)
                popup_content = f"""
                id = {row.get('id', 'N/A')}<br>
                name = {row.get('name', 'N/A')}<br>
                distance = {row.get('distance', 'N/A')}<br>
                variable = {row.get('variable', 'N/A')}<br>
                interval = {row.get('interval', 'N/A')}<br>
                period_op = {row.get('period_op', 'N/A')}
                """
                folium.PolyLine(
                    locations=coordinates,
                    color='darkgreen',
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(feature_group)
        feature_group.add_to(m)
    
    # If filtered_gdf is provided, add the filtered points in red
    if filtered_gdf is not None:
        filtered_points_group = folium.FeatureGroup(name="Filtered Points")
        for _, row in filtered_gdf.iterrows():
            if row.geometry.type == 'Point':  # Ensure geometry is a Point
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x], 
                    popup=f"ID: {row.get('id', 'N/A')}", 
                    icon=folium.Icon(color='red', icon="circle")
                ).add_to(filtered_points_group)
        filtered_points_group.add_to(m)
    
    # If filtered_polylines is provided, add the filtered polylines as a separate layer
    if filtered_polylines is not None:
        filtered_lines_group = folium.FeatureGroup(name="Filtered Polylines")
        for _, row in filtered_polylines.iterrows():
            if row.geometry.type == 'LineString':
                coordinates = [(point[1], point[0]) for point in row.geometry.coords]  # Convert to (lat, lon)
                popup_content = f"""
                id = {row.get('id', 'N/A')}<br>
                name = {row.get('name', 'N/A')}<br>
                distance = {row.get('distance', 'N/A')}<br>
                variable = {row.get('variable', 'N/A')}<br>
                interval = {row.get('interval', 'N/A')}<br>
                period_op = {row.get('period_op', 'N/A')}
                """
                folium.PolyLine(
                    locations=coordinates,
                    color='blue',  # Distinct color for filtered polylines
                    popup=folium.Popup(popup_content, max_width=300)
                ).add_to(filtered_lines_group)
        filtered_lines_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display the map in the notebook
    display(m)

create_multi_layer_map(shapefile_paths, filtered_gdf = nodes_filter, filtered_polylines = filtered_channels)

