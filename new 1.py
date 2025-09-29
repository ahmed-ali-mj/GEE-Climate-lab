import streamlit as st
import ee
import geemap
import pandas as pd

# Initialize Earth Engine (make sure you've authenticated first)
try:
    ee.Initialize()
except:
    st.error("Please authenticate Earth Engine first. Run: earthengine authenticate")
    st.stop()

# Your existing data
# combinedExposureScore = era5MonthlyTemp.map(...)  # Your scoring function

def check_transmission_line_exposure(transmission_lines, exposure_score, threshold):
    """
    Check if transmission lines cross areas with exposure score below threshold
    
    Args:
        transmission_lines: ee.FeatureCollection of transmission lines
        exposure_score: ee.ImageCollection with exposure scores
        threshold: Minimum acceptable score value
    
    Returns:
        ee.FeatureCollection with exposure information
    """
    # Convert ImageCollection to single Image (mean or max depending on your needs)
    exposure_image = exposure_score.mean()  # or .max(), .min() depending on your use case
    
    # Create mask for areas with score below threshold
    low_score_mask = exposure_image.lt(threshold)
    
    # Convert mask to vector (polygons)
    low_score_areas = low_score_mask.reduceToVectors(
        geometry=transmission_lines.geometry().bounds(),
        scale=1000,  # Adjust scale based on your data resolution
        geometryType='polygon',
        eightConnected=False,
        labelProperty='low_score'
    )
    
    # Check intersection between transmission lines and low score areas
    def check_intersection(feature):
        line = feature.geometry()
        # Check if line intersects any low score area
        intersects = low_score_areas.filterBounds(line).size()
        has_low_score = intersects.gt(0)
        
        # Calculate minimum score along the line (optional)
        line_scores = exposure_image.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=line,
            scale=1000,
            bestEffort=True
        )
        
        return feature.set({
            'crosses_low_score': has_low_score,
            'min_score': line_scores.get('your_score_band_name'),  # Replace with actual band name
            'line_length_km': line.length().divide(1000)  # Length in km
        })
    
    return transmission_lines.map(check_intersection)

def main():
    st.title("Transmission Line Exposure Analysis")
    
    # Load your data (replace with your actual data loading)
    # transmission_lines = ee.FeatureCollection('your_transmission_lines_asset_id')
    # combinedExposureScore = ee.ImageCollection('your_exposure_score_asset_id')
    
    # For demonstration, let's create some sample data
    st.sidebar.header("Configuration")
    
    # Threshold input
    threshold = st.sidebar.slider(
        "Minimum acceptable score threshold",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )
    
    # Sample data creation (replace with your actual data)
    # Create sample transmission lines
    transmission_lines = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.LineString([[-100, 40], [-95, 35], [-90, 30]]),
            {'name': 'Line 1', 'voltage': '500kV'}
        ),
        ee.Feature(
            ee.Geometry.LineString([[-105, 45], [-100, 40], [-95, 35]]),
            {'name': 'Line 2', 'voltage': '230kV'}
        )
    ])
    
    # Create sample exposure scores (replace with your combinedExposureScore)
    def create_sample_exposure(date):
        # Create random exposure pattern for demonstration
        return ee.Image.random().multiply(100).rename('exposure_score')
    
    combinedExposureScore = ee.ImageCollection(
        ee.List.sequence(0, 11).map(create_sample_exposure)
    )
    
    # Perform analysis
    with st.spinner("Analyzing transmission line exposure..."):
        result = check_transmission_line_exposure(
            transmission_lines, 
            combinedExposureScore, 
            threshold
        )
    
    # Display results
    st.header("Analysis Results")
    
    # Convert to pandas dataframe for display
    result_info = result.getInfo()
    features = result_info['features']
    
    data = []
    for feature in features:
        props = feature['properties']
        data.append({
            'Line Name': props.get('name', 'Unknown'),
            'Voltage': props.get('voltage', 'Unknown'),
            'Crosses Low Score Area': props.get('crosses_low_score', False),
            'Minimum Score': props.get('min_score', 'N/A'),
            'Length (km)': props.get('line_length_km', 'N/A')
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    # Display map
    st.header("Interactive Map")
    
    # Create map
    m = geemap.Map()
    
    # Add transmission lines
    m.addLayer(transmission_lines, {'color': 'blue'}, 'Transmission Lines')
    
    # Add exposure scores
    exposure_mean = combinedExposureScore.mean()
    m.addLayer(exposure_mean, {'min': 0, 'max': 100, 'palette': ['green', 'yellow', 'red']}, 'Exposure Score')
    
    # Add low score areas
    low_score_areas = exposure_mean.lt(threshold).selfMask()
    m.addLayer(low_score_areas, {'palette': 'red'}, f'Score < {threshold}')
    
    # Add result highlights
    high_risk_lines = result.filter(ee.Filter.eq('crosses_low_score', True))
    m.addLayer(high_risk_lines, {'color': 'red', 'width': 3}, 'High Risk Lines')
    
    # Display map
    m.to_streamlit(height=600)
    
    # Summary statistics
    st.header("Summary Statistics")
    
    total_lines = df.shape[0]
    at_risk_lines = df[df['Crosses Low Score Area'] == True].shape[0]
    percentage_at_risk = (at_risk_lines / total_lines * 100) if total_lines > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Lines", total_lines)
    col2.metric("Lines at Risk", at_risk_lines)
    col3.metric("Percentage at Risk", f"{percentage_at_risk:.1f}%")

if __name__ == "__main__":
    main()