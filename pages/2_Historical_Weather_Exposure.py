# ────────────────────────────────────────────────────────────────────────────
# Page#2 :  Historical Weather Exposure Analysis
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
from datetime import date, datetime, timedelta, timezone
import ee
import geemap.foliumap as geemap

st.header('Page 2: Historical Weather Exposure Analysis')

for k, v in st.session_state.items():
    st.session_state[k] = v
    

# Default date range
start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)

# Date range selector
selected_start, selected_end = st.date_input(
    "Select date range",
    (start_date, end_date),
    min_value=date(2000, 1, 1),
    max_value=date.today()
)

st.write("Start date:", selected_start)
st.write("End date:", selected_end)

#st.write("Code reran at:", datetime.now())

selected_start = ee.Date(str(selected_start))
selected_end = ee.Date(str(selected_end))



# Initialize UI state
if "historical_exposure_measure" not in st.session_state:
    st.session_state.historical_exposure_measure = "NumOfExtremeMonthsOverStudyPeriod"

# UI toggle: Measure of historical exposure
st.session_state.historical_exposure_measure = st.radio(
    "Choose the measure for histoircal weather exposure:",
    ["MeanOverStudyPeriod", "NumOfExtremeMonthsOverStudyPeriod"],
    index=["MeanOverStudyPeriod", "NumOfExtremeMonthsOverStudyPeriod"].index(st.session_state.historical_exposure_measure)
)


if st.session_state.historical_exposure_measure == "MeanOverStudyPeriod":
    # Add some space before the map
    #st.markdown("### \n\n")
    st.markdown("---")  # Horizontal divider
    st.markdown("## Map View")
   
    #opacity = st.slider("Select layer opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    #st.write("Opacity: ", opacity)
    opacity = 0.6

    # Create an interactive map
    Map = geemap.Map()
    Map.centerObject(st.session_state.center_point, 5)
    Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
    Map.addLayer(st.session_state.line_assets, {'color': 'black'}, 'Transmission lines');


    #Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
    #Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Transmission lines');
    era5Monthly = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
    era5MonthlyTemp = era5Monthly.select('temperature_2m').filterDate(selected_start).first().clip(st.session_state.roi)
    temp_vis = {
        'min': 260,
        'max': 320,
        'palette': ['green', 'yellow', 'red'],
        'opacity': opacity
    }

    
    Map.addLayer(era5MonthlyTemp, temp_vis, 'ERA5 Mean Monthly Temp')
    
    era5MonthlyPrecip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate(selected_start, selected_end) \
    .select('total_precipitation_sum');

    #Clip each image in the collection to the ROI
    clippedMonthlyPrecip = era5MonthlyPrecip.map(lambda img: img.clip(st.session_state.roi))
    clippedMonthlyPrecipMM = clippedMonthlyPrecip.map(lambda img: img.multiply(1000))

    #print(clippedMonthlyPrecipMM.first().getInfo())

    avgMonthlyPrecip = clippedMonthlyPrecipMM.mean()

    precip_vis = {
        'min': 0,
        'max': 250,
        'palette': ['green', 'yellow', 'orange', 'red'],
        'opacity': opacity
    }
    
    Map.addLayer(avgMonthlyPrecip, precip_vis, 'ERA5 Avg Monthly Precipitation')

    # Add layer control
    Map.addLayerControl()

    # Render the map in Streamlit
    Map.to_streamlit(width=700, height=500)
    

elif st.session_state.historical_exposure_measure == "NumOfExtremeMonthsOverStudyPeriod":

    # Add some space before the map
    #st.markdown("### \n\n")
    st.markdown("---")  # Horizontal divider
    #st.markdown("## Map View")

    
    #high_temp_threshold = st.slider(
    #    "Select maximum acceptable monthly mean temperature (°C)",
    #    min_value=-20.0,
    #    max_value=50.0,
    #    value=25.0,
    #    step=0.5
    #)
    
    
    #low_temp_threshold = st.slider(
    #    "Select minimum acceptable monthly mean temperature (°C)",
    #    min_value=-20.0,
    #    max_value=50.0,
    #    value=10.0,
    #    step=0.5
    #)

    #high_precip_threshold = st.slider(
    #    "Select maximum acceptable monthly rainfall (mm)",
    #    min_value=10,
    #    max_value=250,
    #    value=50,
    #    step=5
    #)
    
    col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

    with col1:
        st.write("")
        st.write("")
        st.write("Select maximum acceptable monthly mean temperature (°C)")

    with col2:
        high_temp_threshold = st.slider(
            label="",
            min_value=-20.0,
            max_value=50.0,
            value=25.0,
            step=0.5
        )
        
    col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

    with col1:
        st.write("")
        st.write("")
        st.write("Select minimum acceptable monthly mean temperature (°C)")

    with col2:
        low_temp_threshold = st.slider(
            label="",
            min_value=-20.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        )
        
    col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

    with col1:
        st.write("")
        st.write("")
        st.write("Select maximum acceptable monthly rainfall (mm)")

    with col2:
        high_precip_threshold = st.slider(
            label="",
            min_value=10,
            max_value=250,
            value=50,
            step=5
        )

    highTempLimit = ee.Number(high_temp_threshold)
    lowTempLimit = ee.Number(low_temp_threshold)
    highPrecipLimit = ee.Number(high_precip_threshold)

    st.write("Provide the weightage of individual factors to calcuate the combined historical weather exposure score:")
    #st.write("(These weights must add up to 1.)")

    #weightHistoricalMaxTempViolations = st.number_input("Weight for max temperature violations:", value=0.34)
    #weightHistoricalMinTempViolations = st.number_input("Weight for min temperature violations:", value=0.33)
    #weightHistoricalRainfallViolations = st.number_input("Weight for rainfall violations:", value=0.33)

    slider_range = st.slider(
        "Adjust the split ",
        0.0, 1.0, (0.2, 0.7), step=0.01
    )

    weightHistoricalMaxTempViolations = slider_range[0]
    weightHistoricalMinTempViolations = slider_range[1] - slider_range[0]
    weightHistoricalRainfallViolations = 1.0 - slider_range[1]

    st.write(f"Weight for max temperature violations = {weightHistoricalMaxTempViolations:.2f}")
    st.write(f"Weight for min temperature violations = {weightHistoricalMinTempViolations:.2f}")
    st.write(f"Weight for rainfall violations = {weightHistoricalRainfallViolations:.2f}")

    #opacity = st.slider("Select layer opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    #st.write("Opacity: ", opacity)
    opacity = 0.6

    Map = geemap.Map()
    Map.centerObject(st.session_state.center_point, 5)
    Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
    Map.addLayer(st.session_state.line_assets, {'color': 'black'}, 'Transmission lines');
    
    
    era5MonthlyTemp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(selected_start, selected_end) \
        .select('temperature_2m');
     
    era5MonthlyPrecip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate(selected_start, selected_end) \
    .select('total_precipitation_sum');

    #Clip each image in the collection to the ROI
    clippedMonthlyTemp = era5MonthlyTemp.map(lambda img: img.clip(st.session_state.roi))
    clippedMonthlyTempCelsius = clippedMonthlyTemp.map(lambda img: img.subtract(273.5))

    clippedMonthlyPrecip = era5MonthlyPrecip.map(lambda img: img.clip(st.session_state.roi))
    clippedMonthlyPrecipMM = clippedMonthlyPrecip.map(lambda img: img.multiply(1000))

    highMonthlyTempIC = clippedMonthlyTempCelsius.map(lambda img: img.gt(highTempLimit))
    numOfHotMonthsImg = highMonthlyTempIC.reduce(ee.Reducer.sum())
    
    lowMonthlyTempIC = clippedMonthlyTempCelsius.map(lambda img: img.lt(lowTempLimit))
    numOfColdMonthsImg = lowMonthlyTempIC.reduce(ee.Reducer.sum())

    highMonthlyPrecipIC = clippedMonthlyPrecipMM.map(lambda img: img.gt(highPrecipLimit))
    numOfWetMonthsImg = highMonthlyPrecipIC.reduce(ee.Reducer.sum())

    n_months = selected_end.difference(selected_start, 'month')

    hotMonthsRatioImg = numOfHotMonthsImg.divide(ee.Number(n_months))
    coldMonthsRatioImg = numOfColdMonthsImg.divide(ee.Number(n_months))
    wetMonthsRatioImg = numOfWetMonthsImg.divide(ee.Number(n_months))

    #combinedHistoricalScore = hotMonthsRatioImg.add(coldMonthsRatioImg).add(wetMonthsRatioImg).divide(3).rename("avg_score")
    combinedHistoricalScore = (hotMonthsRatioImg.multiply(weightHistoricalMaxTempViolations)).add(coldMonthsRatioImg.multiply(weightHistoricalMinTempViolations)).add(wetMonthsRatioImg.multiply(weightHistoricalRainfallViolations))

    ratio_vis = {
        'min': 0,
        'max': 1,
        'palette': [
            "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b",
            "#fdae61", "#f46d43", "#d73027", "#a50026"
        ],
        'opacity': opacity
    }

    # ratio_vis = {
    #     'min': 0,
    #     'max': 1,
    #     'palette': ['green', 'yellow', 'orange', 'red'],
    #     'opacity': opacity
    # }
    
    #Map.addLayer(numOfHotMonthsImg, temp_vis, f"Number of Hot Months ({high_temp_threshold} °C)")  
    #Map.addLayer(numOfColdMonthsImg, temp_vis, f"Number of Cold Months ({low_temp_threshold} °C)")  

    Map.addLayer(hotMonthsRatioImg, ratio_vis, f"Hot Months Ratio ({high_temp_threshold} °C)", shown=False) 
    Map.addLayer(coldMonthsRatioImg, ratio_vis, f"Cold Months Ratio ({low_temp_threshold} °C)", shown=False) 
    Map.addLayer(wetMonthsRatioImg, ratio_vis, f"Wet Months Ratio ({high_precip_threshold} mm)", shown=False) 

    Map.addLayer(combinedHistoricalScore, ratio_vis, f"Combined Historical Exposure Score ({low_temp_threshold} °C,{high_temp_threshold} °C,{high_precip_threshold} mm)")
    
    # Add color bar
    Map.add_colorbar(ratio_vis, label="Exposure Ratio (0-1)")

    # Add layer control
    #Map.add_child(folium.LayerControl())
    Map.addLayerControl()

    # Render the map in Streamlit
    #st_folium(Map, width=700, height=500)
    Map.to_streamlit(width=700, height=500)



