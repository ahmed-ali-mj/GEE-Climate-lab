# ────────────────────────────────────────────────────────────────────────────
# Page#3 :  Combination of Historical and Forecast Weather Exposure
# ────────────────────────────────────────────────────────────────────────────

import streamlit as st  
from datetime import date, datetime, timedelta, timezone
import ee
import geemap.foliumap as geemap


for k, v in st.session_state.items():
    st.session_state[k] = v
st.header('Page 3: Combination of Historical and Forecast Weather Exposure')

if "point_assets" not in st.session_state:
    st.session_state.point_assets = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point(66.6567, 25.6453), {'name': 'Tower_1'}),
        ee.Feature(ee.Geometry.Point(67.0153, 24.8732), {'name': 'Tower_2'}),
        ee.Feature(ee.Geometry.Point(67.5428, 25.0973), {'name': 'Tower_3'}),
        ee.Feature(ee.Geometry.Point(68.1849, 25.3322), {'name': 'Tower_4'}),
        ee.Feature(ee.Geometry.Point(67.7472, 27.1833), {'name': 'Tower_5'})
        ]);

# Define Weather Exposure Parameters

st.markdown("---")  # Horizontal divider
st.markdown("##### Parameters for Historical Weather Exposure")

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



col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Select maximum acceptable monthly mean temperature (°C)")

with col2:
    st.slider(
        label="",
        min_value=-20,
        max_value=50,
        value=st.session_state.high_temp_threshold_slider_value,
        key='high_temp_threshold_slider',
        step=1
    )
high_temp_threshold = st.session_state.high_temp_threshold_slider_value

if "low_temp_threshold_slider_value" not in st.session_state:    
    st.session_state.low_temp_threshold_slider_value = 10

col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Select minimum acceptable monthly mean temperature (°C)")

with col2:
    st.session_state.low_temp_threshold_slider_value = st.slider(
        label="",
        min_value=-20,
        max_value=50,
        key='low_temp_threshold_slider',
        value=st.session_state.low_temp_threshold_slider_value,
        step=1
    )

low_temp_threshold = st.session_state.low_temp_threshold_slider_value


if "high_precip_threshold_slider_value" not in st.session_state:    
    st.session_state.high_precip_threshold_slider_value = 100

col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Select maximum acceptable monthly rainfall (mm)")

with col2:
    st.session_state.high_precip_threshold_slider_value = st.slider(
        label="",
        min_value=10,
        max_value=250,
        key='high_precip_threshold_slider',
        value=st.session_state.high_precip_threshold_slider_value,
        step=5
    )
high_precip_threshold = st.session_state.high_precip_threshold_slider_value



highTempLimit = ee.Number(high_temp_threshold)
lowTempLimit = ee.Number(low_temp_threshold)
highPrecipLimit = ee.Number(high_precip_threshold)

st.write("Provide the weightage of individual factors to calcuate the combined historical weather exposure score:")
#st.write("(These weights must add up to 1.)")

#weightHistoricalMaxTempViolations = st.number_input("Weightage of max temperature violations:", value=0.34)
#weightHistoricalMinTempViolations = st.number_input("Weightage of min temperature violations:", value=0.33)
#weightHistoricalRainfallViolations = st.number_input("Weightage of rainfall violations:", value=0.33)
if "slider_range_values" not in st.session_state:
    st.session_state.slider_range_values = (0.8,0.9)
st.session_state.slider_range_values = st.slider(
        "Adjust the split ",
        0.0, 1.0, 
        value = st.session_state.slider_range_values, 
        key = 'slider_range',
        step=0.01
    )

weightHistoricalMaxTempViolations = st.session_state.slider_range_values[0]
weightHistoricalMinTempViolations = st.session_state.slider_range_values[1] - st.session_state.slider_range_values[0]
weightHistoricalRainfallViolations = 1.0 - st.session_state.slider_range_values[1]

st.write(f"Weight for max temperature violations = {weightHistoricalMaxTempViolations:.2f}")
st.write(f"Weight for min temperature violations = {weightHistoricalMinTempViolations:.2f}")
st.write(f"Weight for rainfall violations = {weightHistoricalRainfallViolations:.2f}")

st.markdown("---")  # Horizontal divider
st.markdown("##### Parameters for Forecast Weather Exposure")

if "high_forecast_temp_threshold_slider_value" not in st.session_state:    
    st.session_state.high_forecast_temp_threshold_slider_value = 30.0

col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Select maximum acceptable forecast temperature (°C)")

with col2:
    st.session_state.high_forecast_temp_threshold_slider_value = st.slider(
        label="",
        min_value=10.0,
        max_value=50.0,
        value=st.session_state.high_forecast_temp_threshold_slider_value,
        key='high_forecast_temp_threshold',
        step=0.5
    )    
high_forecast_temp_threshold = st.session_state.high_forecast_temp_threshold_slider_value
highForecastTempLimit = ee.Number(high_forecast_temp_threshold)
maxForecastTempPossible = ee.Number(50.0)
highForecastTempRange = maxForecastTempPossible.subtract(highForecastTempLimit)



if "high_forecast_precip_threshold_slider_value" not in st.session_state:    
    st.session_state.high_forecast_precip_threshold_slider_value = 30

col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Select maximum acceptable forecast rainfall (mm)")

with col2:
    st.session_state.high_forecast_precip_threshold_slider_value = st.slider(
        label="",
        min_value=10,
        max_value=200,
        value=st.session_state.high_forecast_precip_threshold_slider_value,
        key='high_forecast_precip_threshold',
        step=5
    ) 

highForecastRainfallLimit = ee.Number(st.session_state.high_forecast_precip_threshold_slider_value)
maxForecastRainfallPossible = ee.Number(200.0)
highForecastRainfallRange = maxForecastRainfallPossible.subtract(highForecastRainfallLimit)

#st.write("Provide the weightage of forecast temperature and rainfall to calculate the combined forecast exposure score:")
#st.write("(These weights must add up to 1.)")

if "forecast_Temperature_Weightage_slider_value" not in st.session_state:    
    st.session_state.forecast_Temperature_Weightage_slider_value = 0.7


col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("")
    st.write("Provide the weightage of forecast temperature to forecast rainfall to calculate the combined forecast exposure score:")

with col2:
    forecast_Temperature_Weightage_slider_value = st.slider(
        label="",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.forecast_Temperature_Weightage_slider_value,
        key='forecast_Temperature_Weightage_slider',
        step=0.01
    )
forecastTemperatureWeightage = forecast_Temperature_Weightage_slider_value   
forecastRainfallWeightage = 1-forecastTemperatureWeightage


st.write(f"Weight for Forecast Temperature Exposure: = {forecastTemperatureWeightage:.2f}")
st.write(f"Weight for Forecast Rainfall Exposure: = {forecastRainfallWeightage:.2f}")


st.markdown("---")  # Horizontal divider
st.markdown("##### Parameters for Combining Historical and Forecast Weather Exposure")

#st.write("Provide the weightage of historical and forecast weather exposure to calculate the overall exposure score:")
#st.write("(These weights must add up to 1.)")

#historicalWeightage = st.number_input("Weightage of Historical Exposure:", value=0.6)
#forecastWeightage = st.number_input("Weightage of Forecast Exposure:", value=0.4)

if "historicalWeightage3_slider_value" not in st.session_state:    
    st.session_state.historicalWeightage3_slider_value = 0.5

col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed

with col1:
    st.write("")
    st.write("Provide the weightage of historical exposure compared to forecast exposure:")

with col2:
    st.session_state.historicalWeightage3_slider_value = st.slider(
        label="",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.historicalWeightage3_slider_value,
        step=0.01,
        key="historical_forecast_slider3"
    )
historicalWeightage = st.session_state.historicalWeightage3_slider_value
forecastWeightage = 1-historicalWeightage

st.write(f"Weight for Historical Exposure: = {historicalWeightage:.2f}")
st.write(f"Weight for Forecast Exposure: = {forecastWeightage:.2f}")




# Process Historical Weather Exposure Analysis

era5MonthlyTemp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate(selected_start, selected_end) \
    .select('temperature_2m');
    
era5MonthlyPrecip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
.filterDate(selected_start, selected_end) \
.select('total_precipitation_sum');

#Clip each image in the collection to the ROI
roi = st.session_state.roi
clippedMonthlyTemp = era5MonthlyTemp.map(lambda img: img.clip(roi))
clippedMonthlyTempCelsius = clippedMonthlyTemp.map(lambda img: img.subtract(273.5))

clippedMonthlyPrecip = era5MonthlyPrecip.map(lambda img: img.clip(roi))
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

#combinedHistoricalScore = hotMonthsRatioImg.add(coldMonthsRatioImg).add(wetMonthsRatioImg).divide(3).rename("avg_historical_score")
combinedHistoricalScore = (hotMonthsRatioImg.multiply(weightHistoricalMaxTempViolations)).add(coldMonthsRatioImg.multiply(weightHistoricalMinTempViolations)).add(wetMonthsRatioImg.multiply(weightHistoricalRainfallViolations))


# Process Forecast Weather Exposure Analysis

ee_now = ee.Date(datetime.now(timezone.utc))    #Gives the curret time
current_time_stamp = ee_now.format("YYYY-MM-dd HH:mm").getInfo()
current_hour = int(ee_now.format("HH").getInfo())
st.write(f"🕒 Current time: {current_time_stamp} UTC")

ee_yesterday = ee_now.advance(-1, 'day')
gfsIC = ee.ImageCollection('NOAA/GFS0P25').filterDate(ee_yesterday, ee_now) 

latestGFSImg = gfsIC.sort("system:time_start", False).first();

latestIndex = ee.String(latestGFSImg.get("system:index"));

#st.write(f"🕒 Latest Index: {latestIndex.getInfo()}")

year = latestIndex.slice(0, 4)
month = latestIndex.slice(4, 6)
day = latestIndex.slice(6, 8)
hour = latestIndex.slice(8, 10)

# Create ISO format string
iso_date = year.cat('-').cat(month).cat('-').cat(day).cat('T').cat(hour).cat(':00:00')

# Create ee.Date object
ee_latest_model_date = ee.Date(iso_date)

st.write(f"🕒 Latest Model Run Time: {ee_latest_model_date.format("YYYY-MM-dd HH:mm").getInfo()} UTC")


forecast_range = 48



run_prefix = latestIndex.slice(0,10)

forecastTempImg = [None] * forecast_range
forecastRainfallImg = [None] * forecast_range
forecastTempRatioImg = [None] * forecast_range
finalForecastTempRatioImg = [None] * forecast_range
forecastRainfallRatioImg = [None] * forecast_range
finalForecastRainfallRatioImg = [None] * forecast_range
combinedForecastScore = [None] * forecast_range
combinedExposureScore = [None] * forecast_range

for x in range(forecast_range):
    #target_index = run_prefix.cat('F072')
    hour = str(x)
    if x<10:
        hour = '0' + hour
    target_index = run_prefix.cat('F0').cat(hour)

    #st.write(f"🕒 Target Index: {target_index.getInfo()}")

    forecastImg = gfsIC.filter(ee.Filter.eq('system:index', target_index)).first();

    clippedForecastImg = forecastImg.clip(roi)
    forecastTempImg[x] = clippedForecastImg.select('temperature_2m_above_ground')
    forecastRainfallImg[x] = clippedForecastImg.select('precipitation_rate')


    # Convert temperature to a ratio (0-1), indicating how high the temperature is from the acceptable limit. 
    forecastTempRatioImg[x] = forecastTempImg[x].subtract(highForecastTempLimit).divide(highForecastTempRange)
    finalForecastTempRatioImg[x] = ee.Image(0).clip(roi).where(forecastTempImg[x].gt(highForecastTempLimit), forecastTempRatioImg[x])

    # Convert rainfall to a ratio (0-1), indicating how high the rainfall is from the acceptable limit. 
    forecastRainfallRatioImg[x] = forecastRainfallImg[x].subtract(highForecastRainfallLimit).divide(highForecastRainfallRange)
    finalForecastRainfallRatioImg[x] = ee.Image(0).clip(roi).where(forecastRainfallImg[x].gt(highForecastRainfallLimit), forecastRainfallRatioImg[x])

    #combinedForecastScore = finalForecastTempRatioImg.add(finalForecastRainfallRatioImg).divide(2).rename("avg_forecast_score")
    combinedForecastScore[x] = (finalForecastTempRatioImg[x].multiply(forecastTemperatureWeightage)).add(finalForecastRainfallRatioImg[x].multiply(forecastRainfallWeightage))
    
    # Process Combined Results of Historical and Forecast Exposure Scores
    #historicalWeightage = 0.9
    #forecastWeightage = 0.1
    combinedExposureScore[x] = (combinedHistoricalScore.multiply(historicalWeightage)).add(combinedForecastScore[x].multiply(forecastWeightage)).rename("avg_combined_score")



#Average out all risk exposures
avgscore = combinedExposureScore[0]
for x in range(1,forecast_range):
    avgscore = avgscore.add(combinedExposureScore[x])
avgscore.divide(forecast_range)


# Visualization parameters
opacity = 0.6

forecast_hour = st.slider(
    "Select forecast hour since the latest model run time (hours)",
    min_value=1,
    max_value=forecast_range,
    value=1,
    step=1
)

temp_vis_params = {
    'min': -10,
    'max': 50,
    'palette': ['blue', 'cyan', 'lime', 'yellow', 'orange', 'red'],
    'opacity': opacity
}

rainfall_vis_params = {
    'min': 0,
    'max': 200,
    'palette': ['blue', 'cyan', 'lime', 'yellow', 'orange', 'red'],
    'opacity': opacity
}

ratio_vis = {
    'min': 0,
    'max': 1,
    'palette': [
        "#1a9850", "#66bd63", "#a6d96a", "#d9ef8b", "#fee08b",
        "#fdae61", "#f46d43", "#d73027", "#a50026"
    ],
    'opacity': opacity
}

# Create map

Map = geemap.Map()
Map.centerObject(roi, 4)

Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Transmission lines');

Map.addLayer(hotMonthsRatioImg, ratio_vis, f"Hot Months Ratio ({high_temp_threshold} °C)", shown=False) 
Map.addLayer(coldMonthsRatioImg, ratio_vis, f"Cold Months Ratio ({low_temp_threshold} °C)", shown=False) 
Map.addLayer(wetMonthsRatioImg, ratio_vis, f"Wet Months Ratio ({high_precip_threshold} mm)", shown=False)   
Map.addLayer(combinedHistoricalScore, ratio_vis, f"Combined Historical Exposure Score ({low_temp_threshold} °C,{high_temp_threshold} °C,{high_precip_threshold} mm)", shown=False)


Map.addLayer(forecastTempImg[forecast_hour], temp_vis_params, f"Temperature (°C) at hour {forecast_hour}", shown=False)
Map.add_colorbar(temp_vis_params, label="°C")
Map.addLayer(forecastRainfallImg[forecast_hour], rainfall_vis_params, f"Rainfall (mm) at hour {forecast_hour}", shown=False)
Map.add_colorbar(rainfall_vis_params, label="mm")
Map.addLayer(finalForecastTempRatioImg[forecast_hour], ratio_vis, f"High Temp Ratio ({high_forecast_temp_threshold} °C)", shown=False) 
Map.addLayer(finalForecastRainfallRatioImg[forecast_hour], ratio_vis, f"High Rainfall Ratio ({st.session_state.high_forecast_precip_threshold_slider_value} mm)", shown=False) 
Map.add_colorbar(ratio_vis, label="Exposure Ratio (0-1)")
Map.addLayer(combinedForecastScore[forecast_hour], ratio_vis, f"Combined Forecast Exposure Score ({high_forecast_temp_threshold} °C,{st.session_state.high_forecast_precip_threshold_slider_value} mm)", shown=False)


st.session_state["exposure_score"] = combinedExposureScore
st.session_state["forecast_range"] = forecast_range

Map.addLayer(combinedExposureScore[forecast_hour], ratio_vis, f"Combined Historical and Forecast Exposure Score ( Historical Weightage: {st.session_state.historicalWeightage3_slider_value}, Forecast Weightage: {forecastWeightage})")

Map.addLayerControl()

# Display map in Streamlit
Map.to_streamlit(width=700, height=500)



