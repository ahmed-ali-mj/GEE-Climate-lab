import pandapower as pp
import streamlit as st
import pandas as pd
import geopandas as gpd
import ee
import geemap.foliumap as geemap
#import geemap
import folium
from streamlit_folium import st_folium
import functions
from datetime import date, datetime, timedelta, timezone
import random
import re
import ast
import nest_asyncio
import ee
from shapely.geometry import LineString
import geemap
import numpy as np
import math
import traceback
import plotly.graph_objects as go
from shapely.geometry import LineString, Point
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["About the App", "Network Initialization", "Historical Weather Exposure", 
         "Combined Historical and Forecast Weather Exposure"]
selection = st.sidebar.radio("Go to", pages)

@st.cache_resource
def initialize_ee():
    ee.Authenticate()
    #ee.Initialize(project='ee-gdss-teacher')
    ee.Initialize(project='ee-ahmedalimj')

#initialze GEE
initialize_ee()


# Shared session state initialization
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "network_data" not in st.session_state:
    st.session_state.network_data = None
if "map_obj" not in st.session_state:
    st.session_state.map_obj = None
if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = None
if "weather_map_obj" not in st.session_state:
    st.session_state.weather_map_obj = None

folium.Map.add_ee_layer = functions.add_ee_layer
# ────────────────────────────────────────────────────────────────────────────
# Page 1 :  Network Initialization
# ────────────────────────────────────────────────────────────────────────────
if selection == "Network Initialization":
    st.header('Network Initialization')
    
    
    #Capstone
    
    
    
    # # File uploader for the Excel file
    # uploaded_file = st.file_uploader("Upload your network Excel file (e.g., Final_IEEE_9Bus_Parameters_only.xlsx)", type=["xlsx"], key="file_uploader")

    st.markdown(
    "Do not have an Excel File in our specified format?  \n"
    "[Download the sample IEEE-9 or 14 bus network parameters]"
    "(https://drive.google.com/drive/folders/1oT10dY6hZiM0q3AYiFzEqe_GQ5vA-eEa?usp=sharing)"
    " from Google Drive.",
    unsafe_allow_html=True,
    )
    
    # File uploader for the Excel file
    uploaded_file = st.file_uploader(
        "Upload your network Excel file",
        type=["xlsx"],
        key="file_uploader",
        help="You can also use the template from Google Drive: "
             "[Sample Excel](https://drive.google.com/drive/folders/1oT10dY6hZiM0q3AYiFzEqe_GQ5vA-eEa?usp=sharing)",
    )
    
    # Check if a new file was uploaded
    if uploaded_file is not None and st.session_state.uploaded_file_key != uploaded_file.name:
        st.session_state.show_results = False
        st.session_state.network_data = None
        st.session_state.map_obj = None
        st.session_state.uploaded_file_key = uploaded_file.name
        st.session_state.uploaded_file = uploaded_file  # Store the file object

    if uploaded_file is not None and not st.session_state.show_results:
        # Create an empty pandapower network
        net = pp.create_empty_network()

        # --- Create Buses ---
        df_bus = pd.read_excel(uploaded_file, sheet_name="Bus Parameters", index_col=0)
        for idx, row in df_bus.iterrows():
            pp.create_bus(net,
                          name=row["name"],
                          vn_kv=row["vn_kv"],
                          zone=row["zone"],
                          in_service=row["in_service"],
                          max_vm_pu=row["max_vm_pu"],
                          min_vm_pu=row["min_vm_pu"])

        # --- Create Loads ---
        df_load = pd.read_excel(uploaded_file, sheet_name="Load Parameters", index_col=0)
        for idx, row in df_load.iterrows():
            pp.create_load(net,
                           bus=row["bus"],
                           p_mw=row["p_mw"],
                           q_mvar=row["q_mvar"],
                           in_service=row["in_service"])

        # --- Create Transformers (if sheet exists) ---
        df_trafo = None
        if "Transformer Parameters" in pd.ExcelFile(uploaded_file).sheet_names:
            df_trafo = pd.read_excel(uploaded_file, sheet_name="Transformer Parameters", index_col=0)
            for idx, row in df_trafo.iterrows():
                pp.create_transformer_from_parameters(net,
                                                      hv_bus=row["hv_bus"],
                                                      lv_bus=row["lv_bus"],
                                                      sn_mva=row["sn_mva"],
                                                      vn_hv_kv=row["vn_hv_kv"],
                                                      vn_lv_kv=row["vn_lv_kv"],
                                                      vk_percent=row["vk_percent"],
                                                      vkr_percent=row["vkr_percent"],
                                                      pfe_kw=row["pfe_kw"],
                                                      i0_percent=row["i0_percent"],
                                                      in_service=row["in_service"],
                                                      max_loading_percent=row["max_loading_percent"])

        # --- Create Generators/External Grid ---
        df_gen = pd.read_excel(uploaded_file, sheet_name="Generator Parameters", index_col=0)
        df_gen['in_service'] = df_gen['in_service'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False}).fillna(True)
        df_gen['controllable'] = df_gen['controllable'].astype(str).str.strip().str.upper().map({'TRUE': True, 'FALSE': False})
        #st.write("Row keys:", row.keys())  # Shows available keys in the row
        #st.write("Row content:", row)      # Displays the full row
        for idx, row in df_gen.iterrows():
            if row["slack_weight"] == 1:
                ext_idx = pp.create_ext_grid(net,
                                             bus=row["bus"],
                                             vm_pu=row["vm_pu"],
                                             va_degree=0)
                pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],   
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])
            else:
                gen_idx = pp.create_gen(net,
                                        bus=row["bus"],
                                        p_mw=row["p_mw"],
                                        vm_pu=row["vm_pu"],
                                        min_q_mvar=row["min_q_mvar"],
                                        max_q_mvar=row["max_q_mvar"],
                                        scaling=row["scaling"],
                                        in_service=row["in_service"],
                                        slack_weight=row["slack_weight"],
                                        controllable=row["controllable"],
                                        max_p_mw=row["max_p_mw"],
                                        min_p_mw=row["min_p_mw"])
                pp.create_poly_cost(net, element=gen_idx, et="gen",
                                    cp0_eur_per_mw=row["cp0_pkr_per_mw"],
                                    cp1_eur_per_mw=row["cp1_pkr_per_mw"],
                                    cp2_eur_per_mw=row["cp2_pkr_per_mw"],
                                    cp0_eur_per_mvar=row["cp0_pkr_per_mvar"],
                                    cq1_eur_per_mvar=row["cq1_pkr_per_mvar"],
                                    cq2_eur_per_mvar=row["cq2_pkr_per_mvar"])

        # --- Create Lines ---
        df_line = pd.read_excel(uploaded_file, sheet_name="Line Parameters", index_col=0)
        for idx, row in df_line.iterrows():
            if isinstance(row["geodata"], str):
                geodata = ast.literal_eval(row["geodata"])
            else:
                geodata = row["geodata"]
            pp.create_line_from_parameters(net,
                                           from_bus=row["from_bus"],
                                           to_bus=row["to_bus"],
                                           length_km=row["length_km"],
                                           r_ohm_per_km=row["r_ohm_per_km"],
                                           x_ohm_per_km=row["x_ohm_per_km"],
                                           c_nf_per_km=row["c_nf_per_km"],
                                           max_i_ka=row["max_i_ka"],
                                           in_service=row["in_service"],
                                           max_loading_percent=row["max_loading_percent"],
                                           geodata=geodata)

        # --- Read Dynamic Profiles ---
        df_load_profile = pd.read_excel(uploaded_file, sheet_name="Load Profile")
        df_load_profile.columns = df_load_profile.columns.str.strip()

        df_gen_profile = pd.read_excel(uploaded_file, sheet_name="Generator Profile")
        df_gen_profile.columns = df_gen_profile.columns.str.strip()

        # --- Build Dictionaries for Dynamic Column Mapping ---
        load_dynamic = {}
        for col in df_load_profile.columns:
            m = re.match(r"p_mw_bus_(\d+)", col)
            if m:
                bus = int(m.group(1))
                q_col = f"q_mvar_bus_{bus}"
                if q_col in df_load_profile.columns:
                    load_dynamic[bus] = {"p": col, "q": q_col}

        gen_dynamic = {}
        for col in df_gen_profile.columns:
            if col.startswith("p_mw"):
                numbers = re.findall(r'\d+', col)
                if numbers:
                    bus = int(numbers[-1])
                    gen_dynamic[bus] = col

        # Store network data in session state
        st.session_state.network_data = {
            'df_bus': df_bus,
            'df_load': df_load,
            'df_gen': df_gen,
            'df_line': df_line,
            'df_load_profile': df_load_profile,
            'df_gen_profile': df_gen_profile,
            # 'df_gen_params':   df_gen_params,      #  ← NEW
            'df_trafo': df_trafo  # Add transformer data to session state
        }

    # --- Button to Display Results ---
    if st.button("Show Excel Network Parameters") and uploaded_file is not None:
        st.session_state.show_results = True
        # Generate map if not already generated
        if st.session_state.map_obj is None and st.session_state.network_data is not None:
            with st.spinner("Generating map..."):
                st.session_state.map_obj = functions.create_map(st.session_state.network_data['df_line'],st.session_state.network_data['df_load'])

    # --- Display Results ---
    if st.session_state.show_results and st.session_state.network_data is not None:
        
        # Display Map
        st.subheader("Transmission Network Map")
        if st.session_state.map_obj is not None:
            st.session_state.map_obj.to_streamlit(width=700, height=500)
        else:
            st.warning("Map could not be generated.")
        
        st.subheader("Network Parameters")

        # Display Bus Parameters
        st.write("### Bus Parameters")
        st.dataframe(st.session_state.network_data['df_bus'])

        # Display Load Parameters
        st.write("### Load Parameters")
        st.dataframe(st.session_state.network_data['df_load'])

        # Display Generator Parameters
        st.write("### Generator Parameters")
        st.dataframe(st.session_state.network_data['df_gen'])

        # Display Transformer Parameters (if exists)
        if st.session_state.network_data['df_trafo'] is not None:
            st.write("### Transformer Parameters")
            st.dataframe(st.session_state.network_data['df_trafo'])

        # Display Line Parameters
        st.write("### Line Parameters")
        st.dataframe(st.session_state.network_data['df_line'])

        # Display Load Profile
        st.write("### Load Profile")
        st.dataframe(st.session_state.network_data['df_load_profile'])

        # Display Generator Profile
        st.write("### Generator Profile")
        st.dataframe(st.session_state.network_data['df_gen_profile'])

        


    if uploaded_file is None and not st.session_state.show_results:
        st.info("Please upload an Excel file to proceed.")



# ────────────────────────────────────────────────────────────────────────────
# Page 2 :  Historical Weather Exposure Analysis
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Historical Weather Exposure":
    st.header('Historical Weather Exposure Analysis')
    
    # Set up the temporary "network of interest"
    karachi = ee.Geometry.Point(67.0011, 24.8607)
    roi = karachi.buffer(600000).bounds()
    if "point_assets" not in st.session_state:
        st.session_state.point_assets = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point(66.6567, 25.6453), {'name': 'Tower_1'}),
            ee.Feature(ee.Geometry.Point(67.0153, 24.8732), {'name': 'Tower_2'}),
            ee.Feature(ee.Geometry.Point(67.5428, 25.0973), {'name': 'Tower_3'}),
            ee.Feature(ee.Geometry.Point(68.1849, 25.3322), {'name': 'Tower_4'}),
            ee.Feature(ee.Geometry.Point(67.7472, 27.1833), {'name': 'Tower_5'})
            ]);
    
    
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
        #Map = geemap.Map(center=[20, 0], zoom=2)
        #Map = geemap.Map(center=karachi, zoom=5)
        Map = st.session_state.map_obj
        #Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Infrastructure Point Assets');
        #Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Infrastructure Line Assets');
        era5Monthly = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
        era5MonthlyTemp = era5Monthly.select('temperature_2m').filterDate(selected_start).first().clip(roi)
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
        clippedMonthlyPrecip = era5MonthlyPrecip.map(lambda img: img.clip(roi))
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
        # Map.add_child(folium.LayerControl())
        Map.addLayerControl()

        # Render the map in Streamlit
        # st_folium(Map, width=700, height=500)
        Map.to_streamlit(width=700, height=500)


    elif st.session_state.historical_exposure_measure == "NumOfExtremeMonthsOverStudyPeriod":

        # Add some space before the map
        #st.markdown("### \n\n")
        st.markdown("---")  # Horizontal divider
        #st.markdown("## Map View")

        high_temp_threshold = st.slider(
            "Select maximum acceptable monthly mean temperature (°C)",
            min_value=-20.0,
            max_value=50.0,
            value=25.0,
            step=0.5
        )

        low_temp_threshold = st.slider(
            "Select minimum acceptable monthly mean temperature (°C)",
            min_value=-20.0,
            max_value=50.0,
            value=10.0,
            step=0.5
        )

        high_precip_threshold = st.slider(
            "Select maximum acceptable monthly rainfall (mm)",
            min_value=10,
            max_value=250,
            value=50,
            step=5
        )

        highTempLimit = ee.Number(high_temp_threshold)
        lowTempLimit = ee.Number(low_temp_threshold)
        highPrecipLimit = ee.Number(high_precip_threshold)

        st.write("Provide the weightage of individual factors to calcuate the combined historical weather exposure score:")
        st.write("(These weights must add up to 1.)")

        weightHistoricalMaxTempViolations = st.number_input("Weight for max temperature violations:", value=0.34)
        weightHistoricalMinTempViolations = st.number_input("Weight for min temperature violations:", value=0.33)
        weightHistoricalRainfallViolations = st.number_input("Weight for rainfall violations:", value=0.33)



        #opacity = st.slider("Select layer opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        #st.write("Opacity: ", opacity)
        opacity = 0.6

        # Create an interactive map
        #Map = geemap.Map(center=[20, 0], zoom=2)
        #Map = geemap.Map(center=karachi, zoom=5)
        Map = geemap.Map()
        Map.centerObject(roi, 5)
        Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Infrastructure Point Assets');
        Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Infrastructure Line Assets');

        era5MonthlyTemp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
            .filterDate(selected_start, selected_end) \
            .select('temperature_2m');
         
        era5MonthlyPrecip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(selected_start, selected_end) \
        .select('total_precipitation_sum');
    
        #Clip each image in the collection to the ROI
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

    
    

   
    
    # # Load ERA5-Land hourly dataset
    # era5Hourly = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
    # .filterDate(selected_start, selected_end) \
    # .select('total_precipitation') \
    # .filterBounds(roi)

    # # Sum hourly precipitation (each image is in meters/hour)
    # totalPrecip = era5Hourly.sum().multiply(1000); # convert from meters to millimeters

    # # Number of full months
    # n_months = selected_end.difference(selected_start, 'month').floor().getInfo()

    # avgMonthlyPrecip = totalPrecip.divide(ee.Number(n_months))
    
    # precip_vis = {
    #     'min': 0,
    #     'max': 320,
    #     'palette': ['green', 'yellow', 'red']
    # }
    
    # Map.addLayer(avgMonthlyPrecip, precip_vis, 'ERA5 Avg Monthly Precipitation')



    



# ────────────────────────────────────────────────────────────────────────────
# Page 3 :  Combination of Historical and Forecast Weather Exposure
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Combined Historical and Forecast Weather Exposure":
    st.header('Combination of Historical and Forecast Weather Exposure')

    # Define region of interest (global extent here)
    #roi = ee.Geometry.Rectangle([-180, -90, 180, 90])
    karachi = ee.Geometry.Point(67.0011, 24.8607)
    roi = karachi.buffer(600000).bounds()
    #roi = ee.Geometry.BBox(60, 23, 78, 38);  # e.g., Pakistan
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

    high_temp_threshold = st.slider(
        "Select maximum acceptable monthly mean temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=25.0,
        step=0.5
    )

    low_temp_threshold = st.slider(
        "Select minimum acceptable monthly mean temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )

    high_precip_threshold = st.slider(
        "Select maximum acceptable monthly rainfall (mm)",
        min_value=10,
        max_value=250,
        value=50,
        step=5
    )

    highTempLimit = ee.Number(high_temp_threshold)
    lowTempLimit = ee.Number(low_temp_threshold)
    highPrecipLimit = ee.Number(high_precip_threshold)

    st.write("Provide the weightage of individual factors to calcuate the combined historical weather exposure score:")
    st.write("(These weights must add up to 1.)")

    weightHistoricalMaxTempViolations = st.number_input("Weightage of max temperature violations:", value=0.34)
    weightHistoricalMinTempViolations = st.number_input("Weightage of min temperature violations:", value=0.33)
    weightHistoricalRainfallViolations = st.number_input("Weightage of rainfall violations:", value=0.33)
    
    st.markdown("---")  # Horizontal divider
    st.markdown("##### Parameters for Forecast Weather Exposure")

    high_forecast_temp_threshold = st.slider(
        "Select maximum acceptable forecast temperature (°C)",
        min_value=10.0,
        max_value=50.0,
        value=30.0,
        step=0.5
    )

    highForecastTempLimit = ee.Number(high_forecast_temp_threshold)
    maxForecastTempPossible = ee.Number(50.0)
    highForecastTempRange = maxForecastTempPossible.subtract(highForecastTempLimit)

    high_forecast_precip_threshold = st.slider(
        "Select maximum acceptable forecast rainfall (mm)",
        min_value=10,
        max_value=200,
        value=20,
        step=5
    )

    highForecastRainfallLimit = ee.Number(high_forecast_precip_threshold)
    maxForecastRainfallPossible = ee.Number(200.0)
    highForecastRainfallRange = maxForecastRainfallPossible.subtract(highForecastRainfallLimit)
    
    st.write("Provide the weightage of forecast temperature and rainfall to calculate the combined forecast exposure score:")
    st.write("(These weights must add up to 1.)")

    forecastTemperatureWeightage = st.number_input("Weightage of Forecast Temperature Exposure:", value=0.5)
    forecastRainfallWeightage = st.number_input("Weightage of Forecast Rainfall Exposure:", value=0.5)


    st.markdown("---")  # Horizontal divider
    st.markdown("##### Parameters for Combining Historical and Forecast Weather Exposure")

    st.write("Provide the weightage of historical and forecast weather exposure to calculate the overall exposure score:")
    st.write("(These weights must add up to 1.)")

    historicalWeightage = st.number_input("Weightage of Historical Exposure:", value=0.6)
    forecastWeightage = st.number_input("Weightage of Forecast Exposure:", value=0.4)

    # Process Historical Weather Exposure Analysis

    era5MonthlyTemp = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(selected_start, selected_end) \
        .select('temperature_2m');
        
    era5MonthlyPrecip = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
    .filterDate(selected_start, selected_end) \
    .select('total_precipitation_sum');

    #Clip each image in the collection to the ROI
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
    #print(current_time_stamp)
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

    
    forecast_hour = st.slider(
        "Select forecast hour since the latest model run time (hours)",
        min_value=0,
        max_value=96,
        value=24,
        step=1
    )

    run_prefix = latestIndex.slice(0,10)
    #target_index = run_prefix.cat('F072')
    target_index = run_prefix.cat('F0').cat(str(forecast_hour))

    #st.write(f"🕒 Target Index: {target_index.getInfo()}")

    forecastImg = gfsIC.filter(ee.Filter.eq('system:index', target_index)).first();

    

    clippedForecastImg = forecastImg.clip(roi)
    
    forecastTempImg = clippedForecastImg.select('temperature_2m_above_ground')
    forecastRainfallImg = clippedForecastImg.select('total_precipitation_surface')


    # Convert temperature to a ratio (0-1), indicating how high the temperature is from the acceptable limit. 
    forecastTempRatioImg = forecastTempImg.subtract(highForecastTempLimit).divide(highForecastTempRange)
    finalForecastTempRatioImg = ee.Image(0).clip(roi).where(forecastTempImg.gt(highForecastTempLimit), forecastTempRatioImg)

    # Convert rainfall to a ratio (0-1), indicating how high the rainfall is from the acceptable limit. 
    forecastRainfallRatioImg = forecastRainfallImg.subtract(highForecastRainfallLimit).divide(highForecastRainfallRange)
    finalForecastRainfallRatioImg = ee.Image(0).clip(roi).where(forecastRainfallImg.gt(highForecastRainfallLimit), forecastRainfallRatioImg)

    #combinedForecastScore = finalForecastTempRatioImg.add(finalForecastRainfallRatioImg).divide(2).rename("avg_forecast_score")
    combinedForecastScore = (finalForecastTempRatioImg.multiply(forecastTemperatureWeightage)).add(finalForecastRainfallRatioImg.multiply(forecastRainfallWeightage))
    
    # Process Combined Results of Historical and Forecast Exposure Scores
    #historicalWeightage = 0.9
    #forecastWeightage = 0.1
    combinedExposureScore = (combinedHistoricalScore.multiply(historicalWeightage)).add(combinedForecastScore.multiply(forecastWeightage)).rename("avg_combined_score")

    # Visualization parameters
    opacity = 0.6

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

    Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Infrastructure Point Assets');
    Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Infrastructure Line Assets');

    Map.addLayer(hotMonthsRatioImg, ratio_vis, f"Hot Months Ratio ({high_temp_threshold} °C)", shown=False) 
    Map.addLayer(coldMonthsRatioImg, ratio_vis, f"Cold Months Ratio ({low_temp_threshold} °C)", shown=False) 
    Map.addLayer(wetMonthsRatioImg, ratio_vis, f"Wet Months Ratio ({high_precip_threshold} mm)", shown=False)   
    Map.addLayer(combinedHistoricalScore, ratio_vis, f"Combined Historical Exposure Score ({low_temp_threshold} °C,{high_temp_threshold} °C,{high_precip_threshold} mm)", shown=False)

    Map.addLayer(forecastTempImg, temp_vis_params, f"Temperature (°C) at hour {forecast_hour}", shown=False)
    Map.add_colorbar(temp_vis_params, label="°C")
    Map.addLayer(forecastRainfallImg, rainfall_vis_params, f"Rainfall (mm) at hour {forecast_hour}", shown=False)
    Map.add_colorbar(rainfall_vis_params, label="mm")
    Map.addLayer(finalForecastTempRatioImg, ratio_vis, f"High Temp Ratio ({high_forecast_temp_threshold} °C)", shown=False) 
    Map.addLayer(finalForecastRainfallRatioImg, ratio_vis, f"High Rainfall Ratio ({high_forecast_precip_threshold} mm)", shown=False) 
    Map.add_colorbar(ratio_vis, label="Exposure Ratio (0-1)")
    Map.addLayer(combinedForecastScore, ratio_vis, f"Combined Forecast Exposure Score ({high_forecast_temp_threshold} °C,{high_forecast_precip_threshold} mm)", shown=False)

    Map.addLayer(combinedExposureScore, ratio_vis, f"Combined Historical and Forecast Exposure Score ( Historical Weightage: {historicalWeightage}, Forecast Weightage: {forecastWeightage})")

    Map.addLayerControl()

    # Display map in Streamlit
    Map.to_streamlit(width=700, height=500)


    

    

# ────────────────────────────────────────────────────────────────────────────
# Page 4 :  Experimental Page 1
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Experimental Page 1":
    st.header('Experimental Page 1')

    # # Initialize the state variable only once
    # if "selected_date" not in st.session_state:
    #     st.session_state.selected_date = date.today()

    # st.title("📅 Date Slider with Session State")

    # # Create the slider using session_state value as the default
    # selected = st.slider(
    #     "Select a date",
    #     min_value=date(2023, 1, 1),
    #     max_value=date.today(),
    #     value=st.session_state.selected_date,
    #     format="YYYY-MM-DD"
    # )

    # # Update session state when user interacts
    # st.session_state.selected_date = selected

    # # Display the value
    # st.success(f"Selected Date: {st.session_state.selected_date}")

    

    # # Define ROI
    # karachi = ee.Geometry.Point([67.0011, 24.8607])
    # roi = karachi.buffer(300000).bounds()

    # # Initialize session state for animation
    # if "month_index" not in st.session_state:
    #     st.session_state.month_index = 0

    # # Create list of monthly timestamps (2023 only)
    # months = [datetime(2023, m, 1) for m in range(1, 13)]
    # month_labels = [d.strftime('%B %Y') for d in months]

    # # Controls
    # st.title("📈 ERA5 Monthly Temperature Time Series")

    # col1, col2 = st.columns([1, 1])
    # if col1.button("⏮️ Previous Month"):
    #     st.session_state.month_index = (st.session_state.month_index - 1) % len(months)
    # if col2.button("⏭️ Next Month"):
    #     st.session_state.month_index = (st.session_state.month_index + 1) % len(months)

    # selected_month = months[st.session_state.month_index]
    # st.markdown(f"### Showing data for: **{selected_month.strftime('%B %Y')}**")

    # # Fetch temperature image for the selected month
    # era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY") \
    #     .select("temperature_2m") \
    #     .filterDate(selected_month.strftime('%Y-%m-%d'),
    #                 (selected_month + timedelta(days=31)).strftime('%Y-%m-%d')) \
    #     .first() \
    #     .clip(roi)

    # # Visualize
    # vis = {
    #     'min': 270,
    #     'max': 310,
    #     'palette': ['blue', 'green', 'yellow', 'orange', 'red']
    # }

    # Map = geemap.Map()
    # Map.centerObject(roi, 5)
    # Map.addLayer(era5, vis, f"Temperature - {selected_month.strftime('%b %Y')}")
    # Map.add_child(geemap.folium.LayerControl())

    # # Display the map
    # st_folium(Map, height=500)

    

    # Define region of interest
    karachi = ee.Geometry.Point([67.0011, 24.8607])
    roi = karachi.buffer(300000).bounds()

    # Set up date for demo (you could expand this to a slider too)
    selected_date = datetime(2023, 6, 1)

    # Title
    st.title("🌦️ ERA5 Toggle Viewer")

    # Initialize UI state
    if "layer" not in st.session_state:
        st.session_state.layer = "Temperature"

    # UI toggle: Temperature or Precipitation
    st.session_state.layer = st.radio(
        "Choose variable to visualize:",
        ["Temperature", "Precipitation"],
        index=["Temperature", "Precipitation"].index(st.session_state.layer)
    )

    # Load selected variable from ERA5
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY")
    start = selected_date.strftime('%Y-%m-%d')
    end = (selected_date + timedelta(days=31)).strftime('%Y-%m-%d')

    if st.session_state.layer == "Temperature":
        image = era5.select("temperature_2m").filterDate(start, end).first().clip(roi)
        vis = {"min": 270, "max": 310, "palette": ["blue", "green", "red"]}
        label = "Monthly Temperature (K)"
    else:
        image = era5.select("total_precipitation_sum").filterDate(start, end).first().clip(roi)
        vis = {"min": 0, "max": 0.5, "palette": ["white", "cyan", "blue"]}
        label = "Monthly Precipitation (m)"

    # Show map
    Map = geemap.Map()
    Map.centerObject(roi, 5)
    Map.addLayer(image, vis, label)
    Map.add_child(geemap.folium.LayerControl())
    st.markdown(f"### {label} — {selected_date.strftime('%B %Y')}")
    st_folium(Map, height=500)


# ────────────────────────────────────────────────────────────────────────────
# Page 5 :  Experimental Page 2
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Experimental Page 2":
    st.header('Experimental Page 2')
    
    # Define region of interest (Karachi buffer)
    karachi = ee.Geometry.Point([67.0011, 24.8607])
    roi = karachi.buffer(300000).bounds()

    # Generate list of months (2023 only)
    months = [datetime(2023, m, 1) for m in range(1, 13)]
    month_labels = [dt.strftime('%B') for dt in months]

    # Initialize session state
    if "month_index" not in st.session_state:
        st.session_state.month_index = 0

    st.title("📅 Monthly Temperature Animation (ERA5)")

    # Slider widget to select month
    st.session_state.month_index = st.slider(
        "Select Month",
        min_value=0,
        max_value=len(months) - 1,
        value=st.session_state.month_index,
        format="%d",  # Just shows the number, we label it manually below
    )

    selected_month = months[st.session_state.month_index]
    st.markdown(f"### Showing data for: **{selected_month.strftime('%B %Y')}**")

    # Load ERA5 data
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .select("temperature_2m") \
        .filterDate(
            selected_month.strftime('%Y-%m-%d'),
            (selected_month + timedelta(days=31)).strftime('%Y-%m-%d')
        ) \
        .first() \
        .clip(roi)

    # Visualization parameters
    vis_params = {
        'min': 270,
        'max': 310,
        'palette': ['blue', 'green', 'yellow', 'orange', 'red']
    }

    # Create and display the map
    Map = geemap.Map()
    Map.centerObject(roi, 5)
    Map.addLayer(era5, vis_params, f"Temperature - {selected_month.strftime('%b %Y')}")
    Map.add_child(geemap.folium.LayerControl())
    st_folium(Map, height=500)


# ────────────────────────────────────────────────────────────────────────────
# Page 6 :  Experimental Page 3
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Experimental Page 3":
    st.header('Experimental Page 3')

    ee_now = ee.Date(datetime.now(timezone.utc))    #Gives the curret time
    current_time_stamp = ee_now.format("YYYY-MM-dd HH:mm").getInfo()
    print(current_time_stamp)
    st.write(f"🕒 Current time: {current_time_stamp} UTC")
    #st.write(str(now.get('day')))
    ee_yesterday = ee_now.advance(-1, 'day')
    gfsIC = ee.ImageCollection('NOAA/GFS0P25').filterDate(ee_yesterday, ee_now) 

    latestGFSImg = gfsIC.sort("system:time_start", False).first();

    latestIndex = ee.String(latestGFSImg.get("system:index"));
    
    st.write(f"🕒 Latest Index: {latestIndex.getInfo()}")

    year = latestIndex.slice(0, 4)
    month = latestIndex.slice(4, 6)
    day = latestIndex.slice(6, 8)
    hour = latestIndex.slice(8, 10)

    # Create ISO format string
    iso_date = year.cat('-').cat(month).cat('-').cat(day).cat('T').cat(hour).cat(':00:00')
    
    # Create ee.Date object
    ee_latest_model_date = ee.Date(iso_date)

    st.write(f"🕒 Latest Model Run Time: {ee_latest_model_date.format("YYYY-MM-dd HH:mm").getInfo()} UTC")

    


    


    opacity = 0.6

    forecast_hour = st.slider(
        "Select forecast hour since the latest model run time (hours)",
        min_value=0,
        max_value=96,
        value=24,
        step=1
    )

    run_prefix = latestIndex.slice(0,10)
    #target_index = run_prefix.cat('F072')
    target_index = run_prefix.cat('F0').cat(str(forecast_hour))

    st.write(f"🕒 Target Index: {target_index.getInfo()}")

    forecastImg = gfsIC.filter(ee.Filter.eq('system:index', target_index)).first();

    roi = ee.Geometry.BBox(60, 23, 78, 38);  # e.g., Pakistan

    clippedImg = forecastImg.clip(roi)
    tempImg = clippedImg.select('temperature_2m_above_ground')

    rainfallImg = clippedImg.select('total_precipitation_surface')

    high_temp_threshold = st.slider(
        "Select maximum acceptable temperature (°C)",
        min_value=10.0,
        max_value=50.0,
        value=30.0,
        step=0.5
    )

    highTempLimit = ee.Number(high_temp_threshold)
    maxTempPossible = ee.Number(50.0)
    highTempRange = maxTempPossible.subtract(highTempLimit)

    high_precip_threshold = st.slider(
        "Select maximum acceptable rainfall (mm)",
        min_value=10,
        max_value=200,
        value=20,
        step=5
    )

    highRainfallLimit = ee.Number(high_precip_threshold)
    maxRainfallPossible = ee.Number(200.0)
    highRainfallRange = maxRainfallPossible.subtract(highRainfallLimit)

    # Convert temperature to a ratio (0-1), indicating how high the temperature is from the acceptable limit. 
    tempRatioImg = tempImg.subtract(highTempLimit).divide(highTempRange)
    finalTempRatioImg = ee.Image(0).clip(roi).where(tempImg.gt(highTempLimit), tempRatioImg)

    rainfallRatioImg = rainfallImg.subtract(highRainfallLimit).divide(highRainfallRange)
    finalRainfallRatioImg = ee.Image(0).clip(roi).where(rainfallImg.gt(highRainfallLimit), rainfallRatioImg)

    avg_forecast_score = finalTempRatioImg.add(finalRainfallRatioImg).divide(2).rename("avg_forecast_score")

    # Visualization parameters
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
    Map.centerObject(roi, 2)
    Map.addLayer(tempImg, temp_vis_params, f"Temperature (°C) at hour {forecast_hour}")
    Map.add_colorbar(temp_vis_params, label="°C")
    Map.addLayer(rainfallImg, rainfall_vis_params, f"Rainfall (mm) at hour {forecast_hour}")
    Map.add_colorbar(rainfall_vis_params, label="mm")
    Map.addLayer(finalTempRatioImg, ratio_vis, f"High Temp Ratio ({high_temp_threshold} °C)") 
    Map.addLayer(finalRainfallRatioImg, ratio_vis, f"High Rainfall Ratio ({high_precip_threshold} mm)") 
    Map.add_colorbar(ratio_vis, label="Exposure Ratio (0-1)")
    Map.addLayer(avg_forecast_score, ratio_vis, f"Combined Forecast Exposure Score ({high_temp_threshold} °C,{high_precip_threshold} mm)")

    Map.addLayerControl()

    # Display map in Streamlit
    Map.to_streamlit(width=700, height=500)


# ────────────────────────────────────────────────────────────────────────────
# Page 0 :  About the App
# ────────────────────────────────────────────────────────────────────────────
elif selection == "About the App":
    st.header('GEE-GridRiskLab: Google Earth Engine based Tool for Continuous Monitoring of Climate Risks to Electricity Grid')


# # Initialize Earth Engine
# ee.Authenticate()
# ee.Initialize(project='ee-gdss-teacher')

# # Streamlit app title
# st.title("🌍 Google Earth Engine + Streamlit Demo")

# # Create an interactive map
# Map = geemap.Map(center=[20, 0], zoom=2)

# # Load a GEE image (e.g., MODIS NDVI)
# ndvi = ee.ImageCollection("MODIS/006/MOD13A2").select("NDVI").mean()

# # Visualization parameters
# ndvi_vis = {
#     'min': 0,
#     'max': 9000,
#     'palette': ['white', 'green']
# }

# # Add the NDVI layer
# Map.addLayer(ndvi, ndvi_vis, "Mean NDVI")

# # Add layer control
# Map.add_child(folium.LayerControl())



# # Render the map in Streamlit
# st_folium(Map, width=700, height=500)
