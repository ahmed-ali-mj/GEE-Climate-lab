import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import pandapower as pp
import re
import ast
import nest_asyncio
import ee
from streamlit_folium import st_folium
from shapely.geometry import LineString
from datetime import date, datetime, timedelta, timezone
import random
import geemap.foliumap as geemap
import numpy as np
import math
import traceback
import plotly.graph_objects as go
from shapely.geometry import LineString, Point
import plotly.express as px
import functions


# Set page configuration
st.set_page_config(
    page_title="Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
pages = ["About the App", "Network Initialization", "Historical Weather Exposure", 
         "Combined Historical and Forecast Weather Exposure", "Optimal power flow", "Weather aware optimal power flow"]
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

if "high_temp_threshold_slider_value" not in st.session_state:
        st.session_state.high_temp_threshold_slider_value = 25

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
                #st.session_state.map_obj = functions.create_map(st.session_state.network_data['df_line'],st.session_state.network_data['df_load'])
                pass

    # --- Display Results ---
    if st.session_state.show_results and st.session_state.network_data is not None:
        
        # Display Map
        st.subheader("Transmission Network Map")
        st.session_state.map_obj = 1
        if st.session_state.map_obj is not None:
            df_line = st.session_state.network_data['df_line']
            df_load = st.session_state.network_data['df_load']
            #st.session_state.map_obj.to_streamlit(width=700, height=500)
            point_features = [functions.parse_point(row) for _, row in df_load.iterrows()]
            point_features = [f for f in point_features if f is not None]
            point_fc = ee.FeatureCollection(point_features)


            st.session_state.point_assets = point_fc


            # Create base map
            avg_lat, avg_lon = functions.compute_average_lat_lon(df_load)
            
            
            #m = folium.Map(location=[avg_lat, avg_lon], zoom_start=5, width=700, height=500)


            # === Parse and create EE FeatureCollection for transmission lines ===
            df_line["geodata"] = df_line["geodata"].apply(
                lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
            )
            line_features = [
                ee.Feature(ee.Geometry.LineString(row["geodata"]))
                for _, row in df_line.iterrows()
            ]
            line_fc = ee.FeatureCollection(line_features)


            st.session_state.line_assets = line_fc
            
            st.session_state.center_point = ee.Geometry.Point(avg_lon, avg_lat)
            
            st.session_state.center_point = st.session_state.center_point
            
            roi = st.session_state.center_point.buffer(600000).bounds()
        
            st.session_state.roi = roi
            Map = geemap.Map()
            Map.centerObject(st.session_state.center_point, 5)
            Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
            Map.addLayer(st.session_state.line_assets, {'color': 'black'}, 'Transmission lines');

            # Add layer control
            Map.addLayerControl()

            # Render the map in Streamlit
            Map.to_streamlit(width=700, height=500)
            
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
    
    for k, v in st.session_state.items():
        st.session_state[k] = v
        
    st.header('Historical Weather Exposure Analysis')
    
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




# ────────────────────────────────────────────────────────────────────────────
# Page 3 :  Combination of Historical and Forecast Weather Exposure
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Combined Historical and Forecast Weather Exposure":
    for k, v in st.session_state.items():
        st.session_state[k] = v
    st.header('Combination of Historical and Forecast Weather Exposure')
    
    # Define region of interest (global extent here)
    #roi = ee.Geometry.Rectangle([-180, -90, 180, 90])
    #karachi = ee.Geometry.Point(67.0011, 24.8607)
    #roi = karachi.buffer(600000).bounds()
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
    
    if "historicalWeightage_slider_value" not in st.session_state:    
        st.session_state.historicalWeightage_slider_value = 0.7
    
    col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed
    
    with col1:
        st.write("")
        st.write("Provide the weightage of historical exposure compared to forecast exposure:")

    with col2:
        st.session_state.historicalWeightage = st.slider(
            label="",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.historicalWeightage_slider_value,
            step=0.01,
            key="historical_slider"
        )
    historicalWeightage = st.session_state.historicalWeightage_slider_value
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

    Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Loads');
    Map.addLayer(st.session_state.line_assets, {'color': 'red'}, 'Transmission lines');

    Map.addLayer(hotMonthsRatioImg, ratio_vis, f"Hot Months Ratio ({high_temp_threshold} °C)", shown=False) 
    Map.addLayer(coldMonthsRatioImg, ratio_vis, f"Cold Months Ratio ({low_temp_threshold} °C)", shown=False) 
    Map.addLayer(wetMonthsRatioImg, ratio_vis, f"Wet Months Ratio ({high_precip_threshold} mm)", shown=False)   
    Map.addLayer(combinedHistoricalScore, ratio_vis, f"Combined Historical Exposure Score ({low_temp_threshold} °C,{high_temp_threshold} °C,{high_precip_threshold} mm)", shown=False)

    Map.addLayer(forecastTempImg, temp_vis_params, f"Temperature (°C) at hour {forecast_hour}", shown=False)
    Map.add_colorbar(temp_vis_params, label="°C")
    Map.addLayer(forecastRainfallImg, rainfall_vis_params, f"Rainfall (mm) at hour {forecast_hour}", shown=False)
    Map.add_colorbar(rainfall_vis_params, label="mm")
    Map.addLayer(finalForecastTempRatioImg, ratio_vis, f"High Temp Ratio ({high_forecast_temp_threshold} °C)", shown=False) 
    Map.addLayer(finalForecastRainfallRatioImg, ratio_vis, f"High Rainfall Ratio ({st.session_state.high_forecast_precip_threshold_slider_value} mm)", shown=False) 
    Map.add_colorbar(ratio_vis, label="Exposure Ratio (0-1)")
    Map.addLayer(combinedForecastScore, ratio_vis, f"Combined Forecast Exposure Score ({high_forecast_temp_threshold} °C,{st.session_state.high_forecast_precip_threshold_slider_value} mm)", shown=False)

    st.session_state["exposure_score"] = combinedExposureScore
    Map.addLayer(combinedExposureScore, ratio_vis, f"Combined Historical and Forecast Exposure Score ( Historical Weightage: {st.session_state.historicalWeightage_slider_value}, Forecast Weightage: {forecastWeightage})")

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
# Page 4 :  Optimal power flow
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Optimal power flow":

    st.header('Projected Operation - Under Current OPF')
    
    for k, v in st.session_state.items():
        st.session_state[k] = v
    # ── PERSISTENT STORAGE (add right after st.title(...)) ────────────────────
    
    
    if "risk_threshold_slider_value" not in st.session_state:    
        st.session_state.risk_threshold_slider_value = 0.7
    
    col1, col2 = st.columns([4, 4])  # Adjust width ratio as needed
    
    with col1:
        st.write("")
        st.write("Provide the risk score above which transmission line fails:")

    with col2:
        st.session_state.risk_threshold_slider_value = st.slider(
            label="",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_threshold_slider_value,
            step=0.01,
            key="risk_threshold_slider"
        )
    risk_threshold = st.session_state.risk_threshold_slider_value
    
    
    for k, v in (
        ("bau_ready",                          False),
        ("bau_day_end_df",                     None),
        ("bau_hourly_cost_df",                 None),
        ("bau_results",                        None),
        ("line_outages",                       None),
        ("line_idx_map",                       None),
        ("trafo_idx_map",                      None),
        ("max_loading_capacity",               None),
        ("max_loading_capacity_transformer",   None),
        ("bau_hour",                           0),        # hour to draw
    ):
        st.session_state.setdefault(k, v)
    # ──────────────────────────────────────────────────────────────────────────

    
    mode = st.selectbox("Select Contingency Mode",
                    ["Capped Contingency Mode (20% of Network Lines)",
                     "Maximum Contingency Mode (All Outage Lines)"])
    cap_flag = (mode == "Capped Contingency Mode (20% of Network Lines)")
    
    if st.button("Run Current Optimal Power Flow (OPF) Analysis"):
        weather_map, risk_df, line_outage_data, outage_data, max_occurrence_t, max_occurrence_p, max_occurrence_w, risk_scores = \
        functions.process_temperature(
                    "High",
                    "Monthly",
                    risk_threshold,
                    st.session_state.network_data['df_line'],
                    st.session_state["exposure_score"]
                    )
        # Store the map and data in session state
        st.session_state.weather_map_obj = weather_map
        st.session_state.line_outage_data = line_outage_data
        st.session_state["outage_hours"] = line_outage_data["hours"]
        st.session_state["line_down"]    = line_outage_data["lines"]
        st.session_state["risk_scores"]  = line_outage_data["risk_scores"]
        st.session_state.risk_df = risk_df
        st.session_state.outage_data = outage_data
        st.session_state.risk_score = risk_threshold
        st.session_state.max_occurrences = {
            "temperature": max_occurrence_t,
            "precipitation": max_occurrence_p,
            "wind": max_occurrence_w
        }

        # build the outage list first
        line_outages = functions.generate_line_outages(
            outage_hours   = st.session_state["outage_hours"],
            line_down      = st.session_state["line_down"],
            risk_scores    = st.session_state["risk_scores"],
            capped_contingency_mode = cap_flag
        )
        st.session_state.line_outages = line_outages
        # store globally for helper functions
        globals()["line_outages"] = line_outages
        
        with st.spinner("Running Current Optimal Power Flow (OPF) Analysis (Estimated Time 5-10 minutes)..."):
            (_lp_bau, _served, _gen, _slack, _rec, _cost,
             _shed, _seen, _shed_buses, _df_lines, _df_trafo,
             _load_df, _line_idx_map, _trafo_idx_map, _gdf,
             day_end_df, hourly_cost_df) = functions.current_opf(line_outages)

        # -----------------------------------------------------------------
        # CACHE RESULTS so they persist across page switches
        # -----------------------------------------------------------------
        # 2-C · WRITE ALL RESULTS TO SESSION STATE  ←── only here!
        st.session_state.update({
            "bau_ready":                        True,
            "bau_day_end_df":                   day_end_df,
            "bau_hourly_cost_df":               hourly_cost_df,
            "bau_results": {
                "loading_percent_bau": _lp_bau,
                "shedding_buses":      _shed_buses,
            },
            "hourly_shed_bau":     _shed,   
            "served_load_per_hour_bau": _served,
            "gen_per_hour_bau":    _gen,        
            "slack_per_hour_bau":        _slack,   
            "line_idx_map":                     _line_idx_map,
            "trafo_idx_map":                    _trafo_idx_map,
            "max_loading_capacity":             _df_lines["max_loading_percent"].max(),
        })
        # if _df_trafo is not None and not _df_trafo.empty:
        #     st.session_state.max_loading_capacity_transformer = (
        #         _df_trafo["max_loading_percent"].max()
        #     )

        if isinstance(_df_trafo, pd.DataFrame) and not _df_trafo.empty:
            st.session_state.max_loading_capacity_transformer = (
                _df_trafo["max_loading_percent"].max()
            )
        else:      # no transformers → fall back to the line limit
            st.session_state.max_loading_capacity_transformer = (
                st.session_state.max_loading_capacity
            )

    
        # -----------------------------------------------------------------
    
        # st.subheader("Day-End Summary")
        # st.dataframe(day_end_df, use_container_width=True)
    
        # st.subheader("Hourly Generation Cost")
        # st.dataframe(hourly_cost_df, use_container_width=True)

        # ────────────────────────────────────────────────────────────────
        # Show cached tables even after you left the page
        # ────────────────────────────────────────────────────────────────
      # ░░ 3 · ALWAYS-VISIBLE OUTPUT (tables + map) ░░
    if st.session_state.bau_ready:

        # 3-A · Summary tables
        st.subheader("Day Ahead Summary Under Current OPF")
        st.dataframe(st.session_state.bau_day_end_df, use_container_width=True)

        #st.subheader("Hourly Generation Cost Under Current OPF")
        #st.dataframe(st.session_state.bau_hourly_cost_df, use_container_width=True)

        # 3-B · Hour picker  – value is *index*, label is pretty text
        num_hours = len(st.session_state.network_data['df_load_profile'])
        
        # --- make sure the value is an int (first run after page reload) ------------
        if isinstance(st.session_state.bau_hour, str):
            try:
                st.session_state.bau_hour = int(st.session_state.bau_hour.split()[-1])
            except Exception:
                st.session_state.bau_hour = 0
        # ----------------------------------------------------------------------------
        
        st.selectbox(
            "Select Hour to Visualize",
            options=list(range(num_hours)),          # real values (ints)
            format_func=lambda i: f"Hour {i}",       # pretty label
            key="bau_hour",                          # stored as int
            help="Choose any hour; the map refreshes automatically.",
        )

        # 3-C · Build the map for that hour
        hr           = st.session_state.bau_hour
        df_line      = st.session_state.network_data['df_line'].copy()
        df_load      = st.session_state.network_data['df_load'].copy()
        df_trafo     = st.session_state.network_data.get('df_trafo')
        loading_rec  = st.session_state.bau_results['loading_percent_bau'][hr]
        shed_buses   = st.session_state.bau_results['shedding_buses']
        line_idx_map = st.session_state.line_idx_map
        trafo_idx_map= st.session_state.trafo_idx_map
        outages      = st.session_state.line_outages

        # ── helper colour fns (same logic) ─────────────────────
        def get_color(pct, max_cap):
            if pct is None:                return '#FF0000'
            if pct == 0:                   return '#000000'
            if pct <= 0.75*max_cap:        return '#00FF00'
            if pct <= 0.90*max_cap:        return '#FFFF00'
            if pct <  max_cap:             return '#FFA500'
            return '#FF0000'
        get_color_trafo = get_color

        # distinguish line vs trafo
        def check_bus_pair_df(df_line, df_trafo, pair):
            fbus, tbus = pair
            if df_trafo is not None:
                if (((df_trafo["hv_bus"] == fbus) & (df_trafo["lv_bus"] == tbus)) |
                    ((df_trafo["hv_bus"] == tbus) & (df_trafo["lv_bus"] == fbus))).any():
                    return True
            if (((df_line["from_bus"] == fbus) & (df_line["to_bus"] == tbus)) |
                ((df_line["from_bus"] == tbus) & (df_line["to_bus"] == fbus))).any():
                return False
            return None

        # GeoDataFrame for lines
        df_line["geodata"] = df_line["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        gdf = gpd.GeoDataFrame(
            df_line,
            geometry=[LineString(c) for c in df_line["geodata"]],
            crs="EPSG:4326",
        )
        gdf["idx"]     = gdf.index
        gdf["loading"] = gdf["idx"].map(lambda i: loading_rec[i] if i < len(loading_rec) else 0.0)
        
        
        
        
        
        
        
        #Here
        
        
        
        
        
        
        
        # mark weather-down equipment
        weather_down = set()
        for fbus, tbus, start_hr in outages:
            if hr >= start_hr:
                is_tf = check_bus_pair_df(df_line, df_trafo, (fbus, tbus))
                if is_tf:
                    idx = trafo_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx + len(df_line))
                else:
                    idx = line_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx)
        gdf["down_weather"] = gdf["idx"].isin(weather_down)

        # Folium map
        coords = st.session_state.center_point.coordinates().getInfo()
        m = folium.Map(location=[coords[1], coords[0]], zoom_start=6, width=800, height=600)
        max_line_cap = st.session_state.max_loading_capacity
        max_trf_cap  = st.session_state.get("max_loading_capacity_transformer", max_line_cap)
        no_of_lines  = len(df_line)

        def style_fn(feat):
            p = feat["properties"]
            if p.get("down_weather", False):
                return {"color": "#000000", "weight": 3}
            pct = p.get("loading", 0.0)
            colour = (get_color_trafo(pct, max_trf_cap)
                      if df_trafo is not None and p["idx"] >= no_of_lines
                      else get_color(pct, max_line_cap))
            return {"color": colour, "weight": 3}

        folium.GeoJson(gdf.__geo_interface__, style_function=style_fn,
                       name=f"Transmission Net – Hour {hr}").add_to(m)

        # load circles
        shed_now = [b for (h, b) in shed_buses if h == hr]
        for _, row in df_load.iterrows():
            bus = row["bus"]
            lat, lon = ast.literal_eval(row["load_coordinates"])
            col = "red" if bus in shed_now else "green"
            folium.Circle((lat, lon), radius=20000,
                          color=col, fill_color=col, fill_opacity=0.5).add_to(m)

         # ---------- legend (replace the whole legend_html string) ------------------
        legend_html = """
        <style>
          .legend-box,* .legend-box { color:#000 !important; }
        </style>
        
        <div class="legend-box leaflet-control leaflet-bar"
             style="position:absolute; top:150px; left:10px; z-index:9999;
                    background:#ffffff; padding:8px; border:1px solid #ccc;
                    font-size:14px; max-width:210px;">
          <strong>Line Load Level&nbsp;(&#37; of Max)</strong><br>
          <span style='display:inline-block;width:12px;height:12px;background:#00FF00;'></span>&nbsp;Below&nbsp;75&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FFFF00;'></span>&nbsp;75–90&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FFA500;'></span>&nbsp;90–100&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FF0000;'></span>&nbsp;Overloaded&nbsp;>&nbsp;100&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#000000;'></span>&nbsp;Weather‑Impacted<br><br>
        
          <strong>Load Status</strong><br>
          <span style='display:inline-block;width:12px;height:12px;background:#008000;border-radius:50%;'></span>&nbsp;Fully Served<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FF0000;border-radius:50%;'></span>&nbsp;Not Fully Served
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        # ---------------- title (overwrite your title_html string) -----------------
        title_html = f"""
        <style>
          .map-title {{ color:#000 !important; }}
        </style>
        
        <div class="map-title leaflet-control leaflet-bar"
             style="position:absolute; top:90px; left:10px; z-index:9999;
                    background:rgba(255,255,255,0.9); padding:4px;
                    font-size:18px; font-weight:bold;">
          Projected Operation - Under Current OPF – Hour {hr}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        folium.LayerControl(collapsed=False).add_to(m)

        # display
        st.write(f"### Network Loading Visualization – Hour {hr}")
        st_folium(m, width=800, height=600, key=f"bau_map_{hr}")

        


# ────────────────────────────────────────────────────────────────────────────
# Page 5 :  Weather aware optimal power flow
# ────────────────────────────────────────────────────────────────────────────
elif selection == "Weather aware optimal power flow":
    st.header('Projected Operation – Under Weather Risk Aware OPF')

    st.title("Projected Operation – Under Weather Risk Aware OPF")
     # 🚦 NEW – make sure the cache from Page 3 exists
    if not st.session_state.get("bau_ready", False):
        st.info(
            "Please run **Projected Operation – Under Current OPF** first.  "
            "When it finishes you can return here and run the Weather-Aware analysis."
        )

    # ── 0 · SESSION  STATE  SLOTS  (create once) ────────────────────────────
    for k, v in (
        ("wa_ready",                       False),
        ("wa_day_end_df",                  None),
        ("wa_hourly_cost_df",              None),
    ):
        st.session_state.setdefault(k, v)
    # -----------------------------------------------------------------------

    # ── 1 · UI  – contingency mode picker + button ─────────────────────────
    mode = st.selectbox(
        "Select Contingency Mode",
        ["Capped Contingency Mode (20% of Network Lines)", "Maximum Contingency Mode (All Outage Lines)"],
    )
    cap_flag = (mode == "Capped Contingency Mode (20% of Network Lines)")

    if st.button("Run Weather Aware Optimal Power Flow (OPF) Analysis"):

        # -------------------------------------------------------------------
        # 1-A · Build the outage list exactly like Page-3
        # -------------------------------------------------------------------
        line_outages = functions.generate_line_outages(
            outage_hours   = st.session_state["outage_hours"],
            line_down      = st.session_state["line_down"],
            risk_scores    = st.session_state["risk_scores"],
            capped_contingency_mode = cap_flag,
        )
        st.session_state.line_outages = line_outages        # (re-use later)

         # ---------------------------------------------------------------------------
        # Aliases so the Colab names still resolve
        # ---------------------------------------------------------------------------
        #def Network_initialize():
        #    return network_initialize(path)          # <— your global helper
       # 
        #def overloaded_transformer_colab(net):
            # keep original single-arg call signature
        #    return overloaded_transformer(net, path, line_outages)
        # ---------------------------------------------------------------------------

        
        # —— helper so existing single-arg calls still work ——
        #def overloaded_transformer_local(net_):
        #    return overloaded_transformer(net_, path, line_outages)
        

        # -------------------------------------------------------------------
        # 1-B · DEFINE  weather_opf()   (same maths – no prints)
        # -------------------------------------------------------------------
        # ─────────────────────────────────────────────────────────────────────────────
        # WEATHER-AWARE OPF  – Streamlit friendly (no prints, returns DataFrames)
        # ─────────────────────────────────────────────────────────────────────────────
        
        # with st.spinner("Running weather-aware OPF …"):
        #     # day_end_df, hourly_cost_df = weather_opf(line_outages)
        #     (*_, day_end_df, hourly_cost_df) = weather_opf(line_outages)

        with st.spinner("Running Weather Aware Optimal Power Flow (OPF) Analysis (Estimated Time 5-10 minutes)…"):
            (
                loading_records,           # 0
                shedding_buses,            # 1
                _cum_load_shed,            # 2  ← underscores for things you don’t use
                _hourly_shed_wa,           # 3
                _wa_cost,                  # 4
                _seen_buses,               # 5
                _hourly_shed_dup,          # 6
                _served_load_wa,           # 7
                _loading_percent_wa,       # 8
                _gen_per_hour_wa,          # 9
                _slack_per_hour_wa,        # 10
                _planned_slack_per_hour,   # 11
                line_idx_map,              # 12
                trafo_idx_map,             # 13
                df_trafo,                  # 14
                df_lines,                  # 15
                _df_load_params,           # 16
                day_end_df,                # 17
                hourly_cost_df,            # 18
            ) = functions.weather_opf(st.session_state.line_outages)


        st.session_state.wa_ready          = True
        st.session_state.wa_day_end_df     = day_end_df
        st.session_state.wa_hourly_cost_df = hourly_cost_df

        # 🔻  put this *inside* the button block, right after the spinner  🔻
        st.session_state.update({
            "wa_ready":                        True,
            "wa_day_end_df":                   day_end_df,
            "wa_hourly_cost_df":               hourly_cost_df,
            "wa_results": {
                "loading_percent_wa": loading_records,
                "shedding_buses":     shedding_buses,
            },
            "hourly_shed_weather": _hourly_shed_wa,  
            "served_load_per_hour_wa": _served_load_wa,
            "gen_per_hour_wa":    _gen_per_hour_wa, 
            "slack_per_hour_wa":         _slack_per_hour_wa,     # ← NEW
            "planned_slack_per_hour":    _planned_slack_per_hour, # ← NEW
            "wa_line_idx_map":  line_idx_map,
            "wa_trafo_idx_map": trafo_idx_map,
            "wa_max_loading_capacity":         df_lines["max_loading_percent"].max(),
        })
        # if df_trafo is not None and not df_trafo.empty:
        #     st.session_state.wa_max_loading_capacity_transformer = (
        #         df_trafo["max_loading_percent"].max()
        #     )

        if isinstance(df_trafo, pd.DataFrame) and not df_trafo.empty:
                st.session_state.max_loading_capacity_transformer = (
                    df_trafo["max_loading_percent"].max()
                )

        # if st.session_state.wa_ready:
        #     st.subheader("Day-End Summary (Weather-Aware OPF)")
        #     st.dataframe(st.session_state.wa_day_end_df, use_container_width=True)
        
        #     st.subheader("Hourly Generation Cost (Weather-Aware OPF)")
        #     st.dataframe(st.session_state.wa_hourly_cost_df, use_container_width=True)


    # if st.session_state.get("wa_ready", False):
    
    #     st.subheader("Day-End Summary (Weather-Aware OPF)")
    #     st.dataframe(
    #         st.session_state.wa_day_end_df, use_container_width=True
    #     )
    
    #     st.subheader("Hourly Generation Cost (Weather-Aware OPF)")
    #     st.dataframe(
    #         st.session_state.wa_hourly_cost_df, use_container_width=True
    #     )
    # ░░ 1 ·  PERSIST RESULTS (right after weather_opf finishes) ░░
    # st.session_state.update({
    #     "wa_ready":                        True,
    #     "wa_day_end_df":                   day_end_df,
    #     "wa_hourly_cost_df":               hourly_cost_df,
    #     "wa_results": {                    # <── NEW: everything the map needs
    #         "loading_percent_wa": loading_records,
    #         "shedding_buses":    shedding_buses,
    #     },
    #     "wa_line_idx_map":                 line_idx_map,
    #     "wa_trafo_idx_map":                trafo_idx_map,
    #     "wa_max_loading_capacity":         df_lines["max_loading_percent"].max(),
    # })
    # if df_trafo is not None and not df_trafo.empty:
    #     st.session_state.wa_max_loading_capacity_transformer = (
    #         df_trafo["max_loading_percent"].max()
    #     )
    
    # ░░ 2 · ALWAYS-VISIBLE OUTPUT (tables + 24-hour map picker) ░░
    if st.session_state.get("wa_ready", False):
    
        # 2-A  summary tables ---------------------------------------------------
        st.subheader("Day Ahead Summary (Weather Risk Aware OPF)")
        st.dataframe(st.session_state.wa_day_end_df, use_container_width=True)
    
        #st.subheader("Hourly Generation Cost (Weather Risk Aware OPF)")
        #st.dataframe(st.session_state.wa_hourly_cost_df, use_container_width=True)
    
        # 2-B  hour picker (keeps its value in session_state.wa_hour) -----------
        num_hours = len(st.session_state.network_data['df_load_profile'])
        hour_options = list(range(num_hours))     # [0,1,…,23]
        hr = st.selectbox(
            "Select Hour to Visualize",
            hour_options,
            format_func=lambda h: f"Hour {h}",    # still shows “Hour 0”…“Hour 23”
            key="wa_hour",                        # now always an int
            help="Choose any hour; the map refreshes automatically.",
        )
    
        # 2-C  build the Folium map for that hour -------------------------------
        hr              = st.session_state.wa_hour
        df_line         = st.session_state.network_data['df_line'].copy()
        df_load         = st.session_state.network_data['df_load'].copy()
        df_trafo        = st.session_state.network_data.get('df_trafo')
        loading_rec     = st.session_state.wa_results['loading_percent_wa'][hr]
        shed_buses      = st.session_state.wa_results['shedding_buses']
        line_idx_map    = st.session_state.wa_line_idx_map
        trafo_idx_map   = st.session_state.wa_trafo_idx_map
        outages         = st.session_state.line_outages          # created earlier
    
        # — helper colour fns (identical logic to Page-3) -----------------------
        def get_color(pct, max_cap):
            if pct is None:                return '#FF0000'
            if pct == 0:                   return '#000000'
            if pct <= 0.75*max_cap:        return '#00FF00'
            if pct <= 0.90*max_cap:        return '#FFFF00'
            if pct <  max_cap:             return '#FFA500'
            return '#FF0000'
        get_color_trafo = get_color
    
        def check_bus_pair_df(df_line, df_trafo, pair):
            fbus, tbus = pair
            if df_trafo is not None:
                if (((df_trafo["hv_bus"] == fbus) & (df_trafo["lv_bus"] == tbus)) |
                    ((df_trafo["hv_bus"] == tbus) & (df_trafo["lv_bus"] == fbus))).any():
                    return True
            if (((df_line["from_bus"] == fbus) & (df_line["to_bus"] == tbus)) |
                ((df_line["from_bus"] == tbus) & (df_line["to_bus"] == fbus))).any():
                return False
            return None
    
        # lines → GeoDataFrame --------------------------------------------------
        df_line["geodata"] = df_line["geodata"].apply(
            lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
        )
        gdf = gpd.GeoDataFrame(
            df_line,
            geometry=[LineString(c) for c in df_line["geodata"]],
            crs="EPSG:4326",
        )
        gdf["idx"]     = gdf.index
        gdf["loading"] = gdf["idx"].map(lambda i: loading_rec[i] if i < len(loading_rec) else 0.0)
    
        # mark weather-down equipment
        weather_down = set()
        for fbus, tbus, start_hr in outages:
            if hr >= start_hr:
                is_tf = check_bus_pair_df(df_line, df_trafo, (fbus, tbus))
                if is_tf:
                    idx = trafo_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx + len(df_line))
                else:
                    idx = line_idx_map.get((fbus, tbus))
                    if idx is not None:
                        weather_down.add(idx)
        gdf["down_weather"] = gdf["idx"].isin(weather_down)
    
        # Folium map ------------------------------------------------------------
        m = folium.Map(location=[27, 66.5], zoom_start=6, width=800, height=600)
        max_line_cap = st.session_state.wa_max_loading_capacity
        max_trf_cap  = st.session_state.get("wa_max_loading_capacity_transformer",
                                            max_line_cap)
        no_of_lines  = len(df_line)
    
        def style_fn(feat):
            p = feat["properties"]
            if p.get("down_weather", False):
                return {"color": "#000000", "weight": 3}
            pct = p.get("loading", 0.0)
            colour = (get_color_trafo(pct, max_trf_cap)
                      if df_trafo is not None and p["idx"] >= no_of_lines
                      else get_color(pct, max_line_cap))
            return {"color": colour, "weight": 3}
    
        folium.GeoJson(gdf.__geo_interface__, style_function=style_fn,
                       name=f"Transmission Net – Hour {hr}").add_to(m)
    
        # load circles (served vs shed) ----------------------------------------
        shed_now = [b for (h, b) in shed_buses if h == hr]
        for _, row in df_load.iterrows():
            bus = row["bus"]
            lat, lon = ast.literal_eval(row["load_coordinates"])
            col = "red" if bus in shed_now else "green"
            folium.Circle((lat, lon), radius=20000,
                          color=col, fill_color=col, fill_opacity=0.5).add_to(m)
    
        # legend + title (same HTML you used before) ----------------------------
        legend_html = """
        <style>
          .legend-box,* .legend-box { color:#000 !important; }
        </style>
        
        <div class="legend-box leaflet-control leaflet-bar"
             style="position:absolute; top:150px; left:10px; z-index:9999;
                    background:#ffffff; padding:8px; border:1px solid #ccc;
                    font-size:14px; max-width:210px;">
          <strong>Line Load Level&nbsp;(&#37; of Max)</strong><br>
          <span style='display:inline-block;width:12px;height:12px;background:#00FF00;'></span>&nbsp;Below&nbsp;75&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FFFF00;'></span>&nbsp;75–90&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FFA500;'></span>&nbsp;90–100&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FF0000;'></span>&nbsp;Overloaded&nbsp;>&nbsp;100&nbsp;%<br>
          <span style='display:inline-block;width:12px;height:12px;background:#000000;'></span>&nbsp;Weather‑Impacted<br><br>
        
          <strong>Load Status</strong><br>
          <span style='display:inline-block;width:12px;height:12px;background:#008000;border-radius:50%;'></span>&nbsp;Fully Served<br>
          <span style='display:inline-block;width:12px;height:12px;background:#FF0000;border-radius:50%;'></span>&nbsp;Not Fully Served
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
       # ---------------- title (overwrite your title_html string) -----------------
        title_html = f"""
        <style>
          .map-title {{ color:#000 !important; }}
        </style>
        
        <div class="map-title leaflet-control leaflet-bar"
             style="position:absolute; top:90px; left:10px; z-index:9999;
                    background:rgba(255,255,255,0.9); padding:4px;
                    font-size:18px; font-weight:bold;">
          Projected Operation - Under Weather Risk Aware OPF – Hour {hr}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
    
        folium.LayerControl(collapsed=False).add_to(m)
    
        st.write(f"### Network Loading Visualization – Hour {hr}")
        st_folium(m, width=800, height=600, key=f"wa_map_{hr}")

