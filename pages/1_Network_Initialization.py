# ────────────────────────────────────────────────────────────────────────────
# Page# 1 :  Network Initialization
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandapower as pp
import pandas as pd
import ast
import re
import ee
import geemap.foliumap as geemap


# === Parse and create EE FeatureCollection for load points ===
def parse_point(row):
    try:
        lon, lat = eval(row["load_coordinates"])
        return ee.Feature(ee.Geometry.Point([lat, lon]), {'name': str(row.get('name', 'Load'))})
    except:
        return None

def compute_average_lat_lon(df_load):
    lon_list = []
    lat_list = []

    for _, row in df_load.iterrows():
        try:
            lat, lon = eval(row["load_coordinates"])  # assumes format "(70.123, 30.456)"
            lon_list.append(float(lon))
            lat_list.append(float(lat))
        except Exception as e:
            print(f"Skipping row: {e}")

    if lon_list and lat_list:
        avg_lon = sum(lon_list) / len(lon_list)
        avg_lat = sum(lat_list) / len(lat_list)
        return avg_lat, avg_lon
    else:
        return None, None




st.header('Page 1: Network Initialization')


st.header('Network Initialization')

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
            #st.session_state.map_obj = create_map(st.session_state.network_data['df_line'],st.session_state.network_data['df_load'])
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
        point_features = [parse_point(row) for _, row in df_load.iterrows()]
        point_features = [f for f in point_features if f is not None]
        point_fc = ee.FeatureCollection(point_features)


        st.session_state.point_assets = point_fc


        # Create base map
        avg_lat, avg_lon = compute_average_lat_lon(df_load)
        
        
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

