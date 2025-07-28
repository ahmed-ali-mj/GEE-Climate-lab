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
from datetime import datetime, timedelta
import random
import geemap.foliumap as geemap
import numpy as np
import math
import traceback
import plotly.graph_objects as go
from shapely.geometry import LineString, Point
import plotly.express as px



# Shared function: Add EE Layer to Folium Map (used in both pages)
def add_ee_layer(self, ee_object, vis_params, name):
    try:
        if isinstance(ee_object, ee.image.Image):
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        elif isinstance(ee_object, ee.geometry.Geometry) or isinstance(ee_object, ee.feature.Feature) or isinstance(ee_object, ee.featurecollection.FeatureCollection):
            map_id_dict = ee_object.getMapId(vis_params)
            folium.raster_layers.TileLayer(
                tiles=map_id_dict['tile_fetcher'].url_format,
                attr='Google Earth Engine',
                name=name,
                overlay=True,
                control=True
            ).add_to(self)
        else:
            print("Could not add EE layer of type {}".format(type(ee_object)))
    except Exception as e:
        print(f"Error adding EE layer: {e}")




# ---------------------------------------------------------------------
# 1) Network INITIALISE  (was `Network_initialize` in Colab)
# ----------------------------------------------------------
def network_initialize(xls_file):
    """
    Re-creates a fresh pandapower network from the uploaded Excel *every* time
    Page-3 (baseline OPF) or Page-4 (weather-aware OPF) is run, so that both
    simulations start from the identical initial state.

    Parameters
    ----------
    xls_file : BytesIO or str
        The same object you get back from `st.file_uploader` (i.e. the upload
        stream).  A local file path also works, so existing unit-tests keep
        passing.

    Returns
    -------
    tuple
        [net, df_bus, df_slack, df_line, num_hours,
         load_dynamic, gen_dynamic,
         df_load_profile, df_gen_profile, *optional df_trafo*]
    """

    # --- 0. Fresh empty network
    net = pp.create_empty_network()

    # --- 1. Read all static sheets ------------------------------------------------
    df_bus   = pd.read_excel(xls_file, sheet_name="Bus Parameters",      index_col=0)
    df_load  = pd.read_excel(xls_file, sheet_name="Load Parameters",     index_col=0)
    df_slack = pd.read_excel(xls_file, sheet_name="Generator Parameters",index_col=0)
    df_line  = pd.read_excel(xls_file, sheet_name="Line Parameters",     index_col=0)
    # df_gen_params = pd.read_excel(xls_file, sheet_name="Generator Parameters")


    # --- 2. Build static elements -------------------------------------------------
    for _, row in df_bus.iterrows():
        pp.create_bus(net,
                      name          = row["name"],
                      vn_kv         = row["vn_kv"],
                      zone          = row["zone"],
                      in_service    = row["in_service"],
                      max_vm_pu     = row["max_vm_pu"],
                      min_vm_pu     = row["min_vm_pu"])

    for _, row in df_load.iterrows():
        pp.create_load(net,
                       bus           = row["bus"],
                       p_mw          = row["p_mw"],
                       q_mvar        = row["q_mvar"],
                       in_service    = row["in_service"])

    for _, row in df_slack.iterrows():
        if row["slack_weight"] == 1:
            ext_idx = pp.create_ext_grid(net,
                                         bus        = row["bus"],
                                         vm_pu      = row["vm_pu"],
                                         va_degree  = 0)
            pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                cp0_eur_per_mw = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])
        else:
            gen_idx = pp.create_gen(net,
                                    bus         = row["bus"],
                                    p_mw        = row["p_mw"],
                                    vm_pu       = row["vm_pu"],
                                    min_q_mvar  = row["min_q_mvar"],
                                    max_q_mvar  = row["max_q_mvar"],
                                    scaling     = row["scaling"],
                                    in_service  = row["in_service"],
                                    slack_weight= row["slack_weight"],
                                    controllable= row["controllable"],
                                    max_p_mw    = row["max_p_mw"],
                                    min_p_mw    = row["min_p_mw"])
            pp.create_poly_cost(net, element=gen_idx, et="gen",
                                cp0_eur_per_mw = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])

    for _, row in df_line.iterrows():
        if pd.isna(row["parallel"]):
            continue
        geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
        pp.create_line_from_parameters(net,
                                       from_bus             = row["from_bus"],
                                       to_bus               = row["to_bus"],
                                       length_km            = row["length_km"],
                                       r_ohm_per_km         = row["r_ohm_per_km"],
                                       x_ohm_per_km         = row["x_ohm_per_km"],
                                       c_nf_per_km          = row["c_nf_per_km"],
                                       max_i_ka             = row["max_i_ka"],
                                       in_service           = row["in_service"],
                                       max_loading_percent  = row["max_loading_percent"],
                                       geodata              = geodata)

    # --- 3. Optional transformers -------------------------------------------------
    xls_obj = pd.ExcelFile(xls_file)
    if "Transformer Parameters" in xls_obj.sheet_names:
        df_trafo = pd.read_excel(xls_file, sheet_name="Transformer Parameters", index_col=0)
        for _, row in df_trafo.iterrows():
            pp.create_transformer_from_parameters(net,
                 hv_bus    = row["hv_bus"],
                 lv_bus    = row["lv_bus"],
                 sn_mva    = row["sn_mva"],
                 vn_hv_kv  = row["vn_hv_kv"],
                 vn_lv_kv  = row["vn_lv_kv"],
                 vk_percent= row["vk_percent"],
                 vkr_percent=row["vkr_percent"],
                 pfe_kw    = row["pfe_kw"],
                 i0_percent= row["i0_percent"],
                 in_service=row["in_service"],
                 max_loading_percent=row["max_loading_percent"])

    # --- 4. Dynamic-profile helpers ----------------------------------------------
    df_load_profile = pd.read_excel(xls_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()

    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus  = int(m.group(1))
            qcol = f"q_mvar_bus_{bus}"
            if qcol in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": qcol}

    df_gen_profile = pd.read_excel(xls_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()

    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            nums = re.findall(r"\d+", col)
            if nums:
                gen_dynamic[int(nums[-1])] = col

    num_hours = len(df_load_profile)

    # --- 5. Return exactly what Colab did -----------------------------------------
    if "Transformer Parameters" in xls_obj.sheet_names:
        return (net, df_bus, df_slack, df_line,
                num_hours, load_dynamic, gen_dynamic,
                df_load_profile, df_gen_profile, df_trafo)

    return (net, df_bus, df_slack, df_line,
            num_hours, load_dynamic, gen_dynamic,
            df_load_profile, df_gen_profile)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Colab-equivalent: rebuild + 24-h OPF loop on every call
# ---------------------------------------------------------------------------
def calculating_hourly_cost(xls_file):
    """
    Streamlit drop-in replacement for the Colab routine.

    Parameters
    ----------
    xls_file : BytesIO | str
        The file object returned by `st.file_uploader` *or* a filesystem path.

    Returns
    -------
    list
        Length == number of rows in “Load Profile”.
        Each element is net.res_cost (DataFrame) when OPF succeeded,
        otherwise the integer 0.
    """
    # 0) Prep
    xls = pd.ExcelFile(xls_file)
    hourly_cost_list = []
    net = pp.create_empty_network()

    # 1) Static sheets ----------------------------------------------------------
    df_bus   = pd.read_excel(xls_file, sheet_name="Bus Parameters",      index_col=0)
    df_load  = pd.read_excel(xls_file, sheet_name="Load Parameters",     index_col=0)
    df_slack = pd.read_excel(xls_file, sheet_name="Generator Parameters",index_col=0)
    df_line  = pd.read_excel(xls_file, sheet_name="Line Parameters",     index_col=0)

    # 2) Create buses -----------------------------------------------------------
    for _, row in df_bus.iterrows():
        pp.create_bus(net,
                      name        = row["name"],
                      vn_kv       = row["vn_kv"],
                      zone        = row["zone"],
                      in_service  = row["in_service"],
                      max_vm_pu   = row["max_vm_pu"],
                      min_vm_pu   = row["min_vm_pu"])

    # 3) Create loads -----------------------------------------------------------
    for _, row in df_load.iterrows():
        pp.create_load(net,
                       bus        = row["bus"],
                       p_mw       = row["p_mw"],
                       q_mvar     = row["q_mvar"],
                       in_service = row["in_service"])

    # 4) Create generators / ext-grid ------------------------------------------
    for _, row in df_slack.iterrows():
        if row["slack_weight"] == 1:
            ext_idx = pp.create_ext_grid(net,
                                         bus       = row["bus"],
                                         vm_pu     = row["vm_pu"],
                                         va_degree = 0)
            pp.create_poly_cost(net, element=ext_idx, et="ext_grid",
                                cp0_eur_per_mw   = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw   = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw   = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])
        else:
            gen_idx = pp.create_gen(net,
                                    bus          = row["bus"],
                                    p_mw         = row["p_mw"],     # overwritten hourly
                                    vm_pu        = row["vm_pu"],
                                    min_q_mvar   = row["min_q_mvar"],
                                    max_q_mvar   = row["max_q_mvar"],
                                    scaling      = row["scaling"],
                                    in_service   = row["in_service"],
                                    slack_weight = row["slack_weight"],
                                    controllable = row["controllable"],
                                    max_p_mw     = row["max_p_mw"],
                                    min_p_mw     = row["min_p_mw"])
            pp.create_poly_cost(net, element=gen_idx, et="gen",
                                cp0_eur_per_mw   = row["cp0_pkr_per_mw"],
                                cp1_eur_per_mw   = row["cp1_pkr_per_mw"],
                                cp2_eur_per_mw   = row["cp2_pkr_per_mw"],
                                cp0_eur_per_mvar = row["cp0_pkr_per_mvar"],
                                cq1_eur_per_mvar = row["cq1_pkr_per_mvar"],
                                cq2_eur_per_mvar = row["cq2_pkr_per_mvar"])

    # 5) Create lines -----------------------------------------------------------
    for _, row in df_line.iterrows():
        if pd.isna(row["parallel"]):
            continue
        geodata = ast.literal_eval(row["geodata"]) if isinstance(row["geodata"], str) else row["geodata"]
        pp.create_line_from_parameters(net,
                                       from_bus            = row["from_bus"],
                                       to_bus              = row["to_bus"],
                                       length_km           = row["length_km"],
                                       r_ohm_per_km        = row["r_ohm_per_km"],
                                       x_ohm_per_km        = row["x_ohm_per_km"],
                                       c_nf_per_km         = row["c_nf_per_km"],
                                       max_i_ka            = row["max_i_ka"],
                                       in_service          = row["in_service"],
                                       max_loading_percent = row["max_loading_percent"],
                                       geodata             = geodata)

    # 6) Optional transformers ---------------------------------------------------
    if "Transformer Parameters" in xls.sheet_names:
        df_trafo = pd.read_excel(xls_file, sheet_name="Transformer Parameters", index_col=0)
        for _, row in df_trafo.iterrows():
            pp.create_transformer_from_parameters(net,
                 hv_bus             = row["hv_bus"],
                 lv_bus             = row["lv_bus"],
                 sn_mva             = row["sn_mva"],
                 vn_hv_kv           = row["vn_hv_kv"],
                 vn_lv_kv           = row["vn_lv_kv"],
                 vk_percent         = row["vk_percent"],
                 vkr_percent        = row["vkr_percent"],
                 pfe_kw             = row["pfe_kw"],
                 i0_percent         = row["i0_percent"],
                 in_service         = row["in_service"],
                 max_loading_percent= row["max_loading_percent"])

    # 7) Dynamic-profile helpers -------------------------------------------------
    df_load_profile = pd.read_excel(xls_file, sheet_name="Load Profile")
    df_load_profile.columns = df_load_profile.columns.str.strip()

    load_dynamic = {}
    for col in df_load_profile.columns:
        m = re.match(r"p_mw_bus_(\d+)", col)
        if m:
            bus   = int(m.group(1))
            q_col = f"q_mvar_bus_{bus}"
            if q_col in df_load_profile.columns:
                load_dynamic[bus] = {"p": col, "q": q_col}

    df_gen_profile = pd.read_excel(xls_file, sheet_name="Generator Profile")
    df_gen_profile.columns = df_gen_profile.columns.str.strip()

    gen_dynamic = {}
    for col in df_gen_profile.columns:
        if col.startswith("p_mw"):
            nums = re.findall(r"\d+", col)
            if nums:
                gen_dynamic[int(nums[-1])] = col

    num_hours = len(df_load_profile)

    # 8) Hour-by-hour OPF loop ---------------------------------------------------
    for hour in range(num_hours):

        # 8.1 update loads
        for bus_id, cols in load_dynamic.items():
            p_val = float(df_load_profile.at[hour, cols["p"]])
            q_val = float(df_load_profile.at[hour, cols["q"]])
            mask  = net.load.bus == bus_id
            net.load.loc[mask, "p_mw"]  = p_val
            net.load.loc[mask, "q_mvar"] = q_val

        # 8.2 update gens / ext grid
        for bus_id, col in gen_dynamic.items():
            p_val = float(df_gen_profile.at[hour, col])
            if bus_id in net.ext_grid.bus.values:
                net.ext_grid.loc[net.ext_grid.bus == bus_id, "p_mw"] = p_val
            else:
                net.gen.loc[net.gen.bus == bus_id, "p_mw"] = p_val

        # 8.3 run OPF
        try:
            pp.runopp(net)
            hourly_cost_list.append(net.res_cost)
        except Exception:
            hourly_cost_list.append(0)
            continue

    return hourly_cost_list

# ------------------------------------------------------------------
# 1)  all_real_numbers  — identical to your Colab version
# ------------------------------------------------------------------
def all_real_numbers(lst):
    invalid_count = 0
    for x in lst:
        # 1) Not numeric
        if not isinstance(x, (int, float)):
            invalid_count += 1
        # 2) NaN or infinite
        elif not math.isfinite(x):
            invalid_count += 1

    if invalid_count > len(line_outages):
        return False
    return True


# ------------------------------------------------------------------
# 2)  overloaded_lines  — identical to your Colab version
# ------------------------------------------------------------------
def overloaded_lines(net):
    overloaded = []
    # turn loading_percent Series into a list once
    loadings = transform_loading(net.res_line["loading_percent"])
    real_check = all_real_numbers(net.res_line["loading_percent"].tolist())

    for idx, (res, loading_val) in enumerate(zip(net.res_line.itertuples(), loadings)):
        # grab this line’s own max
        own_max = net.line.at[idx, "max_loading_percent"]
        # print(f"max loading capacity @ id {id} is {own_max}.")

        if not real_check:
            # any NaN/non-numeric or at-limit is overloaded
            if not isinstance(loading_val, (int, float)) or math.isnan(loading_val) or loading_val >= own_max:
                overloaded.append(idx)
        else:
            # only truly > its own max
            if loading_val is not None and not (isinstance(loading_val, float) and math.isnan(loading_val)) and loading_val > own_max:
                overloaded.append(idx)
    return overloaded

# -------------------------------------------------------------
# 1)  Does the (from_bus, to_bus) pair correspond to a trafo?
# -------------------------------------------------------------
def check_bus_pair(xls_file, bus_pair):
    """
    Parameters
    ----------
    xls_file : BytesIO | str
        Same object you get from st.file_uploader – or a path.
    bus_pair : tuple[int, int]
        (from_bus, to_bus) to look up.

    Returns
    -------
    True   → pair matches a transformer
    False  → pair matches a line
    None   → no match found
    """
    xls = pd.ExcelFile(xls_file)

    if "Transformer Parameters" in xls.sheet_names:
        transformer_df = pd.read_excel(xls_file, sheet_name="Transformer Parameters")
        line_df        = pd.read_excel(xls_file, sheet_name="Line Parameters")

        from_bus, to_bus = bus_pair

        transformer_match = (
            ((transformer_df["hv_bus"] == from_bus) & (transformer_df["lv_bus"] == to_bus)) |
            ((transformer_df["hv_bus"] == to_bus) & (transformer_df["lv_bus"] == from_bus))
        ).any()

        line_match = (
            ((line_df["from_bus"] == from_bus) & (line_df["to_bus"] == to_bus)) |
            ((line_df["from_bus"] == to_bus)  & (line_df["to_bus"] == from_bus))
        ).any()

        if transformer_match:
            return True
        if line_match:
            return False

    # nothing matched
    return None
# -------------------------------------------------------------


# -------------------------------------------------------------
# 2)  Normalise “loading_percent” fields so units are consistent
# -------------------------------------------------------------
def transform_loading(a):
    """
    Multiplies every value < 2.5 by 100 so that fractional %
    values (e.g. 0.95) become full percentages (95.0).
    Works for scalars or lists.  Returns the same “shape” back.
    """
    if a is None:
        return a

    # turn scalars into a list for uniform processing
    is_single = False
    if isinstance(a, (int, float)):
        a         = [a]
        is_single = True

    # decide whether conversion is needed
    flag = True
    for item in a:
        if isinstance(item, (int, float)) and item >= 2.5:
            flag = False

    if flag:
        a = [item * 100 if isinstance(item, (int, float)) else item for item in a]

    return a[0] if is_single else a
# -------------------------------------------------------------

# -------------------------------------------------------------
# 5)  Identify overloaded transformers (if any exist)
# -------------------------------------------------------------
def overloaded_transformer(net, xls_file, line_outages):
    """
    Same logic as Colab version but *xls_file* is explicit.
    Returns list of transformer indices exceeding their max loading.
    """
    overloaded = []

    xls = pd.ExcelFile(xls_file)
    if "Transformer Parameters" not in xls.sheet_names:
        return overloaded

    loadings   = transform_loading(net.res_trafo["loading_percent"])
    # real_check = all_real_numbers(net.res_trafo["loading_percent"].tolist(),
    #                               line_outages)
    real_check = all_real_numbers(net.res_trafo["loading_percent"].tolist())


    for idx, (_, loading_val) in enumerate(zip(net.res_trafo.itertuples(),
                                               loadings)):
        own_max = net.trafo.at[idx, "max_loading_percent"]

        if not real_check:
            if (loading_val is not None and
                not (isinstance(loading_val, float) and math.isnan(loading_val)) and
                loading_val >= own_max):
                overloaded.append(idx)
        else:
            if loading_val > own_max:
                overloaded.append(idx)
    return overloaded
# -------------------------------------------------------------
# ------------------------------------------------------------------


def generate_line_outages(outage_hours, line_down, risk_scores,
                          capped_contingency_mode=False):
    """
    Returns a list of (from_bus, to_bus, outage_hour) tuples.

    When *capped_contingency_mode* is True keep only the worst 20 % lines
    in the sense of
        1) higher risk-score first, then
        2) earlier outage-hour first.
    """

    # nothing to do
    if not (outage_hours and line_down and risk_scores):
        return []

    # ── 1 · normalise / align the risk-score list ──────────────────────
    needed  = len(line_down)
    numeric = []

    for r in risk_scores:
        # risk can arrive either as a plain number or a tiny dict
        if isinstance(r, dict):
            r = r.get("risk_score", 0)

        if isinstance(r, (int, float)):
            numeric.append(r)

        if len(numeric) == needed:                 # we have enough
            break

    # pad with zeros if Page-2 returned fewer scores than lines
    numeric += [0] * (needed - len(numeric))

    # ── 2 · build the working list (fbus, tbus, hour, score) ───────────
    combined = [
        (line[0], line[1], hour, score)
        for line, hour, score in zip(line_down, outage_hours, numeric)
    ]

    # ── 3 · sort by our 2-key rule  (-score → descending) ──────────────
    combined.sort(key=lambda x: (-x[3], x[2]))     # (score desc, hour asc)

    # ── 4 · apply the 20 % cap if requested ────────────────────────────
    if capped_contingency_mode:
        n_lines      = len(pd.read_excel(path, sheet_name="Line Parameters")) - 1
        capped_limit = math.floor(0.20 * n_lines)
        combined     = combined[:capped_limit]

    # ── 5 · return what the rest of the code expects ───────────────────
    return [(f, t, hr) for f, t, hr, _ in combined]



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




# Shared function: Create and display the map (used in Network Initialization)
def create_map(df_line, df_load):

    m = geemap.Map()
    m.to_streamlit(width=700, height=500)

    try:
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
        
        
        
        
        karachi = ee.Geometry.Point(67.0011, 24.8607)
        roi = karachi.buffer(600000).bounds()
    
    
        Map = geemap.Map()
        Map.centerObject(karachi, 5)
        Map.addLayer(st.session_state.point_assets, {'color': 'red'}, 'Infrastructure Point Assets');
        Map.addLayer(st.session_state.line_assets, {'color': 'black'}, 'Infrastructure Line Assets');

        # Add layer control
        Map.addLayerControl()

        # Render the map in Streamlit
        #Map.to_streamlit(width=700, height=500)
        '''
        # Add transmission lines to map
        m.add_ee_layer(
            line_fc.style(**{'color': 'red', 'width': 2}),
            {},
            "Transmission Lines"
        )
       
        

        # Add load points to map
        m.add_ee_layer(
            point_fc.style(**{'color': 'blue', 'pointSize': 5, 'fillColor': 'blue'}),
            {},
            "Load"
        )

        # Layer control
        folium.LayerControl(collapsed=False).add_to(m)

        
        '''
        
        return Map
    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        return None




