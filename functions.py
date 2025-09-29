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


def Network_initialize():
            return network_initialize(st.session_state.get("uploaded_file"))          # <— your global helper
        
def overloaded_transformer_colab(net):
    # keep original single-arg call signature
    return overloaded_transformer(net, st.session_state.get("uploaded_file"), st.session_state.line_outages)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MAIN FUNCTION – unchanged numerical logic, no prints, returns DataFrames
# ---------------------------------------------------------------------------
def current_opf(line_outages):
    df_trafo = []
    
    path = st.session_state.get("uploaded_file")   # BytesIO object

    xls = pd.ExcelFile(path)                       # same object Colab used

    # ----------------------------------------------------------------------
    # 1. Build fresh network + get helper objects
    # ----------------------------------------------------------------------
    if "Transformer Parameters" in xls.sheet_names:
        [net, df_bus, df_slack, df_line, num_hours,
         load_dynamic, gen_dynamic,
         df_load_profile, df_gen_profile, df_trafo] = Network_initialize()
    else:
        [net, df_bus, df_slack, df_line, num_hours,
         load_dynamic, gen_dynamic,
         df_load_profile, df_gen_profile]          = Network_initialize()

    business_as_usuall_cost = calculating_hourly_cost(path)

    # ----------------------------------------------------------------------
    # 2. Set up spatial helpers (identical to Colab)
    # ----------------------------------------------------------------------
    df_lines = df_line.copy()
    df_lines["geodata"] = df_lines["geodata"].apply(
        lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)

    gdf = gpd.GeoDataFrame(
        df_lines,
        geometry=[LineString(coords) for coords in df_lines["geodata"]],
        crs="EPSG:4326")

    load_df = pd.read_excel(path, sheet_name="Load Parameters")
    load_df["coordinates"] = load_df["load_coordinates"].apply(ast.literal_eval)

    # line / trafo index maps
    line_idx_map = {(r["from_bus"], r["to_bus"]): idx for idx, r in net.line.iterrows()}
    line_idx_map.update({(r["to_bus"], r["from_bus"]): idx for idx, r in net.line.iterrows()})

    trafo_idx_map = {}
    if "Transformer Parameters" in xls.sheet_names:
        trafo_idx_map = {(r["hv_bus"], r["lv_bus"]): idx for idx, r in net.trafo.iterrows()}
        trafo_idx_map.update({(r["lv_bus"], r["hv_bus"]): idx for idx, r in net.trafo.iterrows()})

    # ----------------------------------------------------------------------
    # 3. Book-keeping dicts  (unchanged)
    # ----------------------------------------------------------------------
    net.load["bus"] = net.load["bus"].astype(int)
    cumulative_load_shedding = {bus: {"p_mw": 0.0, "q_mvar": 0.0}
                                for bus in net.load["bus"].unique()}

    total_demand_per_bus = {}
    p_cols = [c for c in df_load_profile.columns if c.startswith("p_mw_bus_")]
    q_cols = [c for c in df_load_profile.columns if c.startswith("q_mvar_bus_")]
    bus_ids = set(int(col.rsplit("_", 1)[1]) for col in p_cols)
    for bus in bus_ids:
        p_col, q_col = f"p_mw_bus_{bus}", f"q_mvar_bus_{bus}"
        total_demand_per_bus[bus] = {"p_mw": float(df_load_profile[p_col].sum()),
                                     "q_mvar": float(df_load_profile[q_col].sum())}

    # ----------------------------------------------------------------------
    # 4. Fixed 20 % shed fractions (same logic)
    # ----------------------------------------------------------------------
    initial_load_p = {}   # real power
    initial_load_q = {}   # reactive
    initial_load_p = {int(net.load.at[i, "bus"]): net.load.at[i, "p_mw"]
                      for i in net.load.index}
    initial_load_q = {int(net.load.at[i, "bus"]): net.load.at[i, "q_mvar"]
                      for i in net.load.index}
    shed_pct = 0.10
    fixed_shed_p = {b: shed_pct * p for b, p in initial_load_p.items()}
    fixed_shed_q = {b: shed_pct * q for b, q in initial_load_q.items()}

    # ----------------------------------------------------------------------
    # 5. Storage for hour-by-hour results
    # ----------------------------------------------------------------------
    hourly_shed_bau     = [0] * num_hours
    loading_records     = []
    loading_percent_bau = []
    served_load_per_hour= []
    gen_per_hour_bau    = []
    slack_per_hour_bau  = []
    shedding_buses      = []
    seen_buses          = set()

    # ----------------------------------------------------------------------
    # 6. ====  HOURLY LOOP  =================================================
    # ----------------------------------------------------------------------
    for hour in range(num_hours):
        # print(f"========== HOUR {hour} ==========")

        # 6-a) Apply scheduled outages
        for (fbus, tbus, start_hr) in line_outages:
            if hour < start_hr:
                continue
            is_trafo = check_bus_pair(path, (fbus, tbus))
            if is_trafo == True:
                mask_tf = (((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                           ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus)))    
                if not mask_tf.any():
                    pass
                else:
                    for tf_idx in net.trafo[mask_tf].index:
                        net.trafo.at[tf_idx, "in_service"] = False
            else:
                idx = line_idx_map.get((fbus, tbus))
                if idx is not None:
                    net.line.at[idx, "in_service"] = False

        # 6-b) Update hourly load & gen profiles
        for idx in net.load.index:
            bus = net.load.at[idx, "bus"]
            if bus in load_dynamic:
                net.load.at[idx, "p_mw"] = df_load_profile.at[hour, load_dynamic[bus]["p"]]
                net.load.at[idx, "q_mvar"]= df_load_profile.at[hour, load_dynamic[bus]["q"]]
        for idx in net.gen.index:
            bus = net.gen.at[idx, "bus"]
            if bus in gen_dynamic:
                net.gen.at[idx, "p_mw"] = df_gen_profile.at[hour, gen_dynamic[bus]]

        # 6-c) Re-read criticality each hour (kept identical)
        df_load_params = pd.read_excel(path, sheet_name="Load Parameters", index_col=0)
        crit_map = dict(zip(df_load_params["bus"], df_load_params["criticality"]))
        net.load["bus"] = net.load["bus"].astype(int)
        net.load["criticality"] = net.load["bus"].map(crit_map)

        # 6-d) Initial power-flow try
        flag_initial_fail = False
        try:
            pp.runpp(net)
        except:
            flag_initial_fail = True

        if flag_initial_fail == False:
            inter = transform_loading(net.res_line["loading_percent"])
            if "Transformer Parameters" in xls.sheet_names:
                inter.extend(transform_loading(net.res_trafo["loading_percent"].tolist()))
            loading_records.append(inter)
            loading_percent_bau.append(inter.copy())
        else:
            loading_records.append([0]*(len(net.res_line)+len(df_trafo)))
            loading_percent_bau.append([0]*(len(net.res_line)+len(df_trafo)))

        # 6-e) Check overloads and shed if needed
        overloads       = overloaded_lines(net)
        overloads_trafo = overloaded_transformer_colab(net)
        all_loads_zero_flag = False

        if (overloads == []) and (overloads_trafo == []) and (all_real_numbers(loading_records[-1])):
            slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))
            # served_load_per_hour.append(net.load["p_mw"].tolist())
            # gen_per_hour_bau.append(net.res_gen["p_mw"].tolist())

            if net.load["p_mw"].isnull().any():
                served_load_per_hour.append([None] * len(net.load))
            else:
                hourly_loads = net.load["p_mw"].tolist()
                served_load_per_hour.append(hourly_loads)

            if net.res_gen["p_mw"].isnull().any():
                gen_per_hour_bau.append([None] * len(net.res_gen))
            else:
                hourly_gen = net.res_gen["p_mw"].tolist()
                gen_per_hour_bau.append(hourly_gen)
            continue
        else:
            while ((overloaded_lines(net) or overloaded_transformer_colab(net)) and
                   not all_loads_zero_flag):

                for crit in sorted(net.load["criticality"].dropna().unique(), reverse=True):
                    for ld_idx in net.load[net.load["criticality"] == crit].index:
                        if (not overloaded_lines(net)) and (not overloaded_transformer_colab(net)):
                            break

                        bus = net.load.at[ld_idx, "bus"]
                        dp, dq = fixed_shed_p[bus], fixed_shed_q[bus]
                        net.load.at[ld_idx, "p_mw"]  -= dp
                        net.load.at[ld_idx, "q_mvar"]-= dq

                        shedding_buses.append((hour, int(bus)))
                        cumulative_load_shedding[bus]["p_mw"]  += dp
                        cumulative_load_shedding[bus]["q_mvar"]+= dq
                        hourly_shed_bau[hour]                  += dp

                        try:
                            pp.runopp(net)
                            business_as_usuall_cost[hour] = net.res_cost if net.OPF_converged else business_as_usuall_cost[hour]
                            if net.OPF_converged:
                                pf_loading = transform_loading(net.res_line["loading_percent"])
                                if "Transformer Parameters" in xls.sheet_names:
                                    pf_loading.extend(transform_loading(net.res_trafo["loading_percent"]))
                                if all_real_numbers(pf_loading):
                                    all_loads_zero_flag = True
                            business_as_usuall_cost[hour] = net.res_cost       
                        except:
                            pp.runpp(net)
                        
                        # if this load has now gone negative, slam to zero
                        if net.load.at[ld_idx, "p_mw"] - dp < 0:
                            all_loads_zero_flag = True
                            business_as_usuall_cost[hour] = 0
                            
                            remaining_p = net.load.loc[net.load["bus"] == bus, "p_mw"].sum()
                            remaining_q = net.load.loc[net.load["bus"] == bus, "q_mvar"].sum()
                            cumulative_load_shedding[bus]["p_mw"]  += remaining_p
                            cumulative_load_shedding[bus]["q_mvar"]+= remaining_q
                            hourly_shed_bau[hour] += sum(net.load["p_mw"])
                            
                            for i in range(len(net.load)):
                                net.load.at[i, 'p_mw'] = 0
                                net.load.at[i, 'q_mvar'] = 0
                            break

            # record final served, gen, slack
            if net.load["p_mw"].isnull().any():
                served_load_per_hour.append([None] * len(net.load))
            else:
                hourly_loads = net.load["p_mw"].tolist()
                served_load_per_hour.append(hourly_loads)

            
            if (net.res_gen["p_mw"].isnull().any()) or (business_as_usuall_cost[hour] == 0):
                gen_per_hour_bau.append([None]*len(net.res_gen))
                slack_per_hour_bau.append(None)
            else:
                hourly_gen = net.res_gen["p_mw"].tolist()
                gen_per_hour_bau.append(net.res_gen["p_mw"].tolist())
                slack_per_hour_bau.append(float(net.res_ext_grid.at[0, "p_mw"]))

    # ----------------------------------------------------------------------
    # 7. Build Day-End Summary table  (instead of prints)
    # ----------------------------------------------------------------------
    summary_rows = []
    for bus, shed in cumulative_load_shedding.items():
        total = total_demand_per_bus.get(bus, {"p_mw": 0.0, "q_mvar": 0.0})
        summary_rows.append({
            "Bus": bus,
            "Load Shedding (MWh)":  shed["p_mw"],
            "Load Shedding (MVARh)":shed["q_mvar"],
            "Total Demand (MWh)":   total["p_mw"],
            "Total Demand (MVARh)": total["q_mvar"]
        })
    day_end_df = pd.DataFrame(summary_rows)

    # ----------------------------------------------------------------------
    # 8. Build Hourly Generation Cost table
    # ----------------------------------------------------------------------
    hourly_cost_df = pd.DataFrame({
        "Hour": list(range(len(business_as_usuall_cost))),
        "Current OPF Generation Cost (PKR)": business_as_usuall_cost
    })

    # ----------------------------------------------------------------------
    # 9. Return everything Colab returned *plus* the two DataFrames
    # ----------------------------------------------------------------------
    return (loading_percent_bau, served_load_per_hour, gen_per_hour_bau,
            slack_per_hour_bau, loading_records, business_as_usuall_cost,
            hourly_shed_bau, seen_buses, shedding_buses, df_lines, df_trafo,
            load_df, line_idx_map, trafo_idx_map, gdf,
            day_end_df, hourly_cost_df)









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

    if invalid_count > len(st.session_state.line_outages):
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
        
        
        
        
        center_point = ee.Geometry.Point(avg_lon, avg_lat)
        
        st.session_state.center_point = center_point
        
        roi = center_point.buffer(600000).bounds()
    
        st.session_state.roi = roi
        Map = geemap.Map()
        Map.centerObject(center_point, 5)
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

#def process_temperature(intensity, time_period, risk_score_threshold, df_line):
def process_temperature(intensity, time_period, risk_score_threshold, df_line,exposure_score):
                        # Temperature thresholds for intensity levels
                        thresholds = {"Low": 35, "Medium": 38, "High": 41}
                        thresholds_p = {"Low": 50, "Medium": 100, "High": 150}
                        thresholds_w = {"Low": 10, "Medium": 15, "High": 20}

                        if intensity not in thresholds or time_period not in ["Monthly", "Weekly"]:
                            raise ValueError("Invalid intensity or time period")

                        # Use the transmission line data from session state
                        df = df_line.copy()

                        from_buses = df["from_bus"].tolist()
                        to_buses = df["to_bus"].tolist()
                        all_lines = list(df[["from_bus", "to_bus"]].itertuples(index=False, name=None))

                        df["geodata"] = df["geodata"].apply(lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x)
                        line_geometries = [LineString(coords) for coords in df["geodata"]]
                        gdf = gpd.GeoDataFrame(df, geometry=line_geometries, crs="EPSG:4326")

                        # Create Folium map (instead of geemap.Map)
                        m = folium.Map(location=[30, 70], zoom_start=5, width=800, height=600)

                        # Define date range (last 10 years)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365 * 10)

                        # Select dataset based on time period
                        dataset_name = "ECMWF/ERA5/MONTHLY" if time_period == "Monthly" else "ECMWF/ERA5_LAND/DAILY_AGGR"
                        dataset = ee.ImageCollection(dataset_name).filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

                        dataset_forecast = ee.ImageCollection("NOAA/GFS0P25")
                        d = dataset_forecast.first()

                        # Create land mask
                        land_mask = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
                        land_mask = land_mask.map(lambda feature: feature.set("dummy", 1))
                        land_image = land_mask.reduceToImage(["dummy"], ee.Reducer.first()).gt(0)


                        # Convert transmission lines to FeatureCollection
                        features = [
                            ee.Feature(ee.Geometry.LineString(row["geodata"]), {
                                "line_id": i,
                                "geodata": str(row["geodata"])
                            }) for i, row in df.iterrows()
                        ]
                        line_fc = ee.FeatureCollection(features)

                        bounding_box = line_fc.geometry().bounds()

                        
                        combined_forecast = exposure_score
                        
                        combined_forecast = combined_forecast.updateMask(land_image)


                        # Reduce regions to get risk scores per line
                        reduced = combined_forecast.reduceRegions(
                            collection=line_fc,
                            reducer=ee.Reducer.max(),
                            scale=1000
                        )
                        daily_dfs = {}
                        results_per_day = []
                        max_times = []
                        
                        results = reduced.getInfo()
                        data = []
                        daily_results = []
                        risk_scores = []

                        for feature in results["features"]:
                            props = feature["properties"]
                            line_id = props["line_id"]
                            max_risk = props.get("max", 0)
                            from_bus = df.loc[line_id, "from_bus"]
                            to_bus = df.loc[line_id, "to_bus"]
                            daily_results.append((int(from_bus), int(to_bus), int(max_risk)))
                            risk_scores.append(int(max_risk))  # Add this line to collect risk scores
                            # if max_risk >= risk_score_threshold:
                            #     risk_scores.append(int(max_risk))
                            
                    
                            data.append({
                                "line_id": props["line_id"],
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                    
                            risk_scores.append({
                                "line_id": int(line_id),
                                "from_bus": int(from_bus),
                                "to_bus": int(to_bus),
                                "risk_score": int(max_risk)
                            })
                        results_per_day.append(daily_results)
                        daily_dfs["Day_1"] = pd.DataFrame(data)

                        # Filter lines with risk score >= threshold
                        day_1_results = results_per_day[0]
                        filtered_lines_day1 = [(from_bus, to_bus) for from_bus, to_bus, score in day_1_results if score >= risk_score_threshold]
                        length_lines = len(filtered_lines_day1)
                        outage_hour_day = [random.randint(11, 15) for _ in range(length_lines)]

                        # Create structured output for lines and outage hours
                        line_outages = [{"from_bus": from_bus, "to_bus": to_bus} for from_bus, to_bus in filtered_lines_day1]
                        outage_data = [{"line": f"From Bus {line[0]} to Bus {line[1]}", "outage_hours": hours, "risk_score": score}
                                      for line, hours, score in zip(filtered_lines_day1, outage_hour_day, [score for _, _, score in day_1_results if score >= risk_score_threshold])]

                        # Store in a format that can be used by other pages
                        line_outage_data = {
                            "lines": filtered_lines_day1,
                            "hours": outage_hour_day,
                            "risk_scores": risk_scores
                        }

                        return m, daily_dfs["Day_1"], line_outage_data, outage_data, None, None, None, risk_scores  # Update this line


def weather_opf(line_outages):
            # -----------------------------------------
            # 2. Initialization & data load
            # -----------------------------------------
            path = st.session_state.get("uploaded_file")      # BytesIO object
            xls  = pd.ExcelFile(path)                         # gives .sheet_names
            # ==== NEW – baseline cost frozen by Page 3 =========================
            business_as_usuall_cost = (
                st.session_state.bau_hourly_cost_df
                ["Current OPF Generation Cost (PKR)"]
                .tolist()
            )
            # make a working copy we can edit freely
            weather_aware_cost = business_as_usuall_cost.copy()
            # —— helper so existing single-arg calls still work ——
            def overloaded_transformer_local(net_):
                return overloaded_transformer(net_, path, line_outages)

            df_trafo = []
            if "Transformer Parameters" in xls.sheet_names:
                (net, df_bus, df_slack, df_line, num_hours,
                 load_dynamic, gen_dynamic,
                 df_load_profile, df_gen_profile,
                 df_trafo) = Network_initialize()
            else:
                (net, df_bus, df_slack, df_line, num_hours,
                 load_dynamic, gen_dynamic,
                 df_load_profile, df_gen_profile) = Network_initialize()
        
            # BAU and weather-aware cost arrays
            # business_as_usuall_cost = calculating_hourly_cost(path)
            weather_aware_cost      = business_as_usuall_cost.copy()
        
            # Build GeoDataFrame of lines
            df_lines = df_line.copy()
            df_lines["geodata"] = df_lines["geodata"].apply(
                lambda x: [(lon, lat) for lat, lon in eval(x)] if isinstance(x, str) else x
            )
            gdf = gpd.GeoDataFrame(
                df_lines,
                geometry=[LineString(coords) for coords in df_lines["geodata"]],
                crs="EPSG:4326",
            )
        
            # Bidirectional line & trafo index maps
            line_idx_map = {(r["from_bus"], r["to_bus"]): i for i, r in net.line.iterrows()}
            line_idx_map.update({(r["to_bus"], r["from_bus"]): i for i, r in net.line.iterrows()})
        
            trafo_idx_map = {}
            if "Transformer Parameters" in xls.sheet_names:
                trafo_idx_map = {(r["hv_bus"], r["lv_bus"]): i for i, r in net.trafo.iterrows()}
                trafo_idx_map.update({(r["lv_bus"], r["hv_bus"]): i for i, r in net.trafo.iterrows()})
        
            net.load["bus"] = net.load["bus"].astype(int)
        
            # Containers (names identical to Colab)
            loading_records           = [0] * num_hours
            shedding_buses            = []
            seen_buses                = set()
            served_load_per_hour_wa   = []
            loading_percent_wa        = []
            gen_per_hour_wa           = []
            slack_per_hour_wa         = []
            planned_slack_per_hour    = []
            hourly_shed_weather_aware = [0] * num_hours
        
            cumulative_load_shedding = {
                b: {"p_mw": 0.0, "q_mvar": 0.0} for b in net.load["bus"].unique()
            }
        
            # Total daily demand per bus (unchanged)
            total_demand_per_bus = {}
            p_cols = [c for c in df_load_profile.columns if c.startswith("p_mw_bus_")]
            q_cols = [c for c in df_load_profile.columns if c.startswith("q_mvar_bus_")]
            
            for bus in set(int(c.rsplit("_", 1)[1]) for c in p_cols):
                total_demand_per_bus[bus] = {
                    "p_mw": float(df_load_profile[f"p_mw_bus_{bus}"].sum()),
                    "q_mvar": float(df_load_profile[f"q_mvar_bus_{bus}"].sum()),
                }
        
            # ---------- fixed 20 % shedding per bus (same calc) -------------------
            # Before your hourly loop, record initial loads ---
            initial_load_p = {}   # real power
            initial_load_q = {}   # reactive
        
            for idx in net.load.index:
                bus = int(net.load.at[idx, "bus"])
                # capture whatever the initial profile set at that hour
                initial_load_p[bus] = net.load.at[idx, "p_mw"]
                initial_load_q[bus] = net.load.at[idx, "q_mvar"]
        
            # Precompute the fixed shedding per bus
            shed_pct = 0.10   # 0.05 --> 5% and 0.1 --> 10% Load Shedding
            fixed_shed_p = {bus: shed_pct * p for bus, p in initial_load_p.items()}
            fixed_shed_q = {bus: shed_pct * q for bus, q in initial_load_q.items()}
        
            # -----------------------------------------
            # 3. Hourly simulation: PF → conditional OPF → record
            # -----------------------------------------
            for hour in range(num_hours):
        
                # 3.1  scheduled outages
                for fbus, tbus, start_hr in line_outages:
                    if hour < start_hr:
                        continue
                    is_trafo = check_bus_pair(path, (fbus, tbus))
                    if is_trafo == True:
                        mask_tf = (((net.trafo.hv_bus == fbus) & (net.trafo.lv_bus == tbus)) |
                                ((net.trafo.hv_bus == tbus) & (net.trafo.lv_bus == fbus)))
                        if not mask_tf.any():
                            pass
                        else:
                            for tf_idx in net.trafo[mask_tf].index:
                                net.trafo.at[tf_idx, "in_service"] = False
                    else:
                        idx = line_idx_map.get((fbus, tbus))
                        if idx is not None:
                            net.line.at[idx, "in_service"] = False
        
                # 3.2  load profiles for this hour
                for idx in net.load.index:
                    b = net.load.at[idx, "bus"]
                    if b in load_dynamic:
                        net.load.at[idx, "p_mw"]   = df_load_profile.at[hour, load_dynamic[b]["p"]]
                        net.load.at[idx, "q_mvar"] = df_load_profile.at[hour, load_dynamic[b]["q"]]
        
                # update criticality each hour
                # crit_map = pd.read_excel(path, sheet_name="Load Parameters",
                #                          index_col=0)["criticality"].to_dict()
                # net.load["bus"] = net.load["bus"].astype(int)
                # net.load["criticality"] = net.load.bus.map(crit_map)

                df_load_params = pd.read_excel(path, sheet_name="Load Parameters", index_col=0)
                crit_map = dict(zip(df_load_params["bus"], df_load_params["criticality"]))
                net.load["bus"] = net.load["bus"].astype(int)
                net.load["criticality"] = net.load["bus"].map(crit_map)
        
                # 3.3  PV gen profile
                planned_gen_output = {}
                for idx in net.gen.index:
                    b = net.gen.at[idx, "bus"]
                    if b in gen_dynamic:
                        p = df_gen_profile.at[hour, gen_dynamic[b]]
                        net.gen.at[idx, "p_mw"] = p
                        planned_gen_output[idx] = p
        
                # 3.4  initial power-flow
                try:
                    pp.runpp(net)
                except:  # PF failed  → treat as overload
                    pass
        
                # record PF loading (for later plotting, if needed)
                # record this hour’s loading_percent Series -------------------
                        # ---------------- record PF loading -------------------------------
                intermediate_var = transform_loading(net.res_line["loading_percent"])
                if "Transformer Parameters" in xls.sheet_names:
                    intermediate_var.extend(
                        transform_loading(net.res_trafo["loading_percent"])
                    )
                loading_records[hour] = intermediate_var
        
                overloads        = overloaded_lines(net)
                overloads_trafo  = []
                if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
                    overloads_trafo = overloaded_transformer_local(net)
                all_loads_zero_flag = False
        
                # 3.5 Check for overloads
                if (
                    (overloads == [])
                    and (overloads_trafo == [])
                    and (all_real_numbers(loading_records[hour]) is True)
                ):
        
                    intermediate_cont = transform_loading(net.res_line["loading_percent"])
                    if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
                        intermediate_cont.extend(
                            transform_loading(net.res_trafo["loading_percent"])
                        )
                    loading_percent_wa.append(intermediate_cont)
        
                    slack_per_hour_wa.append(float(net.res_ext_grid.at[0, "p_mw"]))
        
                    if net.load["p_mw"].isnull().any():
                        served_load_per_hour_wa.append([None] * len(net.load))
                    else:
                        hourly_loads = net.load["p_mw"].tolist()
                        served_load_per_hour_wa.append(hourly_loads)
        
                    if net.res_gen["p_mw"].isnull().any():
                        gen_per_hour_wa.append([None] * len(net.res_gen))
                    else:
                        hourly_gen = net.res_gen["p_mw"].tolist()
                        gen_per_hour_wa.append(hourly_gen)
        
                    planned_slack_per_hour.append(float(net.res_ext_grid.at[0, "p_mw"]))
                    continue
        
                # 3.6 Record planned slack output
                planned_slack = {}
                if not net.ext_grid.empty:
                    for idx in net.ext_grid.index:
                        pw = "p_mw" if "p_mw" in net.res_ext_grid else "p_kw"
                        planned_slack[idx] = net.res_ext_grid.at[idx, pw]
                        planned_slack_per_hour.append(float(net.res_ext_grid.at[0, "p_mw"]))
        
                pf_loadings = transform_loading(net.res_line["loading_percent"])
                if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
                    pf_loadings.extend(transform_loading(net.res_trafo["loading_percent"]))
        
                try:
                    pp.runopp(net)
                    if (overloaded_lines(net) == []) and (overloaded_transformer_local(net) == []):
                        weather_aware_cost[hour] = net.res_cost
                        all_loads_zero_flag = True
                except Exception:
                    pass
        
                if (
                    all_real_numbers(
                        transform_loading(
                            net.res_line["loading_percent"] + net.res_trafo["loading_percent"]
                        )
                    )
                    and (overloaded_lines(net) == [])
                    and (overloaded_transformer_local(net) == [])
                ):
                    weather_aware_cost[hour] = net.res_cost
                else:
                    # 3.7 Run OPF to relieve overloads (fallback to shedding)
                    while (
                        ((overloaded_lines(net) != [])
                        or (overloaded_transformer_local(net) != [])
                    ) and (all_loads_zero_flag == False)):
        
                        for crit in sorted(
                            net.load["criticality"].dropna().unique(), reverse=True
                        ):
                            for ld_idx in net.load[net.load["criticality"] == crit].index:
                                if (overloaded_lines(net) == []) and (
                                    overloaded_transformer_local(net) == []
                                ):
                                    break
        
                                bus = net.load.at[ld_idx, "bus"]
                                dp = fixed_shed_p[bus]
                                dq = fixed_shed_q[bus]
        
                                net.load.at[ld_idx, "p_mw"] -= dp
                                net.load.at[ld_idx, "q_mvar"] -= dq
        
                                shedding_buses.append((hour, int(bus)))
                                cumulative_load_shedding[bus]["p_mw"] += dp
                                cumulative_load_shedding[bus]["q_mvar"] += dq
                                hourly_shed_weather_aware[hour] += dp
        
                                try:
                                    pp.runopp(net)
                                    weather_aware_cost[hour] = net.res_cost
                                    if net.OPF_converged:
                                        pf_loadings = transform_loading(
                                            net.res_line["loading_percent"]
                                        )
                                        if "Transformer Parameters" in pd.ExcelFile(
                                            path
                                        ).sheet_names:
                                            pf_loadings.extend(
                                                transform_loading(
                                                    net.res_trafo["loading_percent"]
                                                )
                                            )
                                        if all_real_numbers(pf_loadings):
                                            all_loads_zero_flag = True
                                except Exception:
                                    pp.runpp(net)
        
                                # collapse if load goes negative
                                if net.load.at[ld_idx, "p_mw"] - dp < 0:
                                    all_loads_zero_flag = True
                                    weather_aware_cost[hour] = 0
        
                                    remaining_p = net.load.loc[
                                        net.load["bus"] == bus, "p_mw"
                                    ].sum()
                                    remaining_q = net.load.loc[
                                        net.load["bus"] == bus, "q_mvar"
                                    ].sum()
                                    cumulative_load_shedding[bus]["p_mw"] += remaining_p
                                    cumulative_load_shedding[bus]["q_mvar"] += remaining_q
        
                                    hourly_shed_weather_aware[hour] = hourly_shed_weather_aware[hour] + sum(net.load['p_mw'])
                                    for i in range(len(net.load)):
                                        net.load.at[i, 'p_mw'] = 0
                                        net.load.at[i, 'q_mvar'] = 0
                                    break
        
                # 3.9 Record post-OPF loadings
                intermediate_var = transform_loading(net.res_line["loading_percent"])
                if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
                    intermediate_var.extend(
                        transform_loading(net.res_trafo["loading_percent"])
                    )
                loading_records[hour] = intermediate_var
        
                intermediate_cont = transform_loading(net.res_line["loading_percent"])
                if "Transformer Parameters" in pd.ExcelFile(path).sheet_names:
                    intermediate_cont.extend(
                        transform_loading(net.res_trafo["loading_percent"])
                    )
                loading_percent_wa.append(intermediate_cont)
        
                if net.load["p_mw"].isnull().any():
                    served_load_per_hour_wa.append([None] * len(net.load))
                else:
                    hourly_loads = net.load["p_mw"].tolist()
                    served_load_per_hour_wa.append(hourly_loads)
        
                if net.res_gen["p_mw"].isnull().any() or (weather_aware_cost[hour] == 0):
                    gen_per_hour_wa.append([None] * len(net.res_gen))
                    slack_per_hour_wa.append(None)
                else:
                    hourly_gen = net.res_gen["p_mw"].tolist()
                    gen_per_hour_wa.append(hourly_gen)
                    slack_per_hour_wa.append(float(net.res_ext_grid.at[0, "p_mw"]))
        
            # 4. Day-end summary tables (no prints)
            day_end_rows = []
            for bus, shed in cumulative_load_shedding.items():
                total = total_demand_per_bus.get(bus, {"p_mw": 0.0, "q_mvar": 0.0})
                day_end_rows.append(
                    {
                        "Bus": bus,
                        "Load Shedding (MWh)": shed["p_mw"],
                        "Load Shedding (MVARh)": shed["q_mvar"],
                        "Total Demand (MWh)": total["p_mw"],
                        "Total Demand (MVARh)": total["q_mvar"],
                    }
                )
            day_end_df = pd.DataFrame(day_end_rows)
        
            # hourly_cost_df = pd.DataFrame(
            #     {
            #         "hour": list(range(num_hours)),
            #         "generation_cost (PKR)": weather_aware_cost,
            #     }
            # )

            # --- NEW: add Current-OPF cost & the difference ------------------------------
            hourly_cost_df = pd.DataFrame(
                {
                    "Hour": list(range(num_hours)),
                    "Weather-Aware OPF Cost (PKR)":   weather_aware_cost,
                    "Current OPF Generation Cost (PKR)":         business_as_usuall_cost,
                }
            )
            hourly_cost_df["Δ Cost (WA – Current OPF)"] = (
                hourly_cost_df["Weather-Aware OPF Cost (PKR)"]
                - hourly_cost_df["Current OPF Generation Cost (PKR)"]
            )
        
            return (
                loading_records,
                shedding_buses,
                cumulative_load_shedding,
                hourly_shed_weather_aware,
                weather_aware_cost,
                seen_buses,
                hourly_shed_weather_aware,
                served_load_per_hour_wa,
                loading_percent_wa,
                gen_per_hour_wa,
                slack_per_hour_wa,
                planned_slack_per_hour,
                line_idx_map,
                trafo_idx_map,
                df_trafo,
                df_lines,
                df_load_params,
                day_end_df,
                hourly_cost_df,
            )
        # ─────────────────────────────────────────────────────────────────────────
