# ────────────────────────────────────────────────────────────────────────────
# Page 4 :  Optimal power flow
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
import functions
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import folium
import ast
from streamlit_folium import st_folium

st.header('Day ahead risk planning')

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


#mode = st.selectbox("Select Contingency Mode",
#                ["Capped Contingency Mode (20% of Network Lines)",
#                 "Maximum Contingency Mode (All Outage Lines)"])
cap_flag = False

if st.button("Run Power flow Analysis"):
    forecast_range = st.session_state["forecast_range"]
    line_outage_data = [None] * forecast_range
    outage_data = [None] * forecast_range
    risk_scores = [None] * forecast_range
    
    with st.spinner("Processing weather data and calculating line outages...."):
        
    
        for x in range(forecast_range):
            line_outage_data[x], outage_data[x], risk_scores[x] = \
            functions.process_temperature(
                        risk_threshold,
                        st.session_state.network_data['df_line'],
                        st.session_state["exposure_score"][x]
                        )
        
        # Store the map and data in session state
        # st.session_state.weather_map_obj = weather_map
        st.session_state.line_outage_data = line_outage_data
        # st.session_state["outage_hours"] = line_outage_data["hours"]
        # st.session_state["line_down"]    = line_outage_data["lines"]
        # st.session_state["risk_scores"]  = line_outage_data["risk_scores"]
        # st.session_state.risk_df2 = risk_df
        st.session_state.outage_data = outage_data
        st.session_state.risk_score = risk_scores

        # build the outage list first
        line_outages = [None] * forecast_range
        for x in range(forecast_range):
            line_outages[x] = functions.generate_line_outages(
                outage_hours   = st.session_state["line_outage_data"][x]["hours"],
                line_down      = st.session_state["line_outage_data"][x]["lines"],
                risk_scores    = st.session_state["line_outage_data"][x]["risk_scores"],
                capped_contingency_mode = cap_flag
            )
        st.session_state.line_outages = line_outages
    
    
    
    # store globally for helper functions
    globals()["line_outages"] = line_outages
    
    with st.spinner("Running Power Flow Analysis (Estimated Time 5-10 minutes)..."):
        (_lp_bau, _served, _gen, _slack, _rec, _cost,
         _shed, _seen, _shed_buses, _df_lines, _df_trafo,
         _load_df, _line_idx_map, _trafo_idx_map, _gdf,
         day_end_df, hourly_cost_df,hourly_line_data,isolated_loads) = functions.current_opf(line_outages)

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
    st.session_state["hourly_line_data"] = hourly_line_data
    st.session_state["isolated_loads"] = isolated_loads
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
    outages      = st.session_state.line_outages[hr]

    # ── helper colour fns (same logic) ─────────────────────
    def get_color(pct, max_cap):
        if pct is None:                return '#0000FF'
        if pct == 0:                   return '#0000FF'
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
    
    st.dataframe(st.session_state["hourly_line_data"])
    st.write(st.session_state["isolated_loads"])
    st.dataframe(pd.DataFrame(st.session_state["risk_score"]))
