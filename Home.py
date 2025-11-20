import streamlit as st
import ee
import folium
import functions

st.set_page_config(
    page_title="Continuous Monitoring of Climate Risks to Electricity Grid using Google Earth Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.markdown("## ⚡ Grid Climate Dashboard")
    st.markdown("Monitor and optimize your power grid under climate stress.")
    st.markdown("---")
    
@st.cache_resource
def initialize_ee():
    ee.Authenticate()
    ee.Initialize(project='ee-ahmedalimj')
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


st.title("🌍 Continuous Monitoring of Climate Risks to Electricity Grid")
st.sidebar.success("Select a page from the sidebar to begin.")

st.markdown(
        """
        ### Overview  
        This web application gives an end to end decision support workflow for Grid Operators. It contains following five pages whose description is as follows:

        1. **Network Initialization** – This page ask user to input the Excel File containing Transmission Network Information.  
        2. **Weather‑Risk Visualisation** – This page ask user to set the Weather Analysis Parameters (see below for their details) and then utilize Google Earth Engine to analyze historic and forecasted weather data for day ahead.  
        3. **Projected Operation - Under Current OPF** – This page ask user to select contingency mode (see below for its details) and then yield 24 hourly electric grid operations along with the visualization on map for day ahead. This mode represents the usual operations of electric utilities where the generation does not account for historic weather data and projected extreme weather events that would cause transmissions lines to fail.  
        4. **Projected Operation - Under Weather Risk Aware OPF** – This page ask user to select contingency mode (see below for its details) and then yield 24 hourly operations along with the visualization on map for day ahead. This mode shows the vitality of our tool when it helps utilities to prepare the generation schedule for day ahead while incorporating historic and forecasted weather data and extreme weather risks to the electric grid. 
        5. **Data Analytics** – This page comprises of interactive comparative plots to show comparative analysis between the Projected Operations Under Current OPF vs Weather Risk Aware OPF in terms of cost, amount of load shedding, line loadings, estimated revenue loss under the “Projected Operation Under Current OPF” scenario and the hourly generation and load values. 
        
        The goal is to **quantify the technical and economic benefit** of risk aware dispatch decisions—highlighting potential lost revenue and critical load not served under various contingencies.

        **While an analysis is running, please remain on that page until it finishes. Once the process is complete, you’re free to navigate to any page and explore all options.**
        
        ---

        ### Want to learn more about our Web App?  
        * 📄 **Detailed Thesis** – [Google Drive (PDF)](https://drive.google.com/drive/folders/1mzGOuPhHn2UryrB2q5K4AZH2bPutvNhF?usp=drive_link)  
        * 📄 **Project Poster Link** - [Link](https://drive.google.com/drive/folders/1u2SVV-dwH7qRZuLKusNJBYcRhZsiodh9?usp=sharing)
        * ▶️ **Video Walk‑Through / Tutorial** – [YouTube](https://youtu.be/your-tutorial-video)  

        ---

        ### Key Terminologies

        1)	Weather Analysis Parameters: These are the three parameters set by grid operators.
            *	Risk Tolerance (Low, Medium, and High)
            *	Study Period (Weekly, Monthly)
            *	Risk Score Threshold (6-18)
        2)	Projected Operation Under Current OPF and Projected Operation Under Weather Risk Aware OPF has following options.
            *  Contingency Mode Selection

        ### Risk Tolerance
        
        * Low: In the Low option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are:                                                                   Temperature > 35°C, Precipitation > 50 mm, Wind > 10 m/s.
        
        * Medium: In the Medium option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are: Temperature > 38°C, Precipitation > 100 mm, Wind > 15 m/s.
        
        * High: In the High option, the following weather conditions are considered as thresholds beyond which the weather conditions would cause increased vulnerability to that specific region and a threat to electric network. The threshold values are: Temperature > 41°C, Precipitation > 150 mm, Wind > 20 m/s.
        
        We can also say that these parameters would be based on how resilient an input network is. With Low means the network is least resilient and high means that network is strong against the extreme weather events.

        ### Study Period
        
        * Weekly: Under this option the tool will use weekly weather information (weekly aggregated data) for the historic weather analysis.
        
        * Monthly: Under this option the tool will use monthly weather information (monthly aggregated data) for the historic weather analysis.

        ### Risk Score Threshold
        
        * Risk Score can be chosen on a scale of 6-18 which is important for post weather data analysis. Using our novel Risk scoring Algorithm, when the risk scores are generated for each transmission lines for day ahead, this parameter decides which lines would fail on which projected hour during upcoming extreme weather event.

        ### Contingency Mode Selection:
        
        The Contingency Mode parameter allows the user to define the operational scope of the system’s vulnerability simulation by selecting between two distinct failure modeling strategies. This choice directly impacts the number of lines that would be down after risk scores have been computed for all transmission lines.
        
        * Capped Contingency Mode: This mode evaluates system stability under a constrained failure scenario, assuming that only 20% of the at-risk transmission lines (as identified by the risk score threshold) of the total lines will fail. Any additional forecasted failures beyond this cap are deprioritized, reflecting conservative grid planning under limited disruption assumptions.
        
        * Maximum Contingency Mode: In contrast, this mode simulates a worst-case scenario by assuming that all transmission lines flagged as high risk will fail. It supports comprehensive stress-testing of the network, providing insights into cascading failure risks, load redistribution behavior, and potential stability violations under extreme weather-induced conditions

         ---
         
        ### Key Features
        * **Google Earth Engine Integration** is utilized for having rich historic weather data as well as forecasted weather data. 
        * **Pandapower** is utilized for performing Optimal Power Flow (OPF) Analysis for finding optimized generation dispatch and calculation of lost load. 
        * **GEE, Folium based maps and Plotly analytics** are used for hourly visualization in both scenarios and interactive plots in comparative analysis.

        ---

        ### Usage Workflow
        1. Navigate left‑hand sidebar → **Network Initialization** and upload your Excel model.  
        2. Tune thresholds on **Weather Risk Visualisation** and press *Process*.  
        3. Run **Projected Operation - Under Current OPF** → then **Projected Operation Under Weather Risk Aware OPF**.  
        4. Explore comparative plots in **Data Analytics**.  

        *(You can re‑run any page; session‑state keeps everything consistent.)*

        ---

        ### Data Sources 

        This tool has utilized Google Earth Engine (GEE) which is a cloud-based platform designed for large scale analysis of geospatial data. Its three data sets have been utilized in this tool
        
        * ERA‑5 The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY)
        
        * ERA 5 Land reanalysis: The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)

        * NOAA GFS forecasts: The following dataset is utilized to get hourly weather forecast [Link](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25)

        ---

        ### Authors & Contact  
        * **Muhammad Hasan Khan** – BSc Electrical Engineering, Habib University  
          * ✉️ iamhasan710@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](www.linkedin.com/in/hasankhan710)  
        * **Munim ul Haq** – BSc Electrical Engineering, Habib University  
          * ✉️ themunimulhaq24@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/munim-ul-haq/) 
        * **Syed Muhammad Ammar Ali Jaffri** – BSc Electrical Engineering, Habib University  
          * ✉️ ammarjaffri6515@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/ammarjaffri/) 

        ### Faculty Supervisor  
        * **Muhammad Umer Tariq** – Assistant Professor, Electrical and Computer Engineering at Habib University  
          * ✉️ umer.tariq@sse.habib.edu.pk  

        _We welcome feedback, and collaboration enquiries._
        """,
        unsafe_allow_html=True
    )