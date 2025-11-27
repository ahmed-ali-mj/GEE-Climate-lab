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


st.title("🌍 Integrated Weather Exposure & Grid Risk Planner")
st.sidebar.success("Select a page from the sidebar to begin.")

st.markdown(
        """
        ### Overview  
        This web application gives an end to end decision support workflow for Grid Operators. It contains following four pages whose description is as follows:

        1. **Network Initialization** – This page asks user to input the Excel File containing Transmission Network Information.  
        2. **Historical weather exposure** – This page asks user to set the Weather Analysis Parameters and then utilize Google Earth Engine to analyze historic weather data for day ahead.  
        3. **Combined exposure** – This page asks the user to set parameters for temperature and rainfall for both, historic and forecast weather data. All parameters can be adjusted to calculate a single risk score. The results are displayed in a map overlay. These same parameters are used in the next page. 
        4. **Day ahead risk planning** – This page asks the user to set risk level threshold based on which the program decides wether a line fails. Power flow analysis calculates and displays results for potential overloadings based on weather data.
        **While an analysis is running, please remain on that page until it finishes. Once the process is complete, you’re free to navigate to any page and explore all options.**\n
        <p>Please do not skip any page as results from each page are carried over to the next.</p>
        ---

        ### Want to learn more about our Web App?
        * ▶️ **Video Walk‑Through / Tutorial** – [YouTube](https://youtu.be/your-tutorial-video)  

        ---


         
        ### Key Features
        * **Google Earth Engine Integration** is utilized for having rich historic weather data as well as forecasted weather data. 
        * **Pandapower** is utilized for performing Optimal Power Flow (OPF) Analysis for finding optimized generation dispatch and calculation of lost load. 
        * **GEE, Folium based maps and Plotly analytics** are used for hourly visualization in both scenarios and interactive plots in comparative analysis.

        ---

        ### Usage Workflow
        1. Navigate left‑hand sidebar → **Network Initialization** and upload your Excel model.  
        2. Tune thresholds on **Historical weather exposure** and **Combined exposure**. There is no need to press anything, the map auto refreshes when any slider is moved.  
        3. Run **Day ahead risk planning** by pressing *Run Power flow analysis*. 


        *(You can re‑run any page; session‑state keeps everything consistent.)*

        ---

        ### Data Sources 

        This tool has utilized Google Earth Engine (GEE) which is a cloud-based platform designed for large scale analysis of geospatial data. Its three data sets have been utilized in this tool
        
        * ERA‑5 The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY)
        
        * ERA 5 Land reanalysis: The following dataset is utilized for historic weather analysis [Link](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY)

        * NOAA GFS forecasts: The following dataset is utilized to get hourly weather forecast [Link](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25)

        ---

        ### Authors & Contact  
        * **Ahmed Ali Mustansir** - Dean's fellow, Electrical and Computer Engineering at Habib University  
          * ✉️ ahmedali.mustansir@sse.habib.edu.pk  
        ### Faculty Supervisor  
        * **Muhammad Umer Tariq** – Assistant Professor, Electrical and Computer Engineering at Habib University  
          * ✉️ umer.tariq@sse.habib.edu.pk  
        ### This project is heavily inspired by the work done by the following students
        * **Muhammad Hasan Khan** – BSc Electrical Engineering, Habib University  
          * ✉️ iamhasan710@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](www.linkedin.com/in/hasankhan710)  
        * **Munim ul Haq** – BSc Electrical Engineering, Habib University  
          * ✉️ themunimulhaq24@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/munim-ul-haq/) 
        * **Syed Muhammad Ammar Ali Jaffri** – BSc Electrical Engineering, Habib University  
          * ✉️ ammarjaffri6515@gmail.com&nbsp;&nbsp;|&nbsp;&nbsp;[LinkedIn](https://www.linkedin.com/in/ammarjaffri/) 
        * 📄 **Detailed Thesis** – [Google Drive (PDF)](https://drive.google.com/drive/folders/1mzGOuPhHn2UryrB2q5K4AZH2bPutvNhF?usp=drive_link)  
        
        _We welcome feedback, and collaboration enquiries._
        """,
        unsafe_allow_html=True
    )