# Weather-Driven Grid Risk Planner  
An end-to-end **decision-support workflow** for grid operators, integrating weather analytics, power flow simulation, and risk scoring.

---

## 🌐 Overview
This web application provides a complete workflow for assessing transmission network risk under historical and forecasted weather conditions. It is organized into **four pages**, each building on the previous one:

1. **Network Initialization**  
   Upload an Excel file containing transmission network information.

2. **Historical Weather Exposure**  
   Configure weather analysis parameters and utilize Google Earth Engine (GEE) to study historic weather data for day-ahead risk.

3. **Combined Exposure**  
   Set temperature and rainfall parameters for both historical and forecast data. A unified risk score is generated and displayed through an interactive map overlay.

4. **Day-Ahead Risk Planning**  
   Select a risk threshold to determine potential line outages. Pandapower performs power-flow analysis to estimate possible overloads based on weather-driven failures.

> **Note:**  
> • Please stay on the page while an analysis is running.  
> • Do not skip pages — each step carries results forward to the next.

---

## ▶️ Want to Learn More?
📹 **Video Walk-Through / Tutorial:**  
[YouTube](https://youtu.be/your-tutorial-video)

---

## ⭐ Key Features
- **Google Earth Engine Integration**  
  Access rich historical and forecasted weather datasets.

- **Pandapower OPF**  
  Perform Optimal Power Flow (OPF) for generation dispatch and lost-load calculations.

- **Interactive Visualizations**  
  Hourly maps (GEE, Folium), and analytical plots (Plotly) for scenario comparison.

---

## 🧭 Usage Workflow
1. Go to **Network Initialization** → Upload your Excel model.  
2. Adjust sliders in **Historical Weather Exposure** and **Combined Exposure**. Maps auto-refresh.  
3. Run **Day-Ahead Risk Planning** → Click **Run Power Flow Analysis**.  

*(All pages can be re-run; session-state ensures consistency.)*

---

## 🗂️ Data Sources
This tool uses datasets from **Google Earth Engine**:

- **ERA-5 Daily**  
  Historic climate analysis  
  <https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_DAILY>

- **ERA-5 Land Monthly**  
  Land-focused reanalysis  
  <https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY>

- **NOAA GFS Forecasts**  
  Hourly weather forecasting  
  <https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25>

---

## 👥 Authors & Contact

### Lead Author
**Ahmed Ali Mustansir**  
Dean's Fellow, Electrical & Computer Engineering, Habib University  
📧 **ahmedali.mustansir@sse.habib.edu.pk**

### Faculty Supervisor
**Muhammad Umer Tariq**  
Assistant Professor, Electrical & Computer Engineering, Habib University  
📧 **umer.tariq@sse.habib.edu.pk**

---

## 🙌 Acknowledgements
This project is inspired by and extends the work by:

- **Muhammad Hasan Khan** – BSc Electrical Engineering  
  📧 iamhasan710@gmail.com |  
  [LinkedIn](https://www.linkedin.com/in/hasankhan710)

- **Munim ul Haq** – BSc Electrical Engineering  
  📧 themunimulhaq24@gmail.com |  
  [LinkedIn](https://www.linkedin.com/in/munim-ul-haq/)

- **Syed Muhammad Ammar Ali Jaffri** – BSc Electrical Engineering  
  📧 ammarjaffri6515@gmail.com |  
  [LinkedIn](https://www.linkedin.com/in/ammarjaffri/)

📄 **Detailed Thesis (PDF):**  
<https://drive.google.com/drive/folders/1mzGOuPhHn2UryrB2q5K4AZH2bPutvNhF?usp=drive_link>

---

_We welcome feedback and collaboration inquiries._
