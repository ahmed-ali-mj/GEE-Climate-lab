import streamlit as st
from geemap import Map

import ee
ee.Authenticate()

ee.Initialize(project='ee-ahmedalimj')

st.title("geemap test")

m = Map()
m.to_streamlit(width=700, height=500)
