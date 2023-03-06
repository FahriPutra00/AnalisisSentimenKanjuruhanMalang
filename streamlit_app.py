import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.title("My Streamlit App")

# Create a slider
x = st.slider('Select a value for x', 0, 10)

# Create a chart
df = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
fig = px.scatter(df, x='a', y='b')
st.plotly_chart(fig)
