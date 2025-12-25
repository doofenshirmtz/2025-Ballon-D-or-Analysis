import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("⚽ Ballon d’Or 2025 – who did the numbers say should win?")

csv = st.file_uploader("Upload Player_Stats_All_Competitions.csv", type="csv")
if csv is None:
    st.stop()

full_df = pd.read_csv(csv)
names_col = full_df['Player']
cols = ['Goals','Assists','xG','xA']
df = full_df[cols].dropna()

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(df, np.arange(len(df)))

df['model_score'] = model.predict(df)
df['model_rank']  = df['model_score'].rank(method='min', ascending=False).astype(int)
df['Player']      = names_col.values

st.subheader("Model top 10")
st.dataframe(df[['Player','model_rank']].sort_values('model_rank').head(10))
