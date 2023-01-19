import streamlit as st
import func as rs

st.set_page_config(page_title='Crop Conditions',layout='wide',initial_sidebar_state='expanded')
st.markdown("### Crop Conditions")
st.markdown("---")

with st.sidebar:
    col11, col21 = st.columns(2)
    add_grain = st.selectbox("Choose a Class", rs.grains_class,2)
    if add_grain == 'WINTER'.title():
        add_region = st.selectbox("Choose a Region", rs.wwht_regions)
    elif add_grain == 'SPRING, (EXCL DURUM)'.title():
        add_region = st.selectbox("Choose a Region", rs.swht_regions)
    elif add_grain == 'SPRING, DURUM'.title():
        add_region = st.selectbox("Choose a Region", rs.durum_regions)
    
    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)

df = rs.get_conditions(add_region, add_grain)
st.plotly_chart(rs.get_conditions_chart(df, add_region, add_grain, hovermode=hovermode), use_container_width=True)

fig_model = rs.get_yields_charts(df, add_region, add_grain, hovermode=hovermode)

for f in fig_model:
    st.markdown("---")
    st.plotly_chart(f['fig'], use_container_width=True)

    with st.expander('Model Details'):
        st.write(f['model'].summary())
