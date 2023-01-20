from datetime import datetime as dt
import streamlit as st

import func as fu
import QuickStats as qs

if 'crop_progress' not in st.session_state:
    st.session_state['crop_progress']={}

st.set_page_config(page_title='Crop Progress',layout='wide',initial_sidebar_state='expanded')
st.markdown("### Crop Progress")
st.markdown("---")


with st.sidebar:
    crop_year_start=dt(dt.today().year,1,1)
    commodity = st.selectbox("Commodity", fu.commodities, 0)

    if ((commodity in st.session_state['crop_progress']) and ('options_states' in st.session_state['crop_progress'][commodity])):
        options_states=st.session_state['crop_progress'][commodity]['options_states']
    else:
        with st.spinner('Checking Available States...'):
            options_states=qs.get_USA_progress_states(commodity)
            st.session_state['crop_progress'][commodity]={'options_states':options_states}


    if ((commodity in st.session_state['crop_progress']) and ('options_variables' in st.session_state['crop_progress'][commodity])):
        options_variables=st.session_state['crop_progress'][commodity]['options_variables']
    else:   
        with st.spinner('Checking Available Progress Variables...'):
            options_variables=qs.get_USA_progress_variables(commodity)
            st.session_state['crop_progress'][commodity]['options_variables']=options_variables


    # Commodity customization
    if commodity != 'WHEAT, SPRING, DURUM'.title():
        options_states=['US Total'] + options_states
    if commodity == 'WHEAT, WINTER'.title():        
        crop_year_start=dt(dt.today().year,9,1)

    
    state = st.selectbox("State", options_states)
    progress_variables = st.multiselect("Variable", options_variables)

    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)


for progress_var in progress_variables:
    st.plotly_chart(fu.get_progress_chart(commodity,state,progress_var, crop_year_start=crop_year_start, hovermode=hovermode), use_container_width=True)