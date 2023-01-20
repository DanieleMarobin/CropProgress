from datetime import datetime as dt
import streamlit as st
import QuickStats as qs
import func as fu

if 'crop_conditions' not in st.session_state:
    st.session_state['crop_conditions']={}

st.set_page_config(page_title='Crop Conditions',layout='wide',initial_sidebar_state='expanded')
st.markdown("### Crop Conditions")
st.markdown("---")

with st.sidebar:
    crop_year_start=dt(dt.today().year,1,1)
    commodity = st.selectbox("Commodity", fu.commodities, 0)

    if commodity in st.session_state['crop_conditions']:
        options_states=st.session_state['crop_conditions'][commodity]['options_states']
    else:
        with st.spinner('Checking Available States...'):
            options_states=qs.get_USA_conditions_states(commodity)
            st.session_state['crop_conditions'][commodity]={'options_states':options_states}
    
    # Commodity customization
    if commodity != 'WHEAT, SPRING, DURUM'.title():
        options_states=['US Total'] + options_states
    if commodity == 'WHEAT, WINTER'.title():        
        crop_year_start=dt(dt.today().year,9,1)

    state = st.selectbox("State", options_states)

    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)


df = fu.get_conditions(state, commodity, crop_year_start=crop_year_start)
st.plotly_chart(fu.get_conditions_chart(df, state, commodity, hovermode=hovermode), use_container_width=True)

# Extending the df so to have all the years in the scatter plot
df = df.pivot(index='seas_day',columns='year',values='Value').fillna(method='ffill').fillna(method='bfill').melt(ignore_index=False)
df['seas_day']=df.index
df=df.rename(columns={'value':'Value'})
all_fig_model_chart = fu.get_CCI_yield_model_charts(df, state, commodity, hovermode=hovermode)

for f in all_fig_model_chart:
    st.markdown("---")
    st.plotly_chart(f['fig'], use_container_width=True)

    with st.expander('Model Details'):
        st.write(f['model'].summary())
