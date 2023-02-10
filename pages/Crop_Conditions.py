from datetime import datetime as dt
import streamlit as st
import QuickStats as qs
import func as fu

# Declarations
if 'crop_conditions' not in st.session_state:
    st.session_state['crop_conditions']={}  

st.set_page_config(page_title='Crop Conditions',layout='wide',initial_sidebar_state='expanded')
st.markdown("### Crop Conditions")
st.markdown("---")

# Events
def on_change_commodity():
    if 'crop_conditions' in st.session_state:
        del st.session_state['crop_conditions']


with st.sidebar:
    crop_year_start=dt(dt.today().year,1,1)
    commodity = st.selectbox("Commodity", fu.commodities, 0, on_change=on_change_commodity)

    if commodity in st.session_state['crop_conditions']:
        options_states=st.session_state['crop_conditions'][commodity]['options_states']
    else:
        st.session_state['crop_conditions'][commodity]={}
        st.session_state['crop_conditions'][commodity]['options_states']=None
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=None
        st.session_state['crop_conditions'][commodity]['dfs_yields']=None

        with st.spinner('Checking Available States...'):
            options_states=qs.get_USA_conditions_states(commodity)
            st.session_state['crop_conditions'][commodity]['options_states']=options_states

    
    # Commodity customization
    # if commodity != 'WHEAT, SPRING, DURUM'.title():
    #     options_states=['US Total'] + options_states # Because USDA doesn' provide National numbers for 'WHEAT, SPRING, DURUM'

    # if commodity == 'WHEAT, WINTER'.title():        
    #     crop_year_start=dt(dt.today().year,9,1)

    state = st.selectbox("State", options_states)
    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)


if True: # (st.session_state['crop_conditions'][commodity]['dfs_conditions'] is None):
    # st.write('From USDA')

    selected=[state]
    # selected=options_states # when finished
    
    dfs_conditions=qs.get_USA_conditions_parallel(commodity.upper(), state_name=selected)
    dfs_yields=qs.get_USA_yields_parallel(commodity.upper(), state_name=selected)
    
    # st.session_state['crop_conditions'][commodity]['dfs_conditions']=dfs_conditions
    # st.session_state['crop_conditions'][commodity]['dfs_yields']=dfs_yields

    # st.write(dfs_yields['Illinois'])
    # st.write(qs.get_USA_prod_weights(commodity))
else:
    print('WIP')
    # st.write('In memory')
    # dfs_conditions=st.session_state['crop_conditions'][commodity]['dfs_conditions']
    # dfs_yields=st.session_state['crop_conditions'][commodity]['dfs_yields']

if True:
    # df = fu.get_GE_conditions(state, commodity, crop_year_start=crop_year_start)
    # st.plotly_chart(fu.get_conditions_chart(df, state, commodity, hovermode=hovermode), use_container_width=True)

    # Extending the df so to have all the years in the scatter plot
    # df = df.pivot(index='seas_day',columns='year',values='Value').fillna(method='ffill').fillna(method='bfill').melt(ignore_index=False)
    # df['seas_day']=df.index
    # df=df.rename(columns={'value':'Value'})

    # Extract the Good and Excellent
    dfs_GE={}
    for state, df in dfs_conditions.items():
        dfs_GE[state]=qs.extract_GE_conditions(df)
        # st.write(dfs_GE[state])
        st.plotly_chart(fu.get_conditions_chart(dfs_GE[state], state, commodity, hovermode=hovermode), use_container_width=True)

    all_fig_model_chart = fu.get_CCI_yield_model_charts(dfs_GE, dfs_yields, hovermode=hovermode)

    for f in all_fig_model_chart:
        st.markdown("---")
        st.plotly_chart(f['fig'], use_container_width=True)

        with st.expander('Model Details'):
            st.write(f['model'].summary())
