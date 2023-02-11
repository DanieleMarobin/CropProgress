from datetime import datetime as dt
import pandas as pd
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
    commodity = st.selectbox("Commodity", fu.commodities, 2, on_change=on_change_commodity)

    if commodity in st.session_state['crop_conditions']:
        options_states=st.session_state['crop_conditions'][commodity]['options_states']
    else:
        st.session_state['crop_conditions'][commodity]={}
        st.session_state['crop_conditions'][commodity]['options_states']=None
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=None
        st.session_state['crop_conditions'][commodity]['dfs_yields']=None
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


if True or (st.session_state['crop_conditions'][commodity]['dfs_conditions'] is None):
    st.write('From USDA')

    selected=[state]
    # selected=options_states # for us total
    
    dfs_conditions=qs.get_USA_conditions_parallel(commodity.upper(), state_name=selected)
    dfs_yields=qs.get_USA_yields_parallel(commodity.upper(), state_name=selected)
    df_prod_weights=qs.get_USA_prod_weights(commodity)
    
    st.session_state['crop_conditions'][commodity]['dfs_conditions']=dfs_conditions
    st.session_state['crop_conditions'][commodity]['dfs_yields']=dfs_yields
    st.session_state['crop_conditions'][commodity]['df_prod_weights']=df_prod_weights

    # st.write(qs.get_USA_prod_weights(commodity))
else:
    st.write('In memory')
    dfs_conditions=st.session_state['crop_conditions'][commodity]['dfs_conditions']
    dfs_yields=st.session_state['crop_conditions'][commodity]['dfs_yields']
    df_prod_weights=st.session_state['crop_conditions'][commodity]['df_prod_weights']

# Default
if True:
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
        
        with st.expander(f['state'] + ' - Model Details'):
            st.dataframe(f['df'])
            st.write(f['model'].summary())

# Test all US Calcs
if False:
    # Extract the Good and Excellent
    dfs_GE={}
    for state, df in dfs_conditions.items():
        dfs_GE[state]=qs.extract_GE_conditions(df)

    all_fig_model_chart = fu.get_CCI_yield_model_charts(dfs_GE, dfs_yields, hovermode=hovermode)

    dfs={}
    for f in all_fig_model_chart:
        dfs[f['state']]=f['df'][['Conditions']].rename(columns={'Conditions':f['state'].upper()})

    us_total_conditions=pd.concat(list(dfs.values()),axis=1,join='outer')
    us_total_conditions=us_total_conditions.sort_index()
    st.write('us_total_conditions')
    st.write(us_total_conditions)

    last_year=us_total_conditions.last_valid_index()

    if last_year not in df_prod_weights.index:
        mask=df_prod_weights.index>=last_year-5
        weight_mean=df_prod_weights.mean()
        df_prod_weights.loc[last_year]=weight_mean/weight_mean.sum()

    st.write('df_prod_weights')
    st.write(df_prod_weights)
    # st.write(df_prod_weights.sum(axis=1))

    us_total_weight_cond=df_prod_weights * us_total_conditions
    st.write('us_weight_cond')
    us_total_weight_cond['us_total_conditions']=us_total_weight_cond.sum(axis=1)
    st.write(us_total_weight_cond)

    us_total_yield= qs.get_USA_yields(commodity, aggregate_level='NATIONAL')[['year','Value']]
    us_total_yield=us_total_yield.rename(columns={'Value':'us_total_yield'})
    st.write(us_total_yield)

    df_us_total=pd.concat([us_total_weight_cond['us_total_conditions'],us_total_yield['us_total_yield']], axis=1)
    mask=(df_us_total['us_total_conditions']>0) # this to drop the 0s 
    df_us_total=df_us_total[mask]
    st.write(df_us_total)

    us_total_fig_model_chart=fu.get_CCI_us_total(df_us_total,hovermode)

    for f in us_total_fig_model_chart:
        st.markdown("---")
        st.plotly_chart(f['fig'], use_container_width=True)
        
        with st.expander(f['state'] + ' - Model Details'):
            st.dataframe(f['df'])
            st.write(f['model'].summary())
    