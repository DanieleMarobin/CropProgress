from datetime import datetime as dt
import pandas as pd
import streamlit as st
import QuickStats as qs
import func as fu

# Declarations and preliminaries
if True:
    if 'crop_conditions' not in st.session_state:
        st.session_state['crop_conditions']={}  

    st.set_page_config(page_title='Crop Conditions',layout='wide',initial_sidebar_state='expanded')
    st.markdown("### Crop Conditions")
    st.markdown("---")

    # Events
    def on_change_commodity():
        if 'crop_conditions' in st.session_state:
            del st.session_state['crop_conditions']

# Commodity / State selection
with st.sidebar:
    commodity = st.selectbox("Commodity", fu.commodities, 2, on_change=on_change_commodity)

    if commodity in st.session_state['crop_conditions']:
        options_states=st.session_state['crop_conditions'][commodity]['options_states']
    else:
        st.session_state['crop_conditions'][commodity]={}
        st.session_state['crop_conditions'][commodity]['options_states']=None
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=None
        st.session_state['crop_conditions'][commodity]['df_yields']=None

        with st.spinner('Checking Available States...'):
            options_states=qs.get_USA_conditions_states(commodity)
            st.session_state['crop_conditions'][commodity]['options_states']=options_states
    
    # Commodity customization
    if commodity != 'WHEAT, SPRING, DURUM'.title():
        options_states=['US Total'] + options_states # Because USDA doesn' provide National numbers for 'WHEAT, SPRING, DURUM'

    state = st.selectbox("State", options_states)
    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)

    TOTAL_US_DM = st.checkbox('Avere US Total', False, on_change=on_change_commodity)                    

# Retrieve the data
if False or (st.session_state['crop_conditions'][commodity]['dfs_conditions'] is None):
    st.write('From USDA')

    if TOTAL_US_DM:
        if 'US Total' in options_states:
            options_states.remove('US Total')

        selected=options_states # for us total
        dfs_conditions=qs.get_USA_conditions_parallel(commodity.upper(), state_name=selected)
        df_yields=qs.get_USA_yields_weights(commodity.upper(), aggregate_level='STATE', state_name=selected,output='value')
        
        df_prod_weights=qs.get_USA_prod_weights(commodity)
        df_plant=qs.get_USA_area_planted_weights(commodity, output='value')
        df_harv=qs.get_USA_area_harvested_weights(commodity, output='value')
        
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=dfs_conditions
        st.session_state['crop_conditions'][commodity]['df_yields']=df_yields
        st.session_state['crop_conditions'][commodity]['df_prod_weights']=df_prod_weights
        st.session_state['crop_conditions'][commodity]['df_plant']=df_plant
        st.session_state['crop_conditions'][commodity]['df_harv']=df_harv        
    else:
        selected=[state]
        
        if selected[0]=='US Total':
            dfs_conditions=qs.get_USA_conditions_parallel(commodity.upper(),aggregate_level='NATIONAL', state_name=selected)
            df_yields=qs.get_USA_yields_weights(commodity.upper(), aggregate_level='NATIONAL', state_name=selected,output='value')
        else:
            dfs_conditions=qs.get_USA_conditions_parallel(commodity.upper(), state_name=selected)
            df_yields=qs.get_USA_yields_weights(commodity.upper(), aggregate_level='STATE', state_name=selected,output='value')
else:
    st.write('From cache')
    dfs_conditions=st.session_state['crop_conditions'][commodity]['dfs_conditions']
    df_yields=st.session_state['crop_conditions'][commodity]['df_yields']
    df_prod_weights=st.session_state['crop_conditions'][commodity]['df_prod_weights']
    df_plant=st.session_state['crop_conditions'][commodity]['df_plant']
    df_harv=st.session_state['crop_conditions'][commodity]['df_harv']

# From USDA
if not TOTAL_US_DM:
    # Extract the Good and Excellent
    dfs_GE={}
    
    for state, df in dfs_conditions.items():
        dfs_GE[state]=qs.extract_GE_conditions(df)
        st.plotly_chart(fu.get_conditions_chart(dfs_GE[state], state, commodity, hovermode=hovermode), use_container_width=True)

    states_results = fu.get_CCI_results_by_state(dfs_GE, df_yields, hovermode=hovermode)

    for f in states_results:
        st.markdown("---")
        st.plotly_chart(f['fig'], use_container_width=True)
        
        with st.expander(f['state'] + ' - Model Details'):
            st.dataframe(f['df'])
            st.write(f['model'].summary())

# 'Home made' US total calculation
if TOTAL_US_DM:
    df_harv_pct=df_harv/df_plant
    # Extract the Good and Excellent
    dfs_GE={}
    for state, df in dfs_conditions.items():
        dfs_GE[state]=qs.extract_GE_conditions(df)

    # cci_results = {'state':state,'analysis':analysis, 'fig':fig,'model':model, 'df':df, 'prediction':prediction}
    cci_results = fu.get_CCI_results_by_state(dfs_GE, df_yields, hovermode=hovermode)

    # concatenating together all the state-by-state conditions
    dfs_cond={}
    dfs_pred={}
    for f in cci_results:
        if f['analysis']=='delta':
            dfs_cond[f['state']]=f['df'][['Conditions']].rename(columns={'Conditions':f['state'].upper()})
            dfs_pred[f['state'].upper()]=f['prediction']    

    us_total_conditions=pd.concat(list(dfs_cond.values()),axis=1,join='outer')
    us_total_conditions=us_total_conditions.sort_index()
    last_year=us_total_conditions.last_valid_index()
    # st.write('us_total_conditions',us_total_conditions.sort_index(ascending=False))

    df_yield_pred=pd.DataFrame(dfs_pred, index=[last_year])
    st.write('df_yield_pred', df_yield_pred)

    # Adding the Yield predictions to the 'df_yields'
    df_yields=pd.concat([df_yields, df_yield_pred])

    # Adding 'last year' as the average of the last 5
    df_prod_weights=fu.add_estimate(df_prod_weights, last_year, how='mean', last_n_years=5, normalize=True, overwrite=False)
    df_harv_pct=fu.add_estimate(df_harv_pct, last_year, how='mean', last_n_years=5, normalize=False, overwrite=True)

    # st.write('df_prod_weights',df_prod_weights.sort_index(ascending=False))
    st.write('df_harv_pct',df_harv_pct.sort_index(ascending=False))
    st.write('df_plant',df_plant.sort_index(ascending=False))
    st.write('df_harv',df_harv.sort_index(ascending=False))
    st.write('df_yields',df_yields.sort_index(ascending=False))

    df_harv = df_plant * df_harv_pct
    df_harv['us_total']=df_harv.sum(axis=1)
    
    df_prod=df_harv * df_yields
    df_prod['us_total']=df_prod.sum(axis=1)
    df_yields = df_prod/df_harv     

    # 'State Conditions' * 'Weight' = 'Total US Condition'
    us_total_weight_cond=df_prod_weights * us_total_conditions    
    us_total_weight_cond['us_total_conditions']=us_total_weight_cond.sum(axis=1)
    st.write('us_total_weight_cond',us_total_weight_cond.sort_index(ascending=False))

    # Get NATIONAL Yield to use against the 'Total US Condition'
    us_total_yield= qs.get_USA_yields(commodity, aggregate_level='NATIONAL')[['Value']]
    us_total_yield=us_total_yield.rename(columns={'Value':'us_total_yield'})
    # st.write('us_total_yield',us_total_yield)

    # Concatenate 'US Conditions' vs 'NATIONAL Yield'
    df_us_total=pd.concat([us_total_weight_cond['us_total_conditions'],us_total_yield['us_total_yield']], axis=1)
    mask=(df_us_total['us_total_conditions']>0) # this to drop the 0s 
    df_us_total=df_us_total[mask]
    # st.write('df_us_total', df_us_total)

    # Evaluate
    # st.write('df_yields',df_yields.sort_index(ascending=False))
    st.write('Prediction for '+str(last_year),df_yields.loc[last_year].sort_index(ascending=False))

    us_total_results=fu.get_CCI_results_us_total(df_us_total, hovermode)

    st.write(commodity)
    for f in us_total_results:
        st.markdown("---")
        st.plotly_chart(f['fig'], use_container_width=True)
        
        with st.expander(f['state'] + ' - Model Details'):
            st.dataframe(f['df'])
            st.write(f['model'].summary())
    