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

    # Events
    def on_change_commodity():
        if 'crop_conditions' in st.session_state:
            del st.session_state['crop_conditions']

# Commodity / State selection
with st.sidebar:
    crop_year_start=dt(dt.today().year,1,1)
    commodity = st.selectbox("Commodity", fu.commodities, 2, on_change=on_change_commodity)

    if commodity.upper() in qs.wheat_by_class:
        comm_download=qs.wheat_by_class[commodity.upper()]
    else:
        comm_download=commodity

    if commodity in st.session_state['crop_conditions']:
        options_states=st.session_state['crop_conditions'][commodity]['options_states']
    else:
        st.session_state['crop_conditions'][commodity]={}
        st.session_state['crop_conditions'][commodity]['options_states']=None
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=None
        st.session_state['crop_conditions'][commodity]['df_yields']=None

        with st.spinner('Checking Available States...'):
            options_states=qs.get_USA_conditions_states(comm_download)
            st.session_state['crop_conditions'][commodity]['options_states']=options_states
    
    # Commodity customization
    if 'WHEAT, WINTER' in commodity.upper():
        crop_year_start=dt(dt.today().year,9,1)

    if commodity != 'WHEAT, SPRING, DURUM'.title():
        options_states=['US Total'] + options_states # Because USDA doesn' provide National numbers for 'WHEAT, SPRING, DURUM'

    state = st.selectbox("State", options_states)
    if 'US Total' in options_states:
        state='US Total'

    hovermode = st.selectbox('Hovermode',['x', 'y', 'closest', 'x unified', 'y unified'],index=2)

    TOTAL_US_DM=False
    if state=='US Total':
        TOTAL_US_DM = st.checkbox('Avere US Total', False, on_change=on_change_commodity)

# Retrieve the data
if True or (st.session_state['crop_conditions'][commodity]['dfs_conditions'] is None):
    # st.write('From USDA')

    if TOTAL_US_DM:
        if 'US Total' in options_states:
            options_states.remove('US Total')

        selected=options_states # for us total
        dfs_conditions=qs.get_USA_conditions_parallel(comm_download.upper(), state_name=selected)
        df_yields=qs.get_USA_yields_weights(comm_download.upper(), aggregate_level='STATE', state_name=selected,output='value')        
        df_prod_weights=  qs.get_USA_prod_weights(commodity, aggregate_level='STATE', output='%')
        df_plant= qs.get_USA_area_planted_weights(commodity, aggregate_level='STATE', output='value', n_years_estimate_by_class=5)
        df_harv=qs.get_USA_area_harvested_weights(commodity, aggregate_level='STATE', output='value')
        
        st.session_state['crop_conditions'][commodity]['dfs_conditions']=dfs_conditions
        st.session_state['crop_conditions'][commodity]['df_yields']=df_yields
        st.session_state['crop_conditions'][commodity]['df_prod_weights']=df_prod_weights
        st.session_state['crop_conditions'][commodity]['df_plant']=df_plant
        st.session_state['crop_conditions'][commodity]['df_harv']=df_harv        
    else:
        selected=[state]
        
        if selected[0]=='US Total':
            if commodity.upper() in qs.wheat_by_class:
                st.markdown('The USDA does NOT give a total by-class.')
                st.markdown('* Tick the "Avere US Total" on the left sidebar to Calculate it from State Yields')
                st.markdown('* It takes some time, as I has to download the full history of state-by-state conditions')
                st.stop()

            dfs_conditions=qs.get_USA_conditions_parallel(comm_download.upper(),aggregate_level='NATIONAL', state_name=selected)
            df_yields=qs.get_USA_yields_weights(comm_download.upper(), aggregate_level='NATIONAL', state_name=selected,output='value')
        else:
            dfs_conditions=qs.get_USA_conditions_parallel(comm_download.upper(), state_name=selected)
            df_yields=qs.get_USA_yields_weights(comm_download.upper(), aggregate_level='STATE', state_name=selected,output='value')
else:
    st.write('From cache')
    dfs_conditions=st.session_state['crop_conditions'][commodity]['dfs_conditions']
    df_yields=st.session_state['crop_conditions'][commodity]['df_yields']
    df_prod_weights=st.session_state['crop_conditions'][commodity]['df_prod_weights']
    df_plant=st.session_state['crop_conditions'][commodity]['df_plant']
    df_harv=st.session_state['crop_conditions'][commodity]['df_harv']

# Common calculations
if True:
    # Extract the Good and Excellent
    dfs_GE={}
    
    for state, df in dfs_conditions.items():
        dfs_GE[state]=qs.extract_GE_conditions(df, crop_year_start)

    CCI_results = fu.get_CCI_results(dfs_GE, df_yields, hovermode=hovermode)

# 'Home made' US total calculation
if TOTAL_US_DM:    
    # Extract the Good and Excellent
    dfs_years, dfs_cond={}, {}

    for state, df in dfs_conditions.items():
        dfs_years[state]=dfs_GE[state][['year']]
        dfs_cond[state]=dfs_GE[state][['Value']]    
    
    # all_years df
    all_years=pd.concat(dfs_years, axis=1)
    all_years.columns=all_years.columns.droplevel(level=1)
    all_years.columns=[c.upper() for c in all_years.columns]
    idx_years=all_years.T.mean()    

    # all_conditions df
    all_conditions=pd.concat(dfs_cond, axis=1, join='outer')
    all_conditions.columns=all_conditions.columns.droplevel(level=1)    
    all_conditions.columns=[c.upper() for c in all_conditions.columns]
    all_conditions.index=idx_years    
        
    # Get last available year and Add 'last year' estimate as the average of the 'last N'
    df_harv_pct=df_harv/df_plant
    last_year=all_conditions.last_valid_index()
    last_n_years=3
    df_prod_weights=fu.add_estimate(df_prod_weights, last_year, how='mean', last_n_years=last_n_years, normalize=True, overwrite=False)
    df_harv_pct=fu.add_estimate(df_harv_pct, last_year, how='mean', last_n_years=last_n_years, normalize=False, overwrite=True)

    # Removing useless years from the 'df_prod_weights' (so there the matrix doesn't change its lenght)
    mask= ((df_prod_weights.index >= min(idx_years)) & (df_prod_weights.index <= max(idx_years)))
    df_prod_weights=df_prod_weights[mask]

    # Calculating Total US Yield by using the single state yields (from CCI regressions)
    if True:        
        dfs_pred={}
        for f in CCI_results:
            if f['analysis']=='delta':
                dfs_pred[f['state'].upper()]=f['prediction']

        df_state_yield_pred=pd.DataFrame(dfs_pred, index=[last_year])        

        # Adding 'last_year' Yield predictions to the 'df_yields'
        df_state_yield_pred=pd.concat([df_yields, df_state_yield_pred])

        df_harv = df_plant * df_harv_pct
        df_harv['US TOTAL']=df_harv.sum(axis=1)
        
        df_prod=df_harv * df_state_yield_pred
        df_prod['US TOTAL']=df_prod.sum(axis=1)
        df_state_yield_pred = df_prod/df_harv

    # 'all_conditions_raw' to show in the Data Details
    all_conditions_raw=all_conditions[:]
    all_conditions_raw.index=all_years.index

    all_conditions=all_conditions * df_prod_weights    
    
    # Yearly interpolation (by definition it needs the 'year' column and the )
    all_conditions['year']=idx_years.values
    all_conditions.index=all_years.index # to put back the actual day and calcualte the 'seas_day' (the year in the index was used to be able to multiply by the yearly production %)
    all_conditions=qs.yearly_interpolation(all_conditions, col_year='year', fill_forward=True, fill_backward=True)

    # to find the total conditions (by sum) I need to exclude the 'year'
    sel_col = [c for c in all_conditions.columns if c != 'year']
    all_conditions['US TOTAL']=all_conditions[sel_col].sum(axis=1)        
    all_conditions=fu.add_seas_day(all_conditions, crop_year_start)

    # Adjusting 'dfs_GE' and 'all_conditions' to make it work like 'US TOTAL' was a state (as it is when downloading from QuickStats)
    all_conditions=all_conditions.rename(columns={'US TOTAL':'Value'})
    dfs_GE={'US TOTAL':all_conditions[['year','seas_day', 'Value']]} # output columns are 'year', 'seas_day' (for the chart), 'Value' (GE = good + excellent)

    # Old Implementation ('US Total' from USDA)
    if False:
        df_yields = qs.get_USA_yields(comm_download, aggregate_level='NATIONAL')[['Value']]
        df_yields=df_yields.rename(columns={'Value':'US TOTAL'})
        # New Implementation ('US Total' from calculation)
    else:    
        # This is for when there are still no conditions for the new year and so it provides the last actual value and the calculated value for the same year
        df_yields = df_state_yield_pred[:]        
        mask=(df_yields.index.duplicated(keep='first'))
        df_yields=df_yields[~mask]


    # Calculating the CCI Results
    CCI_results = fu.get_CCI_results(dfs_GE, df_yields, hovermode=hovermode)

    with st.expander('Calculation Data Details'):
        st.write('all_conditions_raw',all_conditions_raw.sort_index(ascending=False))
        
        st.write('df_plant',df_plant.sort_index(ascending=False))
        st.write('df_harv',df_harv.sort_index(ascending=False))
        st.write('df_harv_pct',df_harv_pct.sort_index(ascending=False))
        
        st.write('df_prod_weights',df_prod_weights.sort_index(ascending=False))

        st.write('all_conditions',all_conditions.sort_index(ascending=False))

        sorted_cols=['US TOTAL'] + [c for c in df_state_yield_pred.columns if 'TOTAL' not in c]
        st.write('df_yield_pred', df_state_yield_pred[sorted_cols].sort_index(ascending=False))

    # Bottom Up Approach
    if True:
        st.markdown('---')
        st.markdown('### Bottom Up Approach')
        st.markdown('1) Calculate state-by-state yields (from CCI regressions)')
        st.markdown('2) Calculate the US TOTAL by aggregating the above')
        tmp=df_state_yield_pred[sorted_cols].loc[last_year]
        if len(tmp)==2:
            tmp.index=[str(int(tmp.index[0])), str(int(tmp.index[0])) + ' Avere']
        tmp=tmp.T.dropna()
        st.write(tmp)

    # Production Weighted Average Approach
    if True:
        st.markdown('---')
        st.markdown('### Production Weighted Average Approach')
        st.markdown('1) Calculate US Total by averaging the single states conditions (weighed by production)')
        st.markdown('2) Regress the US Total conditions vs Total US Yield')

# Charts
if True:
    already_plot=[]
    for f in CCI_results:
        state=f['state']
        if state not in already_plot:
            st.plotly_chart(fu.get_conditions_chart(dfs_GE[state], state, commodity, hovermode=hovermode), use_container_width=True)
            already_plot.append(state)

        st.plotly_chart(f['fig'], use_container_width=True)        
        with st.expander(state + ' - Model Details'):
            st.dataframe(f['df'].sort_index(ascending=False))
            st.write(f['model'].summary())
