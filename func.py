import os
from datetime import datetime as dt
from calendar import isleap
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

import QuickStats as qs
import streamlit as st

# Declaration
if True:
    charts_height = 800

    commodities=['CORN',
                 'SOYBEANS',
                 'WHEAT, WINTER',
                 'WHEAT, WINTER, RED, HARD',
                 'WHEAT, WINTER, RED, SOFT',
                 'WHEAT, SPRING, (EXCL DURUM)',
                 'WHEAT, SPRING, DURUM']    
    commodities=[t.title() for t in commodities]

# Core Functions
def get_progress_chart(commodity, state_name, progress_var, crop_year_start, hovermode):

    if state_name=='US Total':
        df = qs.get_USA_progress(commodity, progress_var, aggregate_level='NATIONAL', cols_subset=['week_ending', 'year', 'Value'])
    else:
        df = qs.get_USA_progress(commodity, progress_var, aggregate_level='STATE', state_name=[state_name], cols_subset=['week_ending', 'year', 'Value'])

    df['week_ending'] = pd.to_datetime(df['week_ending'])
    df = df.set_index('week_ending')
        
    df=add_seas_day(df,crop_year_start)
    i=df.index

    last_year = i.year.max()
    print()
    x='seas_day'
    y='Value'
    title_text = state_name+' - '+ commodity+ ' - ' + progress_var

    mask=(i.year<last_year-1)
    fig = px.line(df[mask], x=x, y=y, color='year', title=title_text)
    
    fig.update_traces(line=dict(width=1))

    mask=(i.year==last_year-1)
    fig.add_trace(go.Scatter(x=df[mask][x], y=df[mask][y], mode='lines', name=str(last_year-1), text=str(last_year), line=dict(color='black', width=3)))

    mask=(i.year==last_year)
    fig.add_trace(go.Scatter(x=df[mask][x], y=df[mask][y], mode='lines', name=str(last_year), text=str(last_year), line=dict(color='red', width=3)))
    
    fig.update_layout(hovermode=hovermode, width=1000, height=1000, xaxis=dict(tickformat="%b %d"))
    return fig

def get_conditions_chart(df, state_name, class_desc, hovermode, col='Value'):
    last_year = df['year'].max()
    title_text = state_name + ' - ' + class_desc +' GE Conditions'

    mask=df['year']<last_year-1
    fig = px.line(df[mask], x='seas_day', y=col, color='year', title=title_text, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(line=dict(width=1))

    mask=df['year']==last_year-1
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask][col],fill=None, mode='lines', name=str(last_year-1),text=str(last_year), line=dict(color='black', width=3)))

    mask=df['year']==last_year
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask][col],fill=None, mode='lines+markers', name=str(last_year), text=str(last_year), line=dict(color='red', width=4)))

    fig.update_layout(hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
    return fig

def get_CCI_results(crop_to_estimate, dfs_conditions, dfs_yields, hovermode, n_years_for_trend, crop_year_start, rsq_analysis=False, progress_bar=None):
    fo = []
    metrics={}
    proj_size=8; proj_color='orange'; proj_symbol='x'; proj_name='Proj Condition'
    cci_size=10; cci_color='red'; cci_symbol='star'; cci_name='CCI Model'
    trend_size=8; trend_color='black'; trend_symbol='x'; trend_name='Trend Yield'
    bottom_size=8; bottom_color='green'; bottom_symbol='x'; bottom_name='Bottom up'
    usda_size=8; usda_color='blue'; usda_symbol='x'; usda_name='USDA'

    complete=0
    step=1.0/len(dfs_conditions)
    for state in dfs_conditions:
        if not rsq_analysis:
            complete=complete+step; progress_bar.progress(complete, text=f'Calculating {state}...'  ); 

        # Preliminaries
        if True:
            commodity=''
            trend_yield_value=-1
            end_season_condition=None
            df = dfs_conditions[state][:]   
            
            df_yield = dfs_yields[[state.upper()]].dropna() # this is for the hole in the by-class data            
            
            last_day_data = df.index[-1]

            # Pivoting and Extending the df so to have all the years in the scatter plot
            df = df.pivot(index='seas_day',columns='year',values='Value').fillna(method='ffill').fillna(method='bfill').melt(ignore_index=False)
            df['seas_day']=df.index
            df=df.rename(columns={'value':'Value'})

            # WIP: Selecting the 'last_day' after the extrapolation gives the last available day for the current year and end of the season for every other year
            last_day_season = df.index[-1]

            # Last Year calculated from the conditions
            last_year = int(df['year'].max())

            # If there is no estimate for the Yield, add the trend Yield
            if not last_year in df_yield.index:
                lvi=df_yield.last_valid_index()
                df_yield.loc[last_year] = df_yield.loc[lvi] # add row/index for the current year to be able to calculate the trend yield
                df_yield['year']=df_yield.index # because otherwise 'year' is wrong
                recent_size=trend_size; recent_color=trend_color; recent_symbol=trend_symbol; recent_name=trend_name
                df_yield.loc[last_year] = trend_yield(df_yield, start_year=last_year, n_years_min=n_years_for_trend, rolling=False, yield_col=state.upper()).loc[last_year][['trend_yield']].values[0]
                metrics[trend_name]=df_yield.loc[last_year][state.upper()]
            else:
                if last_year==crop_to_estimate:
                    recent_size=bottom_size; recent_color=bottom_color; recent_symbol=bottom_symbol; recent_name=bottom_name
                else:
                    recent_size=usda_size; recent_color=usda_color; recent_symbol=usda_symbol; recent_name=usda_name

                df_yield['year']=df_yield.index
                trend_yield_value = trend_yield(df_yield, start_year=last_year, n_years_min=n_years_for_trend, rolling=False, yield_col=state.upper()).loc[last_year][['trend_yield']].values[0]
                metrics[trend_name]=trend_yield_value
                metrics[recent_name]=df_yield.loc[last_year][state.upper()]


            df_yield['year']=df_yield.index

            # R-Squared Analysis
            if rsq_analysis:
                # whole period if requested
                days_rsq = df.index.unique()
            else:
                # only the user selected day
                days_rsq = [last_day_data]
                days_rsq = [last_day_season]

            rsq_df={'day':[],'rsq':[]}
            rsq_df_1={'day':[],'rsq':[]}
            df_original=df[:]
            for last_day in days_rsq:  
                # Condition now vs Condition final
                if True:
                    df=df_original[:]
                    x='last_data'
                    y='last_season'

                    # Last data available
                    mask= ((df.index.month==last_day.month) & (df.index.day==last_day.day))
                    df_last_data = df[mask]
                    df_last_data=df_last_data.rename(columns={'Value':'last_data'})
                    df_last_data=df_last_data.set_index('year',drop=True)

                    # Last data of the season
                    mask= ((df.index.month==last_day_season.month) & (df.index.day==last_day_season.day))
                    df_last_season = df[mask] 
                    df_last_season=df_last_season.rename(columns={'Value':'last_season'})
                    df_last_season=df_last_season.set_index('year',drop=True)

                    df_data_vs_seas = pd.concat([df_last_data, df_last_season], axis=1)
                    df_data_vs_seas = df_data_vs_seas.dropna() # because with Delta, the first one it is going to be NaN (as there is no previous year to the first one)
                    mask=((df_data_vs_seas['last_data']>0.0001) & (df_data_vs_seas['last_season']>0.0001)) # to remove the missing data
                    df_data_vs_seas=df_data_vs_seas[mask]

                    rsq = Fit_Model(df_data_vs_seas,y_col=y,x_cols=[x], extract_only='rsquared')
                    rsq_df_1['day'].append(last_day)
                    rsq_df_1['rsq'].append(rsq)

                    # if (last_day.month==last_day_season.month) and (last_day.day==last_day_season.day):
                    if (last_day.month==last_day_data.month) and (last_day.day==last_day_data.day):
                        df_data_vs_seas=df_data_vs_seas.drop(columns=['seas_day'])
                        analysis='now_vs_season'
                        # fig = px.scatter(df_data_vs_seas, x=x, y=y, text=df_data_vs_seas.index, trendline="ols")
                        fig = px.scatter(df_data_vs_seas, x=x, y=y,text=df_data_vs_seas.index, trendline="ols")

                        all_models=px.get_trendline_results(fig).px_fit_results
                        model=all_models[0]
                        
                        current_condition=df_data_vs_seas[x].values[-1]
                        add_today(fig=fig,df=df_data_vs_seas,x_col=x, y_col=y, size=cci_size, color=cci_color, symbol=cci_symbol, name='Current Conditions', model=None) # add today
                        prediction = add_today(fig=fig,df=df_data_vs_seas,x_col=x, y_col=y, size=proj_size, color=proj_color, symbol=proj_symbol, name='End of Season Conditions', model=model) # add prediction
                        end_season_condition=prediction
                        title=state +' - Current Condition: ' + format(current_condition,'.2f') +' - End of season Prediction: ' + format(prediction,'.2f')
                        fig.update_traces(textposition="top center")
                        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
                        fo.insert(0,{'state':state,'analysis':analysis, 'fig':fig,'model':model, 'df':df_data_vs_seas, 'prediction':prediction})

                # CCI analysis R-squared
                if True:
                    x='Delta Conditions'
                    y='Delta Yield'
                    df=df_original[:]

                    # Selecting the same day for each of the available years
                    mask= ((df.index.month==last_day.month) & (df.index.day==last_day.day))        
                    df = df[mask]

                    # Add the yield for each year to the 'condition df'            
                    df = df.merge(df_yield, left_on='year', right_index=True,)

                    df=df.rename({'Value':'Conditions', state.upper():'Yield'}, axis=1)
                    df['Delta Yield'] = df['Yield'].diff()
                    df['Delta Conditions'] = df['Conditions'].diff()
                    df['Prev_Yield']=df['Yield'].shift(1)
                    df = df.dropna() # because with Delta, the first one it is going to be NaN (as there is no previous year to the first one)
                    df=df.set_index('year',drop=False)

                    if len(df)>0:
                        rsq = Fit_Model(df,y_col=y,x_cols=[x], extract_only='rsquared')
                        rsq_df['day'].append(last_day)
                        rsq_df['rsq'].append(rsq)
           
        # Delta Chart
        if True and len(df)>0:
            analysis='delta'
            x='Delta Conditions'
            y='Delta Yield'
            fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]    

            add_today(fig=fig,df=df,x_col=x, y_col=y, size=recent_size, color=recent_color, symbol=recent_symbol, name=recent_name, model=None) # add today
            prediction = df['Prev_Yield'].values[-1]+ add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='CCI Model', model=model) # add prediction
            CCI_yield=prediction
            metrics[cci_name]=CCI_yield
            if trend_yield_value>0: add_point(fig=fig,x=df[x].values[-1], y=trend_yield_value-df['Prev_Yield'].values[-1], size=trend_size, color=trend_color, symbol=trend_symbol, name=trend_name)

            # add end of season estimate
            title_suffix = ''
            if end_season_condition is not None:
                tmp=df.copy().iloc[-2:]
                tmp['Conditions'].iloc[-1]=end_season_condition
                tmp['Delta Conditions'] = tmp['Conditions'].diff()

                pred_df=sm.add_constant(tmp, has_constant='add').loc[last_year][['const',x]]
                end_season_delta=model.predict(pred_df)[0]
                end_season_yield=df['Prev_Yield'].values[-1]+end_season_delta
                metrics[proj_name]=end_season_yield

                title_suffix = ' - End of Season: ' + format(end_season_yield,'.2f')
                add_point(fig=fig,x=tmp[x].values[-1], y=end_season_delta, size=proj_size, color=proj_color, symbol=proj_symbol, name=proj_name)

            # As this is a delta chart, I need to add the previous year to the estimate
            title=state +' - '+y+' vs ' + x +' (YOY) ' + ' - Prediction: ' + format(prediction,'.2f')+ title_suffix
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.insert(0,{'state':state,'analysis':analysis, 'fig':fig,'model':model, 'df':df, 'prediction':prediction})

        # R-Squared Analysis of the delta chart
        if rsq_analysis:
            dfs={1:rsq_df, 3:rsq_df_1}
            titles={1:'R-square Analysis of the CCI Model', 3:'R-square Analysis of the relationship between current and final conditions'}
            for k, rsq_df in dfs.items():
                analysis='rsq'
                rsq_df=pd.DataFrame(rsq_df)
                fig = px.line(rsq_df, x='day', y='rsq')
                title='R-square Analysis of the CCI Model'
                fig.update_layout(title= titles[k], hovermode=hovermode, width=1000, height=charts_height/2, xaxis=dict(tickformat="%b %d"))
                fig.add_vline(x=seas_day(last_day_data, crop_year_start).timestamp() * 1000, line_dash="dash",line_width=1, annotation_text="Last Available Data", annotation_position="bottom")
                fo.insert(k,{'state':state,'analysis':analysis, 'fig':fig,'model':None, 'df':rsq_df, 'prediction':None})                        

        # Historical Yield        
        if True and len(df)>0:
            # For the 'default' plotly color list
            # https://community.plotly.com/t/plotly-colours-list/11730/3
            analysis='hist'
            x='year'
            y=state.upper()
            
            mask= ((df_yield.index>=last_year-n_years_for_trend) & (df_yield.index<=last_year-1))
            trend_df=df_yield[mask]
            fig = px.scatter(trend_df, x=x, y=y, trendline="ols")
            fig.update_traces(marker=dict(size=10, color='black'))
            fig.add_trace(go.Scatter(x=df_yield[x], y=df_yield[y], text=df_yield['year'], mode='markers+text', name='test', line=dict(width=2,color='#1f77b4'), showlegend=False))
            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]

            add_today(fig=fig,df=df_yield,x_col=x, y_col=y, size=recent_size, color=recent_color, symbol=recent_symbol, name=recent_name, model=None) # add today
            # prediction = add_today(fig=fig,df=df_yield,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Trend Yield', model=model) # add prediction
            add_point(fig=fig,x=df_yield[x].values[-1], y=CCI_yield, size=cci_size, color=cci_color, symbol=cci_symbol, name=cci_name)
  
            if end_season_condition is not None:
                add_point(fig=fig,x=df_yield[x].values[-1], y=end_season_yield, size=proj_size, color=proj_color, symbol=proj_symbol, name=proj_name)

            title=state +' - Trend Yield: ' + format(metrics[trend_name],'.2f')
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.append({'state':state,'analysis':analysis, 'fig':fig,'model':model, 'df':df_yield, 'prediction':prediction})
            if trend_yield_value>0: add_point(fig=fig,x=df[x].values[-1], y=trend_yield_value, size=trend_size, color=trend_color, symbol=trend_symbol, name=trend_name)

        # Yield vs Conditions Chart
        if True and len(df)>0:
            analysis='cond'
            x='Conditions'
            y='Yield'
            
            fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]
            
            add_today(fig=fig,df=df,x_col=x, y_col=y, size=recent_size, color=recent_color, symbol=recent_symbol, name=recent_name, model=None) # add today
            # prediction = add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='CCI Model', model=model) # add prediction                
            if trend_yield_value>0: add_point(fig=fig,x=df[x].values[-1], y=trend_yield_value, size=trend_size, color=trend_color, symbol=trend_symbol, name=trend_name)
            add_point(fig=fig,x=df[x].values[-1], y=CCI_yield, size=cci_size, color=cci_color, symbol=cci_symbol, name=cci_name)

            if end_season_condition is not None:
                add_point(fig=fig,x=end_season_condition, y=end_season_yield,  size=proj_size, color=proj_color, symbol=proj_symbol, name=proj_name)

            title=state +' - '+y+' vs ' + x #+ ' - Prediction: ' + format(prediction,'.2f')
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.append({'state':state,'analysis':analysis, 'fig':fig,'model':model, 'df':df, 'prediction':prediction})

        if rsq_analysis:
            metric_cols = st.columns(len(metrics))

            for i, k in enumerate(metrics):
                metric_cols[i].metric(label=k, value="{:.2f}".format(metrics[k]))
    return fo


# Utilities
def add_estimate(df, year_to_estimate, how='mean', last_n_years=5, normalize=False, overwrite=False):
    if (overwrite) or (year_to_estimate not in df.index):
        if how=='mean':
            mask=(df.index>=year_to_estimate-last_n_years)
            mean=df[mask].mean()

        if normalize:
            df.loc[year_to_estimate]=mean/mean.sum()
        else:
            df.loc[year_to_estimate]=mean    
    return df

def last_leap_year():    
    start=dt.today().year
    while(True):
        if isleap(start): return start
        start-=1

def add_seas_day(df, ref_year_start= dt.today(), date_col=None):
    if date_col==None:
        df['seas_day'] = [seas_day(d,ref_year_start) for d in df.index]
    else:
        df['seas_day'] = [seas_day(d,ref_year_start) for d in df[date_col]]
    return df

def seas_day(date, ref_year_start= dt.today()):
    """
    'seas_day' is the X-axis of the seasonal plot:
            - it makes sure to include 29 Feb
            - it is very useful in creating weather windows
    """
    LLY = last_leap_year()
    start_idx = 100 * ref_year_start.month + ref_year_start.day
    date_idx = 100 * date.month + date.day

    if (start_idx<300):
        if (date_idx>=start_idx):
            return dt(LLY, date.month, date.day)
        else:
            return dt(LLY+1, date.month, date.day)
    else:
        if (date_idx>=start_idx):
            return dt(LLY-1, date.month, date.day)
        else:
            return dt(LLY, date.month, date.day)


def add_point(fig, x, y, today_idx=None, size=10, color='black', symbol='star', name='Point', row=1, col=1):
    y_str = 'Y: %{y:.2f}'
    x_str = 'X: %{x:.2f}'
    hovertemplate="<br>".join([name, y_str, x_str, "<extra></extra>"])
    fig.add_trace(go.Scatter(name=name,x=[x], y=[y], mode = 'markers+text', text=name, marker_symbol = symbol,marker_size = size, marker_color=color, hovertemplate=hovertemplate), row=row, col=col)
    return True

def add_today(fig, df, x_col, y_col, today_idx=None, size=10, color='red', symbol='star', name='Today', model=None, row=1, col=1):
    """
    if 'model' is not None, it will calculate the prediction
    markers:
        https://plotly.com/python/marker-style/
    """
    if today_idx is None:
        today_idx=df.index[-1]

    x = df.loc[today_idx][x_col]

    if model is None:    
        y = df.loc[today_idx][y_col]
    else:
        pred_df=sm.add_constant(df, has_constant='add').loc[today_idx][['const',x_col]]
        y=model.predict(pred_df)[0]
    
    y_str = 'Y: '+ y_col +' %{y:.2f}'
    x_str = 'X: '+ x_col +' %{x:.2f}'
    hovertemplate="<br>".join([name, y_str, x_str, "<extra></extra>"])
    fig.add_trace(go.Scatter(name=name,x=[x], y=[y], mode = 'markers', marker_symbol = symbol,marker_size = size, marker_color=color, hovertemplate=hovertemplate), row=row, col=col)
    return y

def predict_with_model(model, pred_df):
    if (('const' in model.params) & ('const' not in pred_df.columns)):
        pred_df = sm.add_constant(pred_df, has_constant='add')

    return model.predict(pred_df[model.params.index])


def Fit_Model(df, y_col: str, x_cols=[], exclude_from=None, extract_only=None):
    """
    'exclude_from' needs to be consistent with the df index
    """

    if not ('const' in df.columns):
        df = sm.add_constant(df, has_constant='add')

    if not ('const' in x_cols):        
        x_cols.append('const')

    if exclude_from!=None:
        df=df.loc[df.index<exclude_from]

    y_df = df[[y_col]]

    if (len(x_cols)>0):
        X_df=df[x_cols]
    else:
        X_df=df.drop(columns = y_col)

    model = sm.OLS(y_df, X_df).fit()

    if extract_only is None:
        fo = model
    elif extract_only == 'rsquared':
        fo = model.rsquared

    return fo

def trend_yield(df_yield, start_year=None, n_years_min=20, rolling=False, yield_col='Value'):
    """
    'start_year'
        - start calculating the trend yield from this year
        - if I put 1995 it will calculate the trend year for 1995 taking into account data up to 1994

    'n_years_min'
        - minimum years included for the trend year calculation
        - if I put 10, and I need to calculate 1995, it will take years from 1985 to 1994 (both included)
    
    simple way to get the input 'df_yield'

    import APIs.QuickStats as qs
    df_yield=qs.get_USA_yields(cols_subset=['Value','year'])
    """

    yield_str='yield'
    trend_str='trend_yield'
    devia_str='yield_deviation'

    # if df_yield.index.name != 'year':
    #     df_yield=df_yield.set_index('year',drop=False)

    df_yield=df_yield.set_index('year',drop=False)

    year_min=int(df_yield.index.min())
    year_max=int(df_yield.index.max())

    if start_year is None:
        start_year=year_min

    fo_dict={'year':[], trend_str:[] }
    for y in range(start_year,year_max+1):
        # this to avoid having nothing when we are considering the first year of the whole df
        year_to=max(y-1,year_min) 
        if rolling:
            mask=((df_yield.index>=y-n_years_min) & (df_yield.index<=year_to))
        else:
            mask=((df_yield.index>=start_year-n_years_min) & (df_yield.index<=year_to))

        if sum(mask)==0:            
            mask=[True]*len(df_yield)

        df_model=df_yield[mask]
        model=Fit_Model(df_model,y_col=yield_col,x_cols=['year'])
        pred = predict_with_model(model,df_yield.loc[y:y])

        fo_dict['year'].append(y)
        fo_dict[trend_str].append(pred[y])

    df=pd.DataFrame(fo_dict)
    df=df.set_index('year')

    df=pd.concat([df_yield,df],axis=1,join='inner')

    df=df.rename(columns={yield_col:yield_str})
    df[devia_str]=100.0*( df[yield_str]/df[trend_str]-1.0)
    df=df.set_index('year')
    return df