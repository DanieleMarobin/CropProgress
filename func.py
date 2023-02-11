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
    charts_height = 850

    commodities=['CORN','SOYBEANS','WHEAT, WINTER','WHEAT, SPRING, (EXCL DURUM)','WHEAT, SPRING, DURUM']    
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

def get_conditions_chart(df: pd.DataFrame, state_name:str, class_desc:str, hovermode: str):
    last_year = df['year'].max()
    title_text = state_name + ' - ' + class_desc +' GE Conditions'

    mask=df['year']<last_year-1
    fig = px.line(df[mask], x='seas_day', y='Value', color='year', title=title_text, color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_traces(line=dict(width=1))

    mask=df['year']==last_year-1
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask]['Value'],fill=None, mode='lines', name=str(last_year-1),text=str(last_year), line=dict(color='black', width=3)))

    mask=df['year']==last_year
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask]['Value'],fill=None, mode='lines', name=str(last_year), text=str(last_year), line=dict(color='red', width=4)))

    fig.update_layout(hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
    return fig

def get_CCI_yield_model_charts(dfs_conditions, dfs_yields, hovermode: str):
    fo = []
    for state in dfs_conditions:
        df = dfs_conditions[state][:]
        df_yield = dfs_yields[state][:]
        df_yield=df_yield[['year','Value']]
        commodity=dfs_yields[state]['commodity_desc'].values[0] + ',' + dfs_yields[state]['class_desc'].values[0]

        # Pivoting and Extending the df so to have all the years in the scatter plot
        df = df.pivot(index='seas_day',columns='year',values='Value').fillna(method='ffill').fillna(method='bfill').melt(ignore_index=False)
        df['seas_day']=df.index
        df=df.rename(columns={'value':'Value'})

        last_year = int(df['year'].max())
        if not last_year in df_yield.index:
            lvi=df_yield.last_valid_index()
            df_yield.loc[last_year] = df_yield.loc[lvi] # add row/index for the current year to be able to calculate the trend yield
            df_yield['year']=df_yield.index # becuase otherwise 'year' is wrong
            df_yield.loc[last_year] =trend_yield(df_yield, start_year=last_year, n_years_min=1000, rolling=False).loc[last_year][['trend_yield']].values[0]
            df_yield['year']=df_yield.index


        # Selecting the last available day overall
        last_day = df.index[-1]

        # Selecting the same day for each of the available years
        mask= ((df.index.month==last_day.month) & (df.index.day==last_day.day))        
        df = df[mask]

        # Add the yield for each year to the 'condition df'
        df = df.merge(df_yield, left_on='year', right_index=True,)

        # Both 'Conditions' (right -> x) and 'Yield' are named 'Value', so after the merging it is necessary to rename
        df=df.rename({'Value_x':'Conditions', 'Value_y':'Yield'}, axis=1)
        df['Delta Yield'] = df['Yield'].diff()
        df['Delta Conditions'] = df['Conditions'].diff()
        df['Prev_Yield']=df['Yield'].shift(1)
        df = df.dropna() # because with Delta, the first one it is going to be NaN (as there is no previous year to the first one)
        df=df.set_index('year',drop=False)
                
        # Historical Yield
        if True and len(df)>0:
            x='year'
            y='Value'
            # st.write(df_yield)
            # mask=df_yield.index<2023
            fig = px.scatter(df_yield, x=x, y=y, text='year', trendline="ols")

            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]
            
            add_today(fig=fig,df=df_yield,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
            prediction = add_today(fig=fig,df=df_yield,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
                
            title=state +' - ' + commodity + ' - '+y+' vs ' + x + ' - Prediction: ' + format(prediction,'.2f')
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.append({'state':state,'fig':fig,'model':model, 'df':df})

        # Yield vs Conditions Chart
        if False and len(df)>0:
            x='Conditions'
            y='Yield'
            fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]
            
            add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
            prediction = add_today(fig=fig,df=df,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
                
            title=state +' - ' + commodity + ' - '+y+' vs ' + x + ' - Prediction: ' + format(prediction,'.2f')
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.append({'state':state,'fig':fig,'model':model, 'df':df})

        # Delta Chart
        if True and len(df)>0:
            x='Delta Conditions'
            y='Delta Yield'
            fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

            all_models=px.get_trendline_results(fig).px_fit_results
            model=all_models[0]
            
            add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
            prediction = df['Prev_Yield'].values[-1]+ add_today(fig=fig,df=df,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
            
            # As this is a delta chart, I need to add the previous year to the estimate

            title=state +' - ' + commodity + ' - '+y+' vs ' + x +' (YOY) ' + ' - Prediction: ' + format(prediction,'.2f')
            fig.update_traces(textposition="top center")
            fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
            fo.append({'state':state,'fig':fig,'model':model, 'df':df})

    return fo

def get_CCI_us_total(df,hovermode: str):
    fo = []
    df['year']=df.index

    last_year = int(df['year'].max())
    df.loc[last_year,'us_total_yield'] =trend_yield(df, start_year=last_year, n_years_min=1000, rolling=False, yield_col='us_total_yield').loc[last_year][['trend_yield']].values[0]

    # Historical Yield
    if True and len(df)>0:
        x='year'
        y='us_total_yield'
        fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

        all_models=px.get_trendline_results(fig).px_fit_results
        model=all_models[0]
        
        add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
        prediction = add_today(fig=fig,df=df,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
            
        title='us total - '+y+' vs ' + x + ' - Prediction: ' + format(prediction,'.2f')
        fig.update_traces(textposition="top center")
        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
        fo.append({'state':'us total','fig':fig,'model':model, 'df':df})

    # Yield vs Conditions Chart
    if True and len(df)>0:
        x='us_total_conditions'
        y='us_total_yield'        
        fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

        all_models=px.get_trendline_results(fig).px_fit_results
        model=all_models[0]
        
        add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
        prediction = add_today(fig=fig,df=df,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
            
        title='us total - '+y+' vs ' + x + ' - Prediction: ' + format(prediction,'.2f')
        fig.update_traces(textposition="top center")
        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
        fo.append({'state':'us total','fig':fig,'model':model, 'df':df})

    # Delta Chart
    df['Delta Yield'] = df['us_total_yield'].diff()
    df['Delta Conditions'] = df['us_total_conditions'].diff()
    df['Prev_Yield']=df['us_total_yield'].shift(1)
    df = df.dropna() # because with Delta, the first one it is going to be NaN (as there is no previous year to the first one)
    df=df.set_index('year',drop=False)
    if True and len(df)>0:
        x='Delta Conditions'
        y='Delta Yield'
        fig = px.scatter(df, x=x, y=y, text='year', trendline="ols")

        all_models=px.get_trendline_results(fig).px_fit_results
        model=all_models[0]
        
        add_today(fig=fig,df=df,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
        prediction = df['Prev_Yield'].values[-1]+ add_today(fig=fig,df=df,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
        
        # As this is a delta chart, I need to add the previous year to the estimate

        title='us total - '+y+' vs ' + x +' (YOY) ' + ' - Prediction: ' + format(prediction,'.2f')
        fig.update_traces(textposition="top center")
        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=charts_height, xaxis=dict(tickformat="%b %d"))
        fo.append({'state':'us total','fig':fig,'model':model, 'df':df})
    return fo

# Utilities
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

def inside_yearly_interpolation(df, col_year):
    """
    Important:
        - I normally pass a very simple df, with year and value and a time index
        - it is important because at the end it just interpolates
        - as it is done on an yearly basis, the year col is going to remain a constant
        - the rest needs to be paid attention to

    the idea is to recreate a new Dataframe by concatenating the yearly interpolated ones
    so there is no risk of interpolating from the end of a crop year to the beginning of the next one
    """
    dfs=[]
    years=np.sort(df[col_year].unique())

    for y in years:
        mask=(df[col_year]==y)

        dfs.append(df[mask].resample('1d').asfreq().interpolate())
    
    return pd.concat(dfs)

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

    if df_yield.index.name != 'year':
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

        df_model=df_yield[mask]
        print(df_model)
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