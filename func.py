import os
from datetime import datetime as dt
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

from calendar import isleap

API_KEY = "F1BC9835-9E98-349E-AAA1-28B4142D289A"

wwht_regions = ['US TOTAL', 'ALABAMA', 'ARKANSAS', 'COLORADO',
       'GEORGIA', 'IDAHO', 'ILLINOIS', 'INDIANA', 'KANSAS',
       'KENTUCKY',  'MICHIGAN', 'MISSISSIPPI',
       'MISSOURI', 'MONTANA', 'NEBRASKA', 'NEW JERSEY', 
       'NEW YORK', 'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'OKLAHOMA',
       'OREGON', 'PENNSYLVANIA', 'SOUTH CAROLINA', 'SOUTH DAKOTA',
       'TENNESSEE', 'TEXAS', 'VIRGINIA', 'WASHINGTON',
       'WEST VIRGINIA', 'WISCONSIN', 'WYOMING']
swht_regions = ['US TOTAL', 'NORTH DAKOTA', 'MINNESOTA','MONTANA', 'SOUTH DAKOTA', 'IDAHO', 'OREGON', 'WASHINGTON']
durum_regions = ['NORTH DAKOTA', 'MONTANA']

grains_class = ['SPRING, (EXCL DURUM)', 'SPRING, DURUM', 'WINTER']
crop_progress = ['PCT EMERGED', 'PCT HARVESTED', 'PCT HEADED', 'PCT PLANTED']

wwht_regions=[t.title() for t in wwht_regions]
swht_regions=[t.title() for t in swht_regions]
durum_regions=[t.title() for t in durum_regions]
grains_class=[t.title() for t in grains_class]
crop_progress=[t.title() for t in crop_progress]


# Functions
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

def get_conditions(state_name:str, class_desc:str) -> pd.DataFrame:
    state_name, class_desc = state_name.replace(" ","%20"), class_desc.replace(" ","%20")
    url_progress = f"https://quickstats.nass.usda.gov/api/api_GET/?key={API_KEY}&commodity_desc=WHEAT&statisticcat_desc=CONDITION"
    url_progress = f'{url_progress}&state_name={state_name}&class_desc={class_desc}&format=JSON'
    response = requests.get(url_progress)
    df = pd.json_normalize(response.json()['data'])

    df[['year', 'Value']] = df[['year', 'Value']].astype(int)
    df['week_ending'] = pd.to_datetime(df['week_ending'])    

    mask=df['unit_desc'].isin(['PCT EXCELLENT', 'PCT GOOD'])
    df = df[mask].groupby(['year', 'week_ending'], as_index=False).agg({'Value': 'sum'})

    mask=(df['Value']>0) # this to drop the 0s 
    df=df[mask]
    df=df.set_index('week_ending')
    
    df=inside_yearly_interpolation(df,'year')
    df=add_seas_day(df, dt(dt.today().year,9,1))
    return df

def get_yields(state_name: str, class_desc: str) -> pd.DataFrame:
    state_name=state_name.upper()
    class_desc=class_desc.upper()
    
    state_name, class_desc = state_name.replace(" ","%20"), class_desc.replace(" ","%20")
    if state_name != 'US%20TOTAL':
        response = requests.get(f"https://quickstats.nass.usda.gov/api/api_GET/?key={API_KEY}&source_desc=SURVEY"
                                f"&state_name={state_name}&commodity_desc=WHEAT&agg_level_desc=STATE&statisticcat_desc"
                                f"=YIELD&reference_period_desc=YEAR&unit_desc=BU%20/%20ACRE&prodn_practice_desc=ALL"
                                f"%20PRODUCTION%20PRACTICES&class_desc={class_desc}&year__GE=1980&format=JSON")
    else:
        response = requests.get(f"https://quickstats.nass.usda.gov/api/api_GET/?key={API_KEY}&source_desc=SURVEY"
                                f"&state_name={state_name}&commodity_desc=WHEAT&statisticcat_desc"
                                f"=YIELD&reference_period_desc=YEAR&unit_desc=BU%20/%20ACRE&prodn_practice_desc=ALL"
                                f"%20PRODUCTION%20PRACTICES&class_desc={class_desc}&year__GE=1980&format=JSON")
    yield_state = pd.json_normalize(response.json()['data'])
    yield_state = yield_state[['year', 'Value']]
    yield_state=yield_state.set_index('year')
    yield_state['Value'] = yield_state['Value'].astype(float)
    yield_state=yield_state.sort_index()
    return yield_state

def get_conditions_chart(df: pd.DataFrame, state_name:str, class_desc:str, hovermode: str):
    last_year = df['year'].max()
    title_text = state_name + ' - ' + class_desc +' GE Conditions'
    labels={"seas_day": "Week Ending","Value": "GE Conditions"}

    mask=df['year']<last_year-1
    fig = px.line(df[mask], x='seas_day', y='Value', color='year', title=title_text, color_discrete_sequence=px.colors.qualitative.Plotly,labels=labels)
    fig.update_traces(line=dict(width=1))

    mask=df['year']==last_year-1
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask]['Value'],fill=None, mode='lines', name=str(last_year-1),text=str(last_year), line=dict(color='black', width=3)))

    mask=df['year']==last_year
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask]['Value'],fill=None, mode='lines', name=str(last_year), text=str(last_year), line=dict(color='red', width=4)))

    fig.update_layout(hovermode=hovermode, width=1000, height=850, xaxis=dict(tickformat="%b %d"))
    return fig

def get_yields_charts(df_conditions: pd.DataFrame, state_name:str, class_desc:str, hovermode: str):
    df_yields = get_yields(state_name, class_desc)
    last_year = df_conditions['year'].max()
    if not last_year in df_yields.index:
        df_yields.loc[last_year] = df_yields.loc[last_year-1]

    i=df_conditions.index
    last_day = i[-1]

    mask= ((i.month==last_day.month) & (i.day==last_day.day))
    
    df_conditions = df_conditions[mask]
    df_conditions = df_conditions.merge(df_yields, left_on='year', right_index=True,) # Add the yield

    # Both 'Conditions' (right -> x) and 'Yield' are named 'Value', so after the merging it is necessary to rename
    df_conditions=df_conditions.rename({'Value_x':'Conditions', 'Value_y':'Yield'}, axis=1)
    df_conditions['Delta Yield'] = df_conditions['Yield'].diff()
    df_conditions['Delta Conditions'] = df_conditions['Conditions'].diff()
    df_conditions['Prev_Yield']=df_conditions['Yield'].shift(1)
    df_conditions = df_conditions.dropna() # because with Delta, the first one it is going to be NaN (as there is no previous year to the first one)
    df_conditions=df_conditions.set_index('year',drop=False)


    fo = []
    # Yield Chart
    if True:
        x='Conditions'
        y='Yield'    
        fig = px.scatter(df_conditions, x=x, y=y, text='year', trendline="ols")

        all_models=px.get_trendline_results(fig).px_fit_results
        model=all_models[0]
        
        add_today(fig=fig,df=df_conditions,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
        prediction = add_today(fig=fig,df=df_conditions,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
            
        title=state_name +' - ' + class_desc + ' - '+y+' vs ' + x + ' - Prediction: ' + format(prediction,'.2f')
        fig.update_traces(textposition="top center")
        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=850, xaxis=dict(tickformat="%b %d"))
        fo.append({'fig':fig,'model':model})

    # Delta Chart
    if True:
        x='Delta Conditions'
        y='Delta Yield'
        fig = px.scatter(df_conditions, x=x, y=y, text='year', trendline="ols")

        all_models=px.get_trendline_results(fig).px_fit_results
        model=all_models[0]
        
        add_today(fig=fig,df=df_conditions,x_col=x, y_col=y, size=10, color='red', symbol='star', name='Today', model=None) # add today
        prediction = df_conditions['Prev_Yield'].values[-1]+ add_today(fig=fig,df=df_conditions,x_col=x, y_col=y, size=7, color='black', symbol='x', name='Model', model=model) # add prediction
        
        # As this is a delta chart, I need to add the previous year to the estimate

        title=state_name +' - ' + class_desc + ' - '+y+' vs ' + x +' (YOY) ' + ' - Prediction: ' + format(prediction,'.2f')
        fig.update_traces(textposition="top center")
        fig.update_layout(title= title, hovermode=hovermode, width=1000, height=850, xaxis=dict(tickformat="%b %d"))
        fo.append({'fig':fig,'model':model})

    return fo

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

# Utilities
def show_excel_index(df, file='check', index=True):
    program =r'"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"'
    df.to_csv(file+'.csv', index=index,)    
    os.system("start " +program+ " "+file+'.csv')