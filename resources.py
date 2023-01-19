import requests 
import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

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

    df[['year', 'end_code', 'Value']] = df[['year', 'end_code', 'Value']].astype(int)
    df['week_ending'] = pd.to_datetime(df['week_ending'])
    df = df[df['unit_desc'].isin(['PCT EXCELLENT', 'PCT GOOD'])].groupby(['year', 'end_code', 'week_ending'], as_index=False).agg({'Value': 'sum'})
    df['day'] = df['week_ending'].dt.day
    df['month'] = df['week_ending'].dt.month
    df['new_year'] = df['week_ending'].dt.year
    df=df.rename({'year':'year1'}, axis=1)

    # 'seas_day' is to chart seasonally
    df['year'] = np.where(df['new_year']==df['year1'], 2020, 2019)
    df['seas_day'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.drop(['day', 'month', 'new_year', 'year'], axis=1)
    df=df.rename({'year1':'year'}, axis=1)
    df = df.sort_values(by=['year', 'seas_day'])
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
    fig.add_trace(go.Scatter(x=df[mask]['seas_day'], y=df[mask]['Value'],fill=None, mode='lines+markers', name=str(last_year), text=str(last_year), line=dict(color='firebrick', width=4)))

    fig.update_layout(hovermode=hovermode, width=1000, height=850, xaxis=dict(tickformat="%b %d"))
    return fig

def get_yields_charts(df_conditions: pd.DataFrame, state_name:str, class_desc:str, hovermode: str):
    df_yields = get_yields(state_name, class_desc)
    last_year = df_conditions['year'].max()
    if not last_year in df_yields.index:
        df_yields.loc[last_year] = df_yields.loc[last_year-1]

    # I don't know why Semen didn't just get the last available row
    # Instead of a asking for 1) last seas day -> 2) identify last day in the last year
    mask= (df_conditions['year']==last_year)
    last_day = df_conditions[mask]['seas_day'].max()

    mask= ((df_conditions['year']==last_year) & (df_conditions['seas_day']==last_day))
    last_week = df_conditions[mask]['end_code'].values[0]

    # Get all the corresponding weeks of all years
    mask=(df_conditions['end_code']==last_week)
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


