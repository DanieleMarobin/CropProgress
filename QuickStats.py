# Quick Stats - Download Interface: https://quickstats.nass.usda.gov
# Quick Stats - API documentation: https://quickstats.nass.usda.gov/api
# Quick Stats - html encoding: https://www.w3schools.com/tags/ref_urlencode.asp
# 50'000 records is the limit for a single call

import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime as dt
from calendar import isleap

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

# 'Time Conversion' dictionary
if True:
    qs_tc = {'CORN': 9,'Daniele': 10}

class QS_input():
    def __init__(self):
        self.source_desc=[]
        self.commodity_desc=[]
        self.short_desc=[]
        self.statisticcat_desc=[]
        self.years=[]
        self.reference_period_desc=[]
        self.domain_desc=[]
        self.agg_level_desc=[]
        self.state_name=[]
        self.freq_desc=[]
        self.class_desc=[]

def QS_url(input:QS_input):
    url = 'http://quickstats.nass.usda.gov/api/api_GET/?key=96002C63-2D1E-39B2-BF2B-38AA97CC7B18&'

    for i in input.source_desc:
        url=url + 'source_desc=' + i +'&'
    for i in input.commodity_desc:
        url=url + 'commodity_desc=' + i +'&'
    for i in input.short_desc: 
        url=url + 'short_desc=' + i +'&'
    for i in input.statisticcat_desc: 
        url=url + 'statisticcat_desc=' + i +'&'        
    for i in input.years: 
        url=url + 'year=' + str(i) +'&'
    for i in input.reference_period_desc: 
        url=url + 'reference_period_desc=' + i +'&'
    for i in input.domain_desc:
        url=url + 'domain_desc=' + i +'&'
    for i in input.agg_level_desc:
        url=url + 'agg_level_desc=' + i +'&'
    for i in input.state_name:
        url=url + 'state_name=' + i +'&'
    for i in input.freq_desc:
        url=url + 'freq_desc=' + i +'&'
    for i in input.class_desc:
        url=url + 'class_desc=' + i +'&'        

    url=url+'format=CSV'
    url = url.replace(" ", "%20")
    return url

def get_data(input: QS_input):
    url = QS_url(input)
    # print(url)
    fo = pd.read_csv(url,low_memory=False)  
    return fo

def get_USA_conditions_states(commodity):
    df=get_USA_conditions(commodity,aggregate_level='STATE',years=[dt.today().year-1])
    fo=list(set(df['state_name']))
    fo.sort()
    fo=[s.title() for s in fo]
    return fo

def get_USA_conditions(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    simple use:
        us_yields=qs.get_USA_yields(cols_subset=['Value','year'])

    commodity = 'CORN', 'SOYBEANS', 'WHEAT', 'WHEAT, WINTER'
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    

    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    
    dl.short_desc.append(commodity+' - CONDITION, MEASURED IN PCT EXCELLENT')
    dl.short_desc.append(commodity+' - CONDITION, MEASURED IN PCT FAIR')
    dl.short_desc.append(commodity+' - CONDITION, MEASURED IN PCT GOOD')
    dl.short_desc.append(commodity+' - CONDITION, MEASURED IN PCT POOR')
    dl.short_desc.append(commodity+' - CONDITION, MEASURED IN PCT VERY POOR')

    # Edit inputs to make the download possible (for example necessary to modify commodity for spring/winter wheat)
    if 'WHEAT' in commodity:
        commodity='WHEAT'

    # dl.statisticcat_desc.append('CONDITION')
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)

    return fo
def get_USA_conditions_parallel(commodity='CORN', aggregate_level='STATE', state_name=[], years=[], cols_subset=[]):
    dfs={}

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results={}
        for s in state_name:
            results[s] = executor.submit(get_USA_conditions, commodity, aggregate_level, [s], years, cols_subset)

    for s, res in results.items():
        dfs[s]=res.result()
    
    # df = pd.concat(dfs.values(), axis=0)

    return dfs

def extract_GE_conditions(df):
    crop_year_start=dt(dt.today().year,1,1)

    df[['year', 'Value']] = df[['year', 'Value']].astype(int)
    df['week_ending'] = pd.to_datetime(df['week_ending'])

    if 'WHEAT, WINTER' in df['short_desc'].values[0]:
        crop_year_start=dt(dt.today().year,9,1)
        # the below 2 are to correct USDA wrong data
        mask=(df['week_ending'].dt.month>crop_year_start.month) & (df['week_ending'].dt.year>=df['year'])
        df.loc[mask,'year']=df['week_ending'].dt.year+1                

    mask=df['unit_desc'].isin(['PCT EXCELLENT', 'PCT GOOD'])
    df = df[mask].groupby(['year', 'week_ending'], as_index=False).agg({'Value': 'sum'})

    mask=(df['Value']>0) # this to drop the 0s 
    df=df[mask]
    df=df.set_index('week_ending')    
    df=inside_yearly_interpolation(df,'year')
    df=add_seas_day(df, crop_year_start)
    return df
def get_USA_yields(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    simple use:
        us_yields=qs.get_USA_yields(cols_subset=['Value','year'])

    commodity = 'CORN', 'SOYBEANS', 'WHEAT', 'WHEAT, WINTER'
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    
    dl.short_desc=[commodity+' - YIELD, MEASURED IN BU / ACRE']

    # Edit inputs to make the download possible (for example necessary to modify commodity for spring/winter wheat)
    if commodity=='CORN':
        dl.short_desc=[commodity+', GRAIN - YIELD, MEASURED IN BU / ACRE']
    elif 'WHEAT' in commodity:
        commodity='WHEAT'

    dl.years.extend(years)
    dl.commodity_desc.append(commodity)
    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]

    fo=fo.sort_values(by='year',ascending=True)
    fo=fo.set_index('year',drop=True)

    return fo
def get_USA_yields_weights(commodity='CORN', aggregate_level='STATE',state_name=[], years=[], subset=[], pivot_column='state_name', output='%'):
    # pivot_column= 'state_name', 'state_alpha'
    # rows:       years
    # columns:    region
    
    fo=get_USA_yields(commodity=commodity,aggregate_level=aggregate_level, state_name=state_name, years=years)
    fo = pd.pivot_table(fo,values='Value',index=pivot_column,columns='year')

    if (len(subset))>0:
        fo=fo.loc[subset]

    if output=='%':
        fo=fo/fo.sum()

    return fo.T

def get_USA_progress_states(commodity):
    df=get_USA_progress(commodity,aggregate_level='STATE',years=[dt.today().year-1])
    fo=list(set(df['state_name']))
    fo.sort()
    fo=[s.title() for s in fo]
    return fo
def get_USA_progress_variables(commodity):
    df=get_USA_progress(commodity,aggregate_level='NATIONAL',progress_var=None,years=[dt.today().year-1])
    fo=list(set(df['unit_desc']))
    fo.sort()
    fo=[s.title() for s in fo]
    return fo
def get_USA_progress(commodity='CORN', progress_var=None, aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    'planting', 'silking', 'BLOOMING', 'harvesting'

    df_planted=qs.get_QS_planting_progress(commodity='SOYBEANS', aggregate_level='NATIONAL', years=[2017],columns_output=['year','week_ending','Value'])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE'

    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.years.extend(years)    

    if progress_var is not None:
        if ((progress_var.lower()=='pct harvested') & (commodity=='CORN')):
            dl.short_desc.append(commodity+', GRAIN - PROGRESS, MEASURED IN PCT HARVESTED')
        else:
            dl.short_desc.append(commodity+' - PROGRESS, MEASURED IN '+progress_var)
            
    
    # Edit inputs to make the download possible (for example necessary to modify commodity for spring/winter wheat)
    elif 'WHEAT' in commodity:
        commodity='WHEAT'

    dl.commodity_desc.append(commodity)
    dl.statisticcat_desc.append('PROGRESS')
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='week_ending',ascending=True)

    return fo

def get_USA_production(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='STATE', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.source_desc.append('SURVEY')
    dl.years.extend(years)
    dl.short_desc=[commodity+' - PRODUCTION, MEASURED IN BU']

    if commodity=='CORN':
        dl.short_desc=[commodity+', GRAIN - PRODUCTION, MEASURED IN BU']
    elif 'WHEAT' in commodity:
        commodity='WHEAT'

    dl.commodity_desc.append(commodity)
    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo
def get_USA_prod_weights(commodity='CORN', aggregate_level='STATE',state_name=[], years=[], subset=[], pivot_column='state_name', output='%'):
    # pivot_column= 'state_name', 'state_alpha'
    # rows:       years
    # columns:    region
    
    fo=get_USA_production(commodity=commodity,aggregate_level=aggregate_level,state_name=state_name, years=years)
    fo = pd.pivot_table(fo,values='Value',index=pivot_column,columns='year')

    if (len(subset))>0:
        fo=fo.loc[subset]

    if output=='%':
        fo=fo/fo.sum()

    return fo.T

def get_USA_area_planted(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='STATE', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.source_desc.append('SURVEY')
    dl.years.extend(years)
    
    dl.short_desc=[commodity+' - ACRES PLANTED']

    if commodity=='CORN':
        dl.short_desc=[commodity+', GRAIN - ACRES PLANTED', commodity+', GRAIN - ACRES HARVESTED']
    elif 'WHEAT' in commodity:
        commodity='WHEAT'

    dl.commodity_desc.append(commodity)
    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo
def get_USA_area_planted_weights(commodity='CORN', aggregate_level='STATE',state_name=[], years=[], subset=[], pivot_column='state_name', output='%'):
    # pivot_column= 'state_name', 'state_alpha'
    # rows:       years
    # columns:    region
    
    fo=get_USA_area_planted(commodity=commodity,aggregate_level=aggregate_level,state_name=state_name, years=years)
    fo = pd.pivot_table(fo,values='Value',index=pivot_column,columns='year')

    if (len(subset))>0:
        fo=fo.loc[subset]

    if output=='%':
        fo=fo/fo.sum()

    return fo.T

def get_USA_area_harvested(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=[], cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='STATE', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.source_desc.append('SURVEY')
    dl.years.extend(years)
    
    dl.short_desc=[commodity+' - ACRES HARVESTED']

    if commodity=='CORN':
        dl.short_desc=[commodity+', GRAIN - ACRES PLANTED', commodity+', GRAIN - ACRES HARVESTED']
    elif 'WHEAT' in commodity:
        commodity='WHEAT'

    dl.commodity_desc.append(commodity)
    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo
def get_USA_area_harvested_weights(commodity='CORN', aggregate_level='STATE',state_name=[], years=[], subset=[], pivot_column='state_name', output='%'):
    # pivot_column= 'state_name', 'state_alpha'
    # rows:       years
    # columns:    region
    
    fo=get_USA_area_harvested(commodity=commodity,aggregate_level=aggregate_level,state_name=state_name, years=years)
    fo = pd.pivot_table(fo,values='Value',index=pivot_column,columns='year')

    if (len(subset))>0:
        fo=fo.loc[subset]

    if output=='%':
        fo=fo/fo.sum()

    return fo.T

def get_ethanol(freq_desc='MONTHLY', years=[], cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='COUNTY', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity='CORN'

    dl = QS_input()
    dl.domain_desc.append('TOTAL')
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)
    dl.freq_desc.append(freq_desc)
    dl.short_desc.append('CORN, FOR FUEL ALCOHOL - USAGE, MEASURED IN BU')

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo

def date_conversion(freq_desc, year, begin_code, end_code, week_ending,commodity_desc):    
    if (freq_desc == 'ANNUAL' and begin_code==0):        
        month = qs_tc[commodity_desc]
        year_offset = 0
    elif (freq_desc == 'WEEKLY'):
        return week_ending
    else: 
        month = begin_code
        year_offset = 0
        
    return dt(int(year + year_offset), month, 1)

def add_date_column(df, col_name='date'):
    df[col_name] = [date_conversion(rr['freq_desc'], rr['year'], rr['begin_code'], rr['end_code'], rr['week_ending'], rr['commodity_desc']) for i, rr in df.iterrows()]
    return df

def extract_date_value(df,output_col_name):
    # After the simple extraction from QuickStat there are a lot of useless columns and not clear timeline
    # This function:
    #       1) Adds the 'date' column
    #       2) Renames the 

    df['date'] = [date_conversion(rr['freq_desc'], rr['year'], rr['begin_code'], rr['end_code'], rr['week_ending'], rr['commodity_desc']) for i, rr in df.iterrows()]

    # I use the 'groupby' because in one line I can:
    #       1) assign 'date' as index
    #       2) just pick the 'Amount' column
    #       3) maybe there is going to be a need in the future not to use 'max' function

    df = df.groupby(by='date')[['Value']].max()
    df=df.rename(columns={'Value': output_col_name}, errors="raise")
    return df

def get_everything(years=[]):
    # Main inputs
    if True:
        dl = QS_input()
        dl.source_desc.append('SURVEY')
        dl.domain_desc.append('TOTAL')
        dl.commodity_desc.append('WHEAT')
        dl.agg_level_desc=['NATIONAL'] # ['NATIONAL','STATE']
        dl.statisticcat_desc=['AREA PLANTED',] # ['YIELD','PRODUCTION','CONDITION','AREA PLANTED','AREA HARVESTED',]
        dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
        dl.class_desc=['WINTER'] # ['WINTER', 'WHEAT, WINTER, WHITE, HARD', ]
        dl.years.extend(years)        

    # Downloading part
    if True:
        fo=get_data(dl)    
        fo=fo.sort_values(by='year',ascending=True)
        # fo['Value'] = fo['Value'].str.replace(',','').astype(float)
        print('Downloaded:', years)

    return fo