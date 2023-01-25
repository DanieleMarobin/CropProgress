# Quick Stats - Download Interface: https://quickstats.nass.usda.gov
# Quick Stats - API documentation: https://quickstats.nass.usda.gov/api
# Quick Stats - html encoding: https://www.w3schools.com/tags/ref_urlencode.asp
# 50'000 records is the limit for a single call

import pandas as pd
from datetime import datetime as dt

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
def get_USA_conditions(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=list(range(1800,2050)), cols_subset=[]):
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
    if commodity=='WHEAT, WINTER':
        commodity='WHEAT'
    elif commodity=='WHEAT, SPRING, (EXCL DURUM)':
        commodity='WHEAT'
    elif commodity=='WHEAT, SPRING, DURUM':
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

def get_USA_yields(commodity='CORN', aggregate_level='NATIONAL', state_name=[], years=list(range(1800,2050)), cols_subset=[]):
    """
    simple use:
        us_yields=qs.get_USA_yields(cols_subset=['Value','year'])

    commodity = 'CORN', 'SOYBEANS', 'WHEAT', 'WHEAT, WINTER'
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    
    # Edit inputs to make the download possible (for example necessary to modify commodity for spring/winter wheat)
    if commodity=='CORN':
        dl.short_desc.append(commodity+', GRAIN - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='SOYBEANS':
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='WHEAT':
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='WHEAT, WINTER':
        commodity='WHEAT'
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='WHEAT, SPRING, (EXCL DURUM)':
        commodity='WHEAT'
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')
    elif commodity=='WHEAT, SPRING, DURUM':
        commodity='WHEAT'
        dl.short_desc.append(commodity+' - YIELD, MEASURED IN BU / ACRE')

    dl.years.extend(years)
    dl.commodity_desc.append(commodity)
    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo=fo.set_index('year',drop=False)

    return fo

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
def get_USA_progress(commodity='CORN', progress_var=None, aggregate_level='NATIONAL', state_name=[], years=list(range(1800,2050)), cols_subset=[]):
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
    if commodity=='WHEAT, WINTER':
        commodity='WHEAT'
    elif commodity=='WHEAT, SPRING, (EXCL DURUM)':
        commodity='WHEAT'
    elif commodity=='WHEAT, SPRING, DURUM':
        commodity='WHEAT'

    dl.commodity_desc.append(commodity)
    dl.statisticcat_desc.append('PROGRESS')
    dl.agg_level_desc.append(aggregate_level)
    dl.state_name.extend(state_name)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='week_ending',ascending=True)

    return fo

def get_USA_production(commodity='CORN', aggregate_level='NATIONAL', years=list(range(1800,2050)), cols_subset=[]):
    """
    df_prod=qs.get_QS_production('soybeans', aggregate_level='COUNTY', years=[2017])\n

    commodity = 'CORN', 'SOYBEANS'\n
    aggregate_level = 'NATIONAL', 'STATE', 'COUNTY'
    """    
    commodity=commodity.upper()
    aggregate_level=aggregate_level.upper()

    dl = QS_input()
    dl.source_desc.append('SURVEY')
    dl.years.extend(years)
    dl.commodity_desc.append(commodity)

    if commodity=='CORN':
        dl.short_desc.append(commodity+', GRAIN - PRODUCTION, MEASURED IN BU')
    elif commodity=='SOYBEANS':
        dl.short_desc.append(commodity+' - PRODUCTION, MEASURED IN BU')

    dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"
    dl.agg_level_desc.append(aggregate_level)

    fo=get_data(dl)
    if len(cols_subset)>0: fo = fo[cols_subset]
    fo=fo.sort_values(by='year',ascending=True)
    fo['Value'] = fo['Value'].str.replace(',','').astype(float)

    return fo

def get_ethanol(freq_desc='MONTHLY', years=list(range(1800,2050)), cols_subset=[]):
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

def get_everything( years=list(range(1800,2050))):
    """
    
    """    

    # Main inputs
    if True:
        dl = QS_input()
        dl.source_desc.append('SURVEY')
        dl.domain_desc.append('TOTAL')    
        dl.agg_level_desc.append('NATIONAL')
        dl.years.extend(years)    
        # dl.reference_period_desc.append('YEAR') # This can also be: "YEAR - AUG FORECAST"


    # Downloading part
    if True:
        fo=get_data(dl)    
        fo=fo.sort_values(by='year',ascending=True)
        # fo['Value'] = fo['Value'].str.replace(',','').astype(float)
        print('Downloaded:', years)

    return fo