
import os

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
import os,sys
import numpy as np
from plotly import graph_objs as go
from plotly.graph_objs import *
from dash.dependencies import Input, Output, State
import dash_daq as daq

# reduce the number of your dash board is laggy
N_data_show = 11000






dash_data = pd.read_csv('cleaned_data.csv')

'''
#' code for generating cleaned_data if you did not run the ipython notebook first
import geopandas
import descartes
from shapely.geometry import Point, Polygon
from geopandas import GeoDataFrame, GeoSeries
def data_cleaning(input_data):
    dataset = input_data.copy()
    dataset = dataset.rename(columns = {'Trip_type ':'Trip_type'})
    dataset = dataset.dropna(subset=['Trip_type'])
    dataset = dataset.loc[dataset['RateCodeID']!=99,:]
    dataset = dataset.loc[dataset['Trip_distance']!=0,:]
    
    dataset.loc[:,'lpep_pickup_datetime'] = pd.to_datetime(dataset.loc[:,'lpep_pickup_datetime'])
    dataset.loc[:,'pickup_day'] = dataset.loc[:,'lpep_pickup_datetime'].dt.day
    dataset.loc[:,'pickup_hour'] = dataset.loc[:,'lpep_pickup_datetime'].dt.hour
    dataset.loc[:,'Lpep_dropoff_datetime'] = pd.to_datetime(dataset.loc[:,'Lpep_dropoff_datetime'])
    dataset.loc[:,'dropoff_day'] = dataset.loc[:,'Lpep_dropoff_datetime'].dt.day
    dataset.loc[:,'dropoff_hour'] = dataset.loc[:,'Lpep_dropoff_datetime'].dt.hour
    
    dataset.loc[(dataset['pickup_hour']<=4),'pickup_time'] = '0~4'
    dataset.loc[(dataset['pickup_hour']<=10) & (dataset['pickup_hour']>=5),'pickup_time'] = '5~10'
    dataset.loc[(dataset['pickup_hour']<=17) & (dataset['pickup_hour']>=11),'pickup_time'] = '11~17'
    dataset.loc[(dataset['pickup_hour']<=23) & (dataset['pickup_hour']>=18),'pickup_time'] = '18~23'
    
    dataset.loc[:,'pickup_day'] = dataset.loc[:,'lpep_pickup_datetime'].dt.day_name()
    dataset.loc[:,'travel_time'] = (dataset.loc[:,'Lpep_dropoff_datetime'] - dataset.loc[:,'lpep_pickup_datetime']).dt.seconds
    dataset = dataset.loc[dataset['travel_time']!=0,:]
       
    dataset = dataset.loc[dataset['Total_amount']>0,:]
    dataset['tip_perc'] = dataset ['Tip_amount']/ dataset ['Total_amount'].values

    print('adding information for airport')
    threshold = 2.5
    airports = ['LGA','EWR','JFK']
    dataset['Airport'] = 'outside'
    # then we select the data that travel to and from airport from the dataset
    for i in airports:
        # select both pickup location and dropoff location
        judge = dis_filter(dataset['Pickup_longitude'], dataset['Pickup_latitude'], 
                    airport_loc[i], threshold) | \
                dis_filter(dataset['Dropoff_longitude'], dataset['Dropoff_latitude'], 
                    airport_loc[i], threshold)
        dataset.loc[judge,'Airport'] = i
    dataset = convert_borough(dataset)
    return dataset

# coordinate of 3 airport in New Your, maybe is not accurate, but i put the link here for refference
airport_loc = {'JFK' : (-73.778139, 40.641311),  # https://www.distancesto.com/coordinates/us/aeropuerto-de-newark-latitude-longitude/history/67419.html
               'EWR' : (-74.172363, 40.6895314), # https://www.distancesto.com/coordinates/us/newark-liberty-international-latitude-longitude/history/3910.html
               'LGA' : (-73.873966, 40.7769271)} # https://www.distancesto.com/coordinates/us/laguardia-airport-lga-latitude-longitude/history/3898.html
threshold = 2.5
airports = ['LGA','EWR','JFK']

def convert_borough(input_data):
    new_data = input_data.copy()
    print('initialize the pickup locations of taxi trips')
    pickup_loc  = GeoSeries([Point(x, y) for x, y in zip(new_data.Pickup_longitude, new_data.Pickup_latitude)]) 
    print('initialize the drop off locations of taxi trips')
    dropoff_loc = GeoSeries([Point(x, y) for x, y in zip(new_data.Dropoff_longitude, new_data.Dropoff_latitude)])
    new_data.loc[:,'Pickup_boro'] = 'outside'
    new_data.loc[:,'Dropoff_boro'] = 'outside'
    boros = GeoDataFrame.from_file('./Borough Boundaries/geo_export_3aad48b7-65d5-492a-8d4c-820506f10f1c.shp')
    for i in range(5):
        print ('working on pickups in ' + boros.boro_name[i])
        pickup_idx = pickup_loc.within(boros.loc[i,'geometry'].simplify(0.008,False))
        new_data.loc[pickup_idx.values,'Pickup_boro'] = boros.boro_name[i]

        print ('working on dropoffs in ' + boros.boro_name[i])
        dropoff_idx = dropoff_loc.within(boros.loc[i,'geometry'].simplify(0.008,False))
        new_data.loc[dropoff_idx.values,'Dropoff_boro'] = boros.boro_name[i]
    return new_data

def dis_filter(lon1,lat1,loc2,threshold):

    # lon1, lat1 [list] : List of longitude and latitude of first location  
    # loc2 (float,float): Longitude and latitude of the airport
    # thereshold        : Thershold of judging whether the distance is acceptable.
    #                     Higher threshold, more trip will be included as the trip
    #                     to/from airport.
    
    # return: a list of Boolean (True/False) to judge wether the location is 
    #         within the reach of selected airport

    lon1,lat1 = np.radians(lon1), np.radians(lat1)
    lon2,lat2 = np.radians(loc2)
    d_lo = lon2 - lon1
    d_la = lat2 - lat1
    a = np.sin(d_la / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lo / 2)**2
    c = 6373 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))  
    return  c < threshold


def find_airport_data(dataset):
    airport_data = dataset.copy()
    airport_data['Airport'] = np.nan
    # then we select the data that travel to and from airport from the dataset
    for i in airports:
        # select both pickup location and dropoff location
        judge = dis_filter(dataset['Pickup_longitude'], dataset['Pickup_latitude'], 
                    airport_loc[i], threshold) | \
                dis_filter(dataset['Dropoff_longitude'], dataset['Dropoff_latitude'], 
                    airport_loc[i], threshold)

        airport_data.loc[judge,'Airport'] = i
    airport_data = airport_data.dropna(subset=['Airport'])
    print('There are {} trips fulfill the criteria.'.format(np.shape(airport_data)[0]))
    return airport_data


original_data = pd.read_csv('green_tripdata_2015-09.csv',engine='python')
cleaned_data = data_cleaning(original_data)

'''






app = dash.Dash(__name__)
server = app.server
app.title = 'NYC Taxi analysis Dashboard'

# API keys and datasets
mapbox_access_token = 'pk.eyJ1IjoibHVuYXJoZXJvIiwiYSI6ImNrNzlqZW9zNjBjYzkzZW1yOTJlNnBsMGQifQ.JukO3NG44HXGSyz_uTSy3A'



dash_data = dash_data.sample(frac=1)
dash_data['average_speed'] = dash_data['Trip_distance']/dash_data['travel_time']
dash_data['intra_trip'] =  dash_data['Pickup_boro']==dash_data['Dropoff_boro']
weekdays = dash_data.pickup_day.unique()
time = [0,23]
boroughs = dash_data.Pickup_boro.unique()
intra_trip = ['inter','intra','both']

tabs_styles = {
    'height': '44px',
    "border": "white",
    "primary": "gold",
    "background": "cornsilk"  
}
tab_style = {
    'borderBottom': '1px solid #cee0d8',
    'padding': '6px',
    'fontWeight': 'bold',
    'color':'#2c7984',
    'backgroundColor': '#edf7ff'
}

tab_selected_style = {
    'borderTop': '1px solid #cee0d8',
    'borderBottom': '1px solid #cee0d8',
    'backgroundColor': '#8fabbc',
    'color': 'white',
    'padding': '6px',
    'fontWeight': 'bold'
}

headerColor='#434343'
indexcolumncolor='#e9f6f7'
indexcolumncolor0='#e4f2f0'
valuecolumncolor='#fcfffe'
valuecolumncolor2='#f5fcfb'




def create_table(df1):
    '''
    This is a function that convert pandas dataframe into html display format
    '''
    return dash_table.DataTable(
    data=df1.to_dict('records'),
    columns=[
        {'name': i, 'id': i} for i in df1.columns
    ],
    style_cell={'backgroundColor':valuecolumncolor,'color': '#343434','minWidth':'120px',
                'textAlign': 'center','border':'1px solid #dedede','height':'35',
                'font_size':'10'},
    style_data_conditional=[{
        'if': {'row_index': 'odd'},
        'backgroundColor': valuecolumncolor
        }]+[{
        'if': {'column_id': 'PickUp Borough'},
        'backgroundColor': indexcolumncolor,
    }]+[{
        'if': {'column_id': 'Travel'},
        'backgroundColor': indexcolumncolor,
    }],
    style_header={
        'backgroundColor': headerColor,
        'fontWeight': 'bold',
        'color':'white',
        'height':'42',
        'padding': '15px'
    })




def filter_data(dash_data, weekdays, time, dropoff_loc, intra_trip):
    new_df = []
    for day in weekdays:
        filtered = dash_data.loc[dash_data['pickup_day']==day,:]
        new_df.append(filtered)
    filtered = pd.concat(new_df)
    filtered = filtered.loc[(filtered['pickup_hour']>=time[0]) & (filtered['pickup_hour']<time[1]),:]
    new_df = []
    for item in dropoff_loc:
        filtered2 = filtered.loc[filtered['Dropoff_boro']==item,:]
        new_df.append(filtered2)
        
    filtered = pd.concat(new_df)
    if intra_trip == 'inter':
        filtered = filtered.loc[(filtered['intra_trip']==False) ,:]
    elif intra_trip == 'intra':
        filtered = filtered.loc[(filtered['intra_trip']==True)  ,:]
    return filtered



#  Layouts
layout_table = dict(
    autosize=True,
    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
)
layout_table['font-size'] = '12'
layout_table['margin-top'] = '20'

layout_map = dict(

    height=500,
    font=dict(color="#191A1A"),
    titlefont=dict(color="#191A1A", size='14'),
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    plot_bgcolor='#fffcfc',
    paper_bgcolor='#fffcfc',
    legend=dict(font=dict(size=10), orientation='h'),
    title='Taxi Pickup Locations',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lon=-73.91251,
            lat=40.7342
        ),
        zoom=10,
    )
)

# functions
def gen_map(map_data,dropoff_loc):
    # groupby returns a dictionary mapping the values of the first field
    # 'classification' onto a list of record dictionaries with that
    # classification value.
    return {
        "data": [{
                "type": "scattermapbox",
                "lat": list(map_data.loc[map_data['Pickup_boro']==i,'Pickup_latitude']),
                "lon": list(map_data.loc[map_data['Pickup_boro']==i,'Pickup_longitude']),
                'customdata': map_data.loc[map_data['Pickup_boro']==i,:].index,
                "hoverinfo": "text",
                "hovertext": [["Starting time: {} ".format(i)] for i in map_data['lpep_pickup_datetime']],
                "mode": "markers",
                "name": i,              
                "marker": {
                    "size": 6,
                    "opacity": 0.7
                
                }
        } for i in dropoff_loc],
        "layout": layout_map
    }


titles=['pickup_hour',
        'No. trips',
        'Trip_distance',
        'Mean travel_speed (mph)',
        'Number of inter-boundary trips from different boroughs',
        'Percentage of inter-borough trip',
        'Mean tip percentage']

dash_data['inter_trip'] =  dash_data['Pickup_boro']!=dash_data['Dropoff_boro']

layout = lambda i: dict(height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                title= titles[i],
                paper_bgcolor='rgba(240,240,240, 0.001)',
                plot_bgcolor='rgba(250,250,250, 0.5)',
                showlegend=True)

def create_figures(boros=['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island']):
    fig1,fig2,fig3 = go.Figure(layout=layout(1)),go.Figure(layout=layout(2)),go.Figure(layout=layout(3))
    fig4,fig5,fig6 = go.Figure(layout=layout(4)),go.Figure(layout=layout(5)),go.Figure(layout=layout(6)) 
    for i in boros:
        selected = dash_data.loc[dash_data['Pickup_boro']==i,:]
        summary_data = pd.pivot_table(selected.loc[:,['tip_perc','inter_trip','intra_trip','Trip_distance','average_speed','pickup_hour','Dropoff_boro']],
                                            index   = ["pickup_hour"],
                                            aggfunc = {'Dropoff_boro':'count',
                                                        'tip_perc':np.mean,
                                                    'intra_trip' :np.mean,
                                                    'inter_trip' :np.sum,
                                                    'Trip_distance':np.mean,
                                                    'average_speed':np.mean
                                                    },
                                        ).reset_index()
        summary_data['intra_trip'] = 1-summary_data['intra_trip'].values
        summary_data['average_speed'] = np.round(summary_data['average_speed']*3600,1)
        summary_data = summary_data.rename(columns={'Dropoff_boro':'No. trips',
                                                        'average_speed' : 'Mean travel_speed (mph)',
                                                        'tip_perc' : 'Mean tip percentage',
                                                        'intra_trip': 'Percentage of inter-borough trip',
                                                        'inter_trip': 'Number of inter-boundary trips from different boroughs'})
        summary_data.loc[summary_data['Mean travel_speed (mph)']>150,'Mean travel_speed (mph)'] = np.nan
        fig1.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,1], mode='lines', name=i))
        fig2.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,2], mode='lines', name=i))
        fig3.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,3], mode='lines', name=i))
        fig4.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,4], mode='lines', name=i))
        fig5.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,5], mode='lines', name=i))
        fig6.add_trace(go.Scatter(x=summary_data.pickup_hour, y=summary_data.iloc[:,6], mode='lines', name=i))
    return fig1,fig2,fig3,fig4,fig5,fig6









app.layout = html.Div(
    html.Div([
        html.Div('',style={'width': '2%','height':'30px'}),
        html.H1('NYC Green Taxi Analysis Dashboard',style={'font-size':'40px','textAlign':'center','color':'#2c7984','marginTop':'10px'}),
        html.Div('',style={'width': '2%','height':'30px'}),
        html.H4('Inter/intra borough data graphs',style={'font-size':'25px','color':'#505050','marginLeft':'10%'}),
        html.Div('',style={'width': '2%','height':'10px'}),
        html.Div([
            html.Div('The Staten Island is miss-leading because of limited number of travels',style={'font-size':'15px','color':'#505050'}),
            html.Div([
                html.H6('Show Staten Island Data',style={'font-size':'17px','color':'#505050','width': '25%','display': 'inline-block'}),
                html.Div([daq.BooleanSwitch(
                    id = 'hide_Staten',
                    on = False
                )],style={'width': '10%','display': 'inline-block'})
            ]),
        ],style={'width': '85%','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle'}),
        dcc.Loading(html.Div(id = '6_plots')),
        html.Div('',style={'width': '2%','height':'20px'}),
        html.H2('Data filtering and selection',style={'font-size':'30px','textAlign':'center','color':'#2c7984','marginTop':'10px'}),
        # Selectors
        html.Div([
            html.Div([
                html.H6('Choose Dropoff Borough',style={'font-size':'17px','color':'#505050'}),
                dcc.Dropdown(
                    id = 'select_borough',
                    options=[
                        {'label': i, 'value': i} for i in boroughs
                    ],
                    multi=True,
                    value = ['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island']
                ),  
            ],style={'width': '30%','display': 'inline-block'}),

            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),

            html.Div([
                html.H6('Whether cross borough',style={'font-size':'17px','color':'#505050'}),
                dcc.Dropdown(
                    id = 'select_trip',
                    options=[
                        {'label': i+' trips', 'value': i} for i in intra_trip
                    ],
                    value='both'
                ),   
            ],style={'width': '30%','display': 'inline-block'}),

            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),

            html.Div([
                html.H6('Choose weekday',style={'font-size':'17px','color':'#505050'}),
                dcc.Dropdown(
                    id = 'select_weekday',
                    options=[
                        {'label': i, 'value': i} for i in weekdays
                    ],
                    multi=True,
                    value=weekdays
                ),  
            ],style={'width': '30%','display': 'inline-block'}),
        ],style={'width': '85%','textAlign':'center','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle'}),
        html.Div('',style={'width': '2%','height':'20px'}),
        html.Div([
            html.Div([html.H5('Choose Pickup time',style={'color':'#343434'})],style={'width': '15%', 'display': 'inline-block'}),
            html.Div([
                dcc.RangeSlider(
                    id='Select_time',
                    marks={i: {'label':'{}:00'.format(i),'style':{'font-size':'14px','font-weight':'600'}} for i in range(25)}, 
                    min=0, 
                    max=24, 
                    value=[0,24]
                )
            ],style={'width': '65%', 'display': 'inline-block'}),
             html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Button('Update', id='Update_chart',style={'width': '15%','display': 'inline-block'}),

        ],style={'width': '85%','textAlign':'center','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle'}),
        html.Div('',style={'width': '2%','height':'70px'}),
        
        html.Div(id='summary_table',style={'width': '80%','textAlign':'center','marginLeft':'10%','marginRight':'10%','vertical-align': 'middle'}),
        # map
        html.Div([
            html.Div([
                dcc.Loading(dcc.Graph(id = 'NYC_map', hoverData={'points': [{'customdata':-1}]}))
            ],style={'width': '50%','display': 'inline-block','vertical-align': 'middle'}),
            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Div(id='test1',style={'display': 'inline-block','width': '40%','vertical-align': 'middle'})
        ],style={'width': '85%','textAlign':'center','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle','display': 'inline-block'}),
        html.Div('',style={'width': '2%','height':'70px'}),


    ]),style={'min-width':'1800px'}
)




@app.callback([Output('NYC_map', 'figure'),
               Output('summary_table', 'children')],
              [Input('Update_chart', 'n_clicks')],
              [State('select_borough', 'value'),
               State('select_trip', 'value'),
               State('select_weekday', 'value'),
               State('Select_time', 'value')])
def loading_dataset(n_clicks, dropoff_loc, intra_trip, weekdays, time):
    new_data = filter_data(dash_data, weekdays, time, dropoff_loc, intra_trip)
    new_data = new_data.sample(frac=1)
    summary_data = pd.pivot_table(new_data.loc[:,['travel_time','Total_amount','Pickup_boro','Dropoff_boro','tip_perc','average_speed','intra_trip']],
                                    index   = ["Dropoff_boro"],
                                    aggfunc = {'Total_amount':np.mean,
                                               'travel_time' :np.mean,
                                               'average_speed':np.mean,
                                               'tip_perc':np.mean,
                                               'Pickup_boro':'count',
                                              },
                                ).reset_index()
    summary_data = summary_data.rename(columns={'Pickup_boro':'No. trips',
                                'Dropoff_boro' : 'Dropoff Borough',
                                                    'Total_amount': 'Mean Fees',
                                                    'travel_time' : 'Mean travel_time (mins)',
                                                    'tip_perc' : 'Mean tip percentage',
                                                    'average_speed': 'Average_speed (mph)'})
    summary_data.iloc[:,2] = np.round(summary_data.iloc[:,2].values,2)
    summary_data.iloc[:,3] = np.round(summary_data.iloc[:,3].values*3600,2)
    summary_data.iloc[:,4] = np.round(summary_data.iloc[:,4].values,3)
    summary_data.iloc[:,5] = np.round(summary_data.iloc[:,5].values/60,1)
    return gen_map(new_data.iloc[:N_data_show,:],dropoff_loc) , create_table(summary_data)

@app.callback(
    Output('6_plots', 'children'),
    [Input('hide_Staten', 'on')])
def hide_staten(choice):
    if choice==False:
        boros=['Queens', 'Brooklyn', 'Manhattan', 'Bronx']
    else:
        boros=['Queens', 'Brooklyn', 'Manhattan', 'Bronx', 'Staten Island']
    
    fig1,fig2,fig3,fig4,fig5,fig6 = create_figures(boros)
    return html.Div([html.Div([
            html.Div([dcc.Graph(figure=fig1)],style={'width': '32%','display': 'inline-block'}),
            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig2)],style={'width': '32%','display': 'inline-block'}),
            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig3)],style={'width': '32%','display': 'inline-block'}),
        ],style={'width': '85%','textAlign':'center','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle'}),
        html.Div('',style={'width': '2%','height':'20px'}),
        html.Div([
            html.Div([dcc.Graph(figure=fig4)],style={'width': '32%','display': 'inline-block'}),
            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig5)],style={'width': '32%','display': 'inline-block'}),
            html.Div('',style={'width': '2%','height':'30px','display': 'inline-block'}),
            html.Div([dcc.Graph(figure=fig6)],style={'width': '32%','display': 'inline-block'}),
        ],style={'width': '85%','textAlign':'center','marginLeft':'7.5%','marginRight':'7.5%','vertical-align': 'middle'})])

@app.callback(
    Output('test1', 'children'),
    [Input('NYC_map', 'hoverData')])
def update_hover1(hoverData):
    hoverData = hoverData['points'][0]['customdata']
    if hoverData == -1:
        data = pd.DataFrame(np.array([
                        ['pickup location','N/A'],
                        ['dropoff location','N/A'], 
                        ['Trip time (mins)','N/A'], 
                        ['Trip length','N/A'],
                        ['Fees','N/A'],
                        ['Tips','N/A']
                        ]),
                    columns=['Travel', ''])
    else:
        data = pd.DataFrame(np.array([
                        ['pickup location',dash_data.loc[hoverData,'Pickup_boro']],
                        ['dropoff location',dash_data.loc[hoverData,'Dropoff_boro']], 
                        ['Trip time (mins)',np.round(dash_data.loc[hoverData,'travel_time']/60,1)], 
                        ['Trip length',dash_data.loc[hoverData,'Trip_distance']],
                        ['Fees',dash_data.loc[hoverData,'Total_amount']],
                        ['Tips',dash_data.loc[hoverData,'Tip_amount']]
                        ]),
                    columns=['Travel', ''])
    return create_table(data)










if __name__ == '__main__':
    app.run_server(debug=True)

