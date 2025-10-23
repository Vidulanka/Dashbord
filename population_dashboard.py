import os
import sys
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load and preprocess the data
BASE_DIR = os.path.dirname(__file__)
CSV_NAME = 'us-population-2010-2019.csv'
csv_path = os.path.join(BASE_DIR, CSV_NAME)

if not os.path.exists(csv_path):
    # Try to find a CSV in the same folder as a fallback
    candidates = [f for f in os.listdir(BASE_DIR) if f.lower().endswith('.csv')]
    if candidates:
        csv_path = os.path.join(BASE_DIR, candidates[0])
        print(f"Using found CSV: {candidates[0]}")
    else:
        print(f"ERROR: Could not find '{CSV_NAME}' in {BASE_DIR}. Please add the CSV file and try again.")
        sys.exit(1)

df = pd.read_csv(csv_path)

# Clean the data - remove commas and convert to numeric
years = [str(year) for year in range(2010, 2020)]
for year in years:
    # Some CSVs include commas for thousands separators - remove them safely
    if year in df.columns:
        df[year] = df[year].astype(str).str.replace(',', '', regex=False).replace({'': np.nan}).astype(float)
    else:
        # If a year column is missing, create it with NaNs so downstream code won't fail
        df[year] = np.nan

# Create a melted version for easier plotting
# Normalize state column name (accept variations like 'state' or 'States')
state_col = next((c for c in df.columns if c.lower() == 'states' or c.lower().startswith('state')), None)
if state_col is None:
    raise KeyError(f"Could not find a state column in CSV. Available columns: {list(df.columns)}")
if state_col != 'states':
    df = df.rename(columns={state_col: 'states'})

# Ensure we have an 'id' column (2-letter state abbreviations) for the choropleth.
# If not present, try to map from full state names.
if 'id' not in df.columns:
    # Mapping of full state names to USPS abbreviations
    STATE_ABBR = {
        'Alabama': 'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA','Colorado':'CO',
        'Connecticut':'CT','Delaware':'DE','District of Columbia':'DC','Florida':'FL','Georgia':'GA','Hawaii':'HI',
        'Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
        'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS',
        'Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
        'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK',
        'Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
        'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
        'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
    }
    df['id'] = df['states'].map(STATE_ABBR)

df_melted = df.melt(id_vars=['states', 'id'], 
                    value_vars=years,
                    var_name='year', 
                    value_name='population')

df_melted['year'] = df_melted['year'].astype(int)

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("US Population Dashboard (2010-2019)", className='title', style={'textAlign': 'center', 'marginBottom': 20}),

    # Controls row
    html.Div([
        html.Div([
            html.Div([
                html.Label("Select State(s):"),
                dcc.Dropdown(
                    id='state-dropdown',
                    options=[{'label': state, 'value': state} for state in df['states']],
                    value=['California', 'Texas', 'Florida', 'New York'],
                    multi=True,
                    style={'width': '100%'}
                )
            ], className='card')
        ], style={'width': '48%', 'display': 'inline-block'}, className='controls-row'),

        html.Div([
            html.Div([
                html.Label("Select Year Range:"),
                dcc.RangeSlider(
                    id='year-slider',
                    min=2010,
                    max=2019,
                    step=1,
                    marks={i: str(i) for i in range(2010, 2020)},
                    value=[2010, 2019]
                )
            ], className='card')
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}, className='controls-row')
    ], style={'marginBottom': 20}),

    # Graphs row 1
    html.Div([
        html.Div([html.Div(dcc.Graph(id='population-trend'), className='card')], style={'width': '100%', 'marginBottom': 20}),
    ]),

    # Graphs row 2
    html.Div([
        html.Div([html.Div(dcc.Graph(id='population-map'), className='card')], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([html.Div(dcc.Graph(id='population-change'), className='card')], style={'width': '48%', 'display': 'inline-block', 'float': 'right', 'verticalAlign': 'top'})
    ]),

    # Summary statistics
    html.Div([
        html.Div([html.H3("Summary Statistics", style={'marginTop': 10}), html.Div(id='summary-stats')], className='card')
    ], style={'marginTop': 20})
], style={'padding': 20})

# Callbacks for interactivity
@app.callback(
    [Output('population-trend', 'figure'),
     Output('population-map', 'figure'),
     Output('population-change', 'figure'),
     Output('summary-stats', 'children')],
    [Input('state-dropdown', 'value'),
     Input('year-slider', 'value')]
)
def update_dashboard(selected_states, year_range):
    # Defensive: if user clears selection, use all states
    if not selected_states:
        selected_states = df['states'].unique().tolist()

    # Filter data based on selections
    filtered_df = df_melted[
        (df_melted['states'].isin(selected_states)) & 
        (df_melted['year'] >= int(year_range[0])) & 
        (df_melted['year'] <= int(year_range[1]))
    ]
    
    # Population trend line chart
    trend_fig = px.line(
        filtered_df,
        x='year',
        y='population',
        color='states',
        title='Population Trends Over Time',
        labels={'population': 'Population', 'year': 'Year'},
        template='plotly_dark'
    )
    trend_fig.update_traces(mode='lines+markers')
    trend_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(title='State', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Population map for the latest selected year
    latest_year = int(year_range[1])
    map_data = df_melted[df_melted['year'] == latest_year].copy()
    
    # Use the 2-letter state id for choropleth locations. If 'id' is missing for some rows,
    # those states will be omitted from the map.
    map_fig = px.choropleth(
        map_data.dropna(subset=['id']),
        locations='id',
        locationmode='USA-states',
        color='population',
        scope='usa',
        title=f'Population by State - {latest_year}',
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name='states',
        hover_data={'population': ':,', 'id': False},
        template='plotly_dark'
    )
    map_fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Population change bar chart
    pop_change_data = []
    for state in selected_states:
        state_data = df_melted[df_melted['states'] == state]
        start_vals = state_data[state_data['year'] == int(year_range[0])]['population'].values
        end_vals = state_data[state_data['year'] == int(year_range[1])]['population'].values
        if start_vals.size == 0 or end_vals.size == 0 or pd.isna(start_vals[0]) or start_vals[0] == 0:
            # Skip states with missing data for the selected years
            continue
        start_pop = float(start_vals[0])
        end_pop = float(end_vals[0])
        change = ((end_pop - start_pop) / start_pop) * 100
        pop_change_data.append({
            'state': state,
            'population_change': change
        })
    
    change_df = pd.DataFrame(pop_change_data)
    change_fig = px.bar(
        change_df,
        x='state',
        y='population_change',
        title=f'Population Change (%) {year_range[0]}-{year_range[1]}',
        labels={'population_change': 'Population Change (%)', 'state': 'State'},
        color='population_change',
        color_continuous_scale=px.colors.diverging.RdYlGn,
        template='plotly_dark'
    )
    change_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    
    # Summary statistics
    latest_data = df_melted[df_melted['year'] == latest_year]
    if latest_data.empty:
        total_population = 0
        avg_population = 0
        max_state = {'states': 'N/A', 'population': 0}
        min_state = {'states': 'N/A', 'population': 0}
    else:
        total_population = latest_data['population'].sum()
        avg_population = latest_data['population'].mean()
        # Use .iloc to avoid errors if idxmax/min fail
        max_row = latest_data.loc[latest_data['population'].idxmax()]
        min_row = latest_data.loc[latest_data['population'].idxmin()]
        max_state = max_row
        min_state = min_row
    
    stats = html.Div([
        html.P(f"Total US Population ({latest_year}): {total_population:,.0f}"),
        html.P(f"Average State Population: {avg_population:,.0f}"),
        html.P(f"Most Populous State: {max_state['states']} ({max_state['population']:,.0f})"),
        html.P(f"Least Populous State: {min_state['states']} ({min_state['population']:,.0f})")
    ])
    
    return trend_fig, map_fig, change_fig, stats

# Run the app
if __name__ == '__main__':
    # Dash v3+ removed app.run_server in favor of app.run
    # Keep same behavior: debug mode and port configurable
    app.run(debug=True, port=8050)