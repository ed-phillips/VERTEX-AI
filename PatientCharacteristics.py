import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import IsaricDraw as idw
import pandas as pd
import plotly.graph_objs as go
import pycountry
from dash.dependencies import Input, Output, State
import dash
import numpy as np
import IsaricDraw as idw
import IsaricAnalytics as ia
import getREDCapData as getRC
import redcap_config as rc_config


suffix='pc'

############################################
#REDCap elements
site_mapping=rc_config.site_mapping
redcap_api_key=rc_config.redcap_api_key
redcap_url=rc_config.redcap_url



############################################
############################################
## Data reading and initial proccesing 
############################################
############################################

countries = [{'label': country.name, 'value': country.alpha_3} for country in pycountry.countries]
sections=getRC.getDataSections(redcap_api_key)
vari_list=getRC.getVariableList(redcap_api_key,['dates','demog','comor','daily','outco','labs','vital','adsym','inter','treat'])
df_map=getRC.get_REDCAP_Single_DB(redcap_url, redcap_api_key,site_mapping,vari_list)
df_map_count=df_map[['country_iso','slider_country','usubjid']].groupby(['country_iso','slider_country']).count().reset_index()
unique_countries = df_map[['slider_country', 'country_iso']].drop_duplicates().sort_values(by='slider_country')
country_dropdown_options=[]
for uniq_county in range(len(unique_countries)):
    name_country=unique_countries['slider_country'].iloc[uniq_county]
    code_country=unique_countries['country_iso'].iloc[uniq_county]
    country_dropdown_options.append({'label': name_country, 'value': code_country})
bins = [0, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101]
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '91-95', '96-100']
df_map['age_group'] = pd.cut(df_map['age'], bins=bins, labels=labels, right=False)
df_map['mapped_outcome'] = df_map['outcome']


def visuals_creation(df_map):
    ############################################
    #get Variable type
    ############################################
    dd=getRC.getDataDictionary(redcap_api_key)        
    variables_binary,variables_date,variables_number,variables_freeText,variables_units,variables_categoricas=getRC.getVaribleType(dd)   
    correct_names=dd[['field_name','field_label']]#Variable and label dictionary
    
    color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}
    df_age_gender=df_map[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
    df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)
    pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='age_gender_pyramid_chart')


    proportions_symptoms, set_data_symptoms = ia.get_proportions(df_map,'symptoms')
    freq_chart_sympt = idw.frequency_chart(proportions_symptoms, title='Frequency of signs and symptoms on presentation')
    upset_plot_sympt = idw.upset(set_data_symptoms, title='Frequency of combinations of the five most common signs or symptoms')

    proportions_comor, set_data_comor= ia.get_proportions(df_map,'comorbidities')
    freq_chart_comor = idw.frequency_chart(proportions_comor, title='Frequency of comorbidities on presentation')
    upset_plot_comor = idw.upset(set_data_comor, title='Frequency of combinations of the five most common comorbidities')

    #descriptive = ia.descriptive_table(ia.obtain_variables(df_map, 'symptoms'))
    descriptive = ia.descriptive_table(df_map,correct_names,variables_binary,variables_number)
    fig_table_symp=idw.table(descriptive)

    symptoms_columns = [col for col in df_map.columns if col.startswith('adsym_')]
    df1=df_map[symptoms_columns]
 
    comor_columns = [col for col in df_map.columns if col.startswith('comor_')]
    df2 = df_map[comor_columns]

    mapper = {'Yes':1,'No':0}
    df1 = df1.replace(mapper)
    df2 = df2.replace(mapper)                     
    heatmap=idw.heatmap(df1,df2,"Title",graph_id="Heatmap1")
    return fig_table_symp,pyramid_chart,freq_chart_sympt,upset_plot_sympt,freq_chart_comor,upset_plot_comor,heatmap


############################################
############################################
## Modal creation
############################################
############################################


def create_modal():
    ############################################
    #Modal Intructions
    ############################################
    linegraph_about= html.Div([
        html.Div("1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster)"),
        html.Br(),
        html.Div("2. Change datasets using the dropdown (country selections are remembered)"),
        html.Br(),
        html.Div("3. Hover mouse on chart for tooltip data "),
        html.Br(),
        html.Div("4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it."),
        html.Br(),
        html.Div("5. Toggle selected countries on/off by clicking on the legend (far right)"),
        html.Br(),
        html.Div("6. Download button will export all countries and available years for the selected dataset"),    
    ])   
    linegraph_instructions = html.Div([
        html.Div("1. Select/remove countries using the dropdown (type directly into the dropdowns to search faster)"),
        html.Br(),
        html.Div("2. Change datasets using the dropdown (country selections are remembered)"),
        html.Br(),
        html.Div("3. Hover mouse on chart for tooltip data "),
        html.Br(),
        html.Div("4. Zoom-in with lasso-select (left-click-drag on a section of the chart). To reset the chart, double-click on it."),
        html.Br(),
        html.Div("5. Toggle selected countries on/off by clicking on the legend (far right)"),
        html.Br(),
        html.Div("6. Download button will export all countries and available years for the selected dataset"),    
    ])   


    fig_table_symp,pyramid_chart,freq_chart_sympt,upset_plot_sympt,freq_chart_comor,upset_plot_comor,heatmap=visuals_creation(df_map)
    


    '''
    color_map = {'Discharge': '#00C26F', 'Censored': '#FFF500', 'Death': '#DF0069'}
    df_age_gender=df_map[['age_group','usubjid','mapped_outcome','slider_sex']].groupby(['age_group','mapped_outcome','slider_sex']).count().reset_index()
    df_age_gender.rename(columns={'slider_sex': 'side', 'mapped_outcome': 'stack_group', 'usubjid': 'value', 'age_group': 'y_axis'}, inplace=True)
    pyramid_chart = idw.dual_stack_pyramid(df_age_gender, base_color_map=color_map, graph_id='age_gender_pyramid_chart')


    proportions_symptoms, set_data_symptoms = ia.get_proportions(df_map,'symptoms')
    freq_chart_sympt = idw.frequency_chart(proportions_symptoms, title='Frequency of signs and symptoms on presentation')
    upset_plot_sympt = idw.upset(set_data_symptoms, title='Frequency of combinations of the five most common signs or symptoms')

    proportions_comor, set_data_comor= ia.get_proportions(df_map,'comorbidities')
    freq_chart_comor = idw.frequency_chart(proportions_comor, title='Frequency of comorbidities on presentation')
    upset_plot_comor = idw.upset(set_data_comor, title='Frequency of combinations of the five most common comorbidities')

    #descriptive = ia.descriptive_table(ia.obtain_variables(df_map, 'symptoms'))
    descriptive = ia.descriptive_table(df_map,correct_names,variables_binary,variables_number)
    fig_table_symp=idw.table(descriptive)

    symptoms_columns = [col for col in df_map.columns if col.startswith('adsym_')]
    df1=df_map[symptoms_columns]
 
    comor_columns = [col for col in df_map.columns if col.startswith('comor_')]
    df2 = df_map[comor_columns]

    mapper = {'Yes':1,'No':0}
    df1 = df1.replace(mapper)
    df2 = df2.replace(mapper)                     
    heatmap=idw.heatmap(df1,df2,"Title",graph_id="Heatmap1")
    '''


    
    #cumulative_chart = idw.cumulative_bar_chart(df_epiweek, title='Cumulative Patient Outcomes by Timepoint', base_color_map=color_map, graph_id='my-cumulative-chart')
    np.random.seed(0)


    modal = [
        dbc.ModalHeader(html.H3("Clinical Features", id="line-graph-modal-title", style={"fontSize": "2vmin", "fontWeight": "bold"})),  

        dbc.ModalBody([
            dbc.Accordion([
                dbc.AccordionItem(
                    title="Filters and Controls",  
                    children=[idw.filters_controls(suffix,country_dropdown_options)]
                ),                
                dbc.AccordionItem(
                    title="Insights",  
                    children=[
                        dbc.Tabs([
                            dbc.Tab(dbc.Row([dbc.Col([fig_table_symp],id='table_symo')]), label='Descriptive table'),
                            dbc.Tab(dbc.Row([dbc.Col(pyramid_chart,id='pyramid-chart-col')]), label='Age and Sex'),
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart_sympt,id='freqSympt_chart')]), label='Signs and symptoms on presentation: Frequency'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot_sympt,id='upsetSympt_chart')]), label='Signs and symptoms on presentation:Intersections'),
                            #dbc.Tab(dbc.Row([dbc.Col(boxplot_graph,id='boxplot_graph-col')]), label='Length of hospital stay by age group'),
                            dbc.Tab(dbc.Row([dbc.Col(freq_chart_comor,id='freqcomor_chart')]), label='Comorbidities on presentation: Frequency'),
                            dbc.Tab(dbc.Row([dbc.Col(upset_plot_comor,id='upsetcomor_chart')]), label='Comorbidities on presentation:Intersections'),
                            dbc.Tab(dbc.Row([dbc.Col(heatmap,id='heatmap_chart')]), label='Comorbidities and symptoms'),
                            
                        ])
                    ]
                )
            ])
        ], style={ 'overflowY': 'auto','minHeight': '75vh','maxHeight': '75vh'}),

        idw.ModalFooter(suffix,linegraph_instructions,linegraph_about)


    ]
    return modal    


############################################
############################################
## Callbacks
############################################
############################################
def register_callbacks(app, suffix):
    @app.callback(
        [Output(f'country-checkboxes_{suffix}', 'value'),
         Output(f'country-selectall_{suffix}', 'options'),
         Output(f'country-selectall_{suffix}', 'value')],
        [Input(f'country-selectall_{suffix}', 'value'),
         Input(f'country-checkboxes_{suffix}', 'value')],
        [State(f'country-checkboxes_{suffix}', 'options')]
    )
    def update_country_selection(select_all_value, selected_countries, all_countries_options):
        ctx = dash.callback_context

        if not ctx.triggered:
            # Initial load, no input has triggered the callback yet
            return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == f'country-selectall_{suffix}':
            if 'all' in select_all_value:
                # "Select all" (now "Unselect all") is checked
                return [[option['value'] for option in all_countries_options], [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # "Unselect all" is unchecked
                return [[], [{'label': 'Select all', 'value': 'all'}], []]

        elif trigger_id == f'country-checkboxes_{suffix}':
            if len(selected_countries) == len(all_countries_options):
                # All countries are selected manually
                return [selected_countries, [{'label': 'Unselect all', 'value': 'all'}], ['all']]
            else:
                # Some countries are deselected
                return [selected_countries, [{'label': 'Select all', 'value': 'all'}], []]

        return [selected_countries, [{'label': 'Select all', 'value': 'all'}], select_all_value]

    @app.callback(
        Output(f"country-fade_{suffix}", "is_in"),
        [Input(f"country-display_{suffix}", "n_clicks")],
        [State(f"country-fade_{suffix}", "is_in")]
    )
    def toggle_fade(n_clicks, is_in):
        if n_clicks:
            return not is_in
        return is_in

    @app.callback(
        Output(f'country-display_{suffix}', 'children'),
        [Input(f'country-checkboxes_{suffix}', 'value')],
        [State(f'country-checkboxes_{suffix}', 'options')]
    )
    def update_country_display(selected_values, all_options):
        if not selected_values:
            return "Country:"

        # Create a dictionary to map values to labels
        value_label_map = {option['value']: option['label'] for option in all_options}

        # Build the display string
        selected_labels = [value_label_map[val] for val in selected_values if val in value_label_map]
        display_text = ", ".join(selected_labels)

        if len(display_text) > 20:  # Adjust character limit as needed
            return f"Country: {selected_labels[0]}, +{len(selected_labels) - 1} more..."
        else:
            return f"Country: {display_text}"


    ############################################
    ############################################
    ## Specific Callbacks
    ## Modify outputs
    ############################################
    ############################################

    @app.callback(
        [Output('table_symo', 'children'),
         Output('pyramid-chart-col', 'children'),
         Output('freqSympt_chart', 'children'),
         Output('upsetSympt_chart', 'children'),
         Output('freqcomor_chart', 'children'),
         Output('upsetcomor_chart', 'children'),
         Output('heatmap_chart', 'children')],
        [Input(f'submit-button_{suffix}', 'n_clicks')],
        [State(f'gender-checkboxes_{suffix}', 'value'),
         State(f'age-slider_{suffix}', 'value'),
         State(f'outcome-checkboxes_{suffix}', 'value'),
         State(f'country-checkboxes_{suffix}', 'value')]
    )
    def update_figures(click, genders, age_range, outcomes, countries):
        filtered_df = df_map[
                        (df_map['slider_sex'].isin(genders))& 
                        (df_map['age'] >= age_range[0]) & 
                        (df_map['age'] <= age_range[1]) & 
                        (df_map['outcome'].isin(outcomes)) &
                        (df_map['country_iso'].isin(countries)) ]
        print(len(filtered_df))

        if filtered_df.empty:

            return None

        fig_table_symp,pyramid_chart,freq_chart_sympt,upset_plot_sympt,freq_chart_comor,upset_plot_comor,heatmap=visuals_creation(filtered_df)
        return [fig_table_symp,pyramid_chart,freq_chart_sympt,upset_plot_sympt,freq_chart_comor,upset_plot_comor,heatmap]

