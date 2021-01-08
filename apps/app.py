#!/usr/bin/env python
"""
Name: Rio Atmadja
Date: January 01, 2021
Description: DSC Milestones (Text Mining on Islamic State Posts) DSC630 & DSC640
"""
import dash 
import pandas as pd 
import numpy as np 
from base64 import urlsafe_b64decode 
import seaborn as sns 
import os 
from sklearn.ensemble import RandomForestClassifier
from nltk import sent_tokenize
import joblib 

# ACME Libraries 

# Dash libraries 
import dash_core_components as doc 
import dash_html_components as html 
import plotly.express as px 
import plotly as plt 
import plotly.graph_objects as go
import dash_bootstrap_components as dbc 
from dash.dependencies import Input, Output
import sqlite3 

# alias 
from typing import Dict, List 
from pandas.core.frame import DataFrame 
from pandas.core.series import Series 

APP_PATH: str = '/var/www/applications/'
csv_files: List[str]  = ['main_posts.csv' , 'embedded_messages.csv' ] 
file_unions: set = set(csv_files) & set(os.listdir(APP_PATH))

if not file_unions:
    raise FileNotFoundError(f"Error: Please provide the following files {' , '.join(csv_files)}")
    
# Load the main posts from the local sqlite 
posts_df: DataFrame = pd.read_csv(os.path.join(APP_PATH,"main_posts.csv"))
posts_df['date_created'] = pd.to_datetime( posts_df['date_created'] ) 
post_sentiment_trends = posts_df.set_index('date_created')['sentiment_polarity'].resample('M').mean().dropna() 
post_numberviews_trends = posts_df.set_index('date_created')['number_views'].resample('M').mean().dropna() 

# Load the embedde posts from the local sqlite 
embedded_messages_df: DataFrame = pd.read_csv(os.path.join( APP_PATH,"embedded_messages.csv"))
embedded_messages_df.index = pd.to_datetime( embedded_messages_df['date_created'] )
embedded_messages_df.drop('date_created', axis=1, inplace=True)

# Unload the ML-models, tfidf-transformers, and countvectorizer 
if not os.path.exists('./random_forest.gz'):
    raise FileNotFoundError("Please provide your pickle file.")

pickle_obj: Dict  = joblib.load('./random_forest.gz') 
rf_clf, tfidf, vect = tuple(pickle_obj.values()) 
colnames: List[str] = ['Analytical', 'Joy', 'Sadness', 'Fear', 'Confident', 'Anger', 'Tentative'] 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, title='TextMining', update_title="Loading ...", external_stylesheets=external_stylesheets)

# make subplots of posts sentiment and number of views 
fig = plt.subplots.make_subplots(rows=3, cols=2, subplot_titles=("Trends in Sentiment Polarity", "Trends in Number of Views", "Frequency of the Tones", "Frequency of the Topics and the Tones", "Monthly trends of the language used in the post. ".title(), "Number of Languages embedded inside the images".title()))
fig.add_trace(go.Scatter(x=post_sentiment_trends.index.tolist(), y=post_sentiment_trends.tolist(), name='Sentiment Polarity'), row=1, col=1)
fig.add_trace(go.Scatter(x=post_numberviews_trends.index.tolist(), y=post_numberviews_trends.tolist(), name='Number of Views'), row=1, col=2)

colors: List = sns.color_palette(palette='ocean_r', n_colors=100)
tones, freq_tones = zip(*posts_df.groupby(['tone_name']).size().sort_values(ascending=False).to_dict().items())
fig.add_trace(go.Bar(x=list(tones), y=list(freq_tones), name='By Tone and Languages', marker=dict(color=colors, coloraxis='coloraxis'), marker_color='crimson' ), row=2, col=1)

# topics by the tones
by_topics: DataFrame = posts_df.groupby(['search_keyword','tone_name']).size().unstack('search_keyword')
by_topics.columns = pd.Series( by_topics.columns).astype(str).apply(lambda topic: topic.replace('_', ' ').title())

for index,topic in enumerate(by_topics.columns.tolist()): 
    fig.add_trace(go.Bar(x=by_topics[topic].sort_values(ascending=False).index.tolist(), y=by_topics[topic].sort_values(ascending=False).tolist(), text=topic, name=topic, showlegend=True), row=2, col=2 )

# monthly frequency of the language used in the post. 
line_colors: List[tuple] = [sns.dark_palette('blue')[0], sns.dark_palette('red')[-1]]
monthly_lang_freq: DataFrame = posts_df.pivot_table(values='tone_name', index='date_created', columns='Post Language', aggfunc='count').interpolate().resample('M').mean().rename({'ar':'Arabic', 'en':'English'}, axis=1).interpolate()
for index,lang in enumerate(monthly_lang_freq.columns.tolist()):
    fig.add_trace(go.Scatter(x=monthly_lang_freq.index.tolist() , y=monthly_lang_freq[lang].tolist(), name=lang, mode='lines+markers', marker_color=line_colors[index]), row=3, col=1)

# Language in embedded messages    
for column in embedded_messages_df.columns.tolist():
    current_df: DataFrame = embedded_messages_df[column].resample('M').mean().interpolate() 
    fig.add_trace(go.Scatter(x=current_df.index.tolist(), y=current_df.tolist(), name=column, mode='lines+markers'), row=3, col=2)

# y-axis labels
fig.update_yaxes(title_text="Sentiment Polarity Scores", row=1, col=1)
fig.update_yaxes(title_text="Number of Views", row=1, col=2)
fig.update_yaxes(title="Number of Posts", row=3, col=1)
fig.update_yaxes(title="Number of Posts", row=3, col=2)

# Plot the trends in tones used: (Embedded Messages and Actual Posts)

posts_data_tones = pd.read_csv("./posts_data_tones.csv")
trends = go.Figure() 
trends_df: DataFrame = posts_data_tones.groupby(['date_created','post_tones']).size().unstack('post_tones') 
trends_df.index = pd.to_datetime(trends_df.index)

colors: List[str] = [ "rgb{}".format(color) for color in sns.color_palette(palette='ocean', n_colors=7) ] 
for index,tone in enumerate(trends_df.columns.tolist()):
    current_trend: DataFrame = trends_df[tone].resample('M').mean().interpolate() 
    trends.add_trace(go.Scatter(x=current_trend.index.tolist() ,y=current_trend.tolist(), name=tone, mode='lines+markers' , line=dict(color=colors[index], dash='dash') ))
trends.update_layout(yaxis_title='Number of Posts', xaxis_title='Date Created', annotations=[dict(xref='paper', yref='paper', xanchor='center', yanchor='top', text='Trends of different types of tones'.title(), x=0.5, y=1.15, font=dict(family='Courier New', size=22) ,showarrow=False)])


# CSS Styles 
row: Dict = {'display': 'flex', 'padding-bottom': '15%'}
column: Dict = {'flex': '30%', 'padding': '5px'}
tab_style: Dict = {'fontWeight': 'bold', 
                   'font-family': 'courier', 
                   'font-size':'150%', 
                   'background-color': '#4445ef'}
        
selected_tab_style: Dict = {'fontWeight': 'bold', 
                   'font-family': 'courier', 
                   'font-size':'150%', 
                   'background-color': '#262728',
                   'border': '9px solid gray'}
            
center_div: Dict = {'display': 'block', 
                    'margin-left': 'auto', 
                    'margin-right': 'auto'
                   }

text_area_style: Dict = {'width': '100%', 
                         'height': '12%',
                         'padding': '12px 20px',
                         'box-sizing': 'border-box',
                         'border': '2px solid #ccc',
                         'border-radius': '4px',
                         'background-color': '#f8f8f8',
                         'resize': 'none'
                        }
        
fig.update_layout(width=1750, height=1750)
app.layout = html.Div(children=[html.H1(children=[html.B('Text Mining and Analysis: Islamic State Recruitment', style={'font-family':'courier', 'fontWeight':'bold', 'font-size':'150%'})] , 
                                        style={'textAlign':'center', 'color': 'black'}), 
                                        doc.Tabs([
                                            doc.Tab(style=tab_style, selected_style=selected_tab_style, label='Project Background', children=[html.Div(children=[html.B(children=[html.H1("Project Bacground", style={'textAlign': 'center', 'color': 'black', 'font-size': '300%', 'font-family': 'courier', 'fontWeight':'bold'})]),
                                                                                                            html.P("""Like many terrorist or state-sponsored organizations, the Islamic state used intimidation, ideology (i.e., promoting the image of holy war), and money to recruit its member.  
                                                                                                                    According to the Wikipedia article, in the early two thousand fourteen, the Islamic State (Daesh) used mainstream platforms such as Twitter, Facebook, and YouTube to promote their organization and recruit fighters by posting shared photos of countless murders, executions, and battlegrounds. 
                                                                                                                    For instance, there were many foreign fighters from countries such as Russia, Tunisia, Jordan, Saudi Arabia. Not all Daesh recruits end up in combat roles. 
                                                                                                                    For instance, women have a support role such as providing first aid, cooking, nursing, etc. """, style={'font-size': '200%', 'color': 'black', 'font-family': 'courier', 'padding-top': '50px', 'fontWeight': 'bold'}
                                                                                                                   ),
                                                                                                            html.P("""Since the large social media platforms such as YouTube, Facebook, and Twitter removed the Islamic State (Daesh) content. 
                                                                                                                      They have chosen to utilize other social media platforms that either protect their content or allow the user to repost the content. 
                                                                                                                      These platforms of choice are Telegram, JustPasteIt, and Surespot. The Islamic State (Daesh) also employed marketing initiatives like "Jihadist Follow Friday", which encourages users to follow new Daesh-related accounts each Friday [2].
                                                                                                                   """, style={'font-size': '200%', 'color': 'black', 'font-family': 'courier', 'padding-top': '50px', 'padding-bottom': '500px', 'fontWeight': 'bold'})

                                                                                                            ]
                                                                                                   )
                                                                                         ]
                                                   ), 
                                            doc.Tab(label='Data Visualizations',selected_style=selected_tab_style, children=doc.Graph(id='page-content',figure=fig, style=center_div), style=tab_style),
                                            doc.Tab(label='Different types of tones used in the posts'.title(), selected_style=selected_tab_style, style=tab_style, children=html.Div(children=[doc.Graph(figure=trends)]) ), 
                                            doc.Tab(label='Network Graphs',selected_style=selected_tab_style, style=tab_style, children=[html.H3(children=[html.B("Network Graph: Relationship between Tones, Keywords, and Word Frequency")], style={'textAlign': 'center', 'color': 'black', 'fontWeight':'bold','padding-top':'2%', 'font-size': '200%', 'font-family':'courier' ,'padding-bottom':'2%'}), html.Div(children=[ html.Div(html.Img(src=app.get_asset_url("keywords_and_tones.jpeg"), style={'width': '100%', 'height': '100%', 'border': '3px solid black'}),style=column), 
                                                                                                                                                                                                                                                                  html.Div(html.Img(src=app.get_asset_url("tones_topics_languages.jpeg"), style={'width': '100%', 'height': '100%', 'border': '3px solid black'}),style=column),  
                                                                                                                                                                                                                                                                  html.Div(html.Img(src=app.get_asset_url("wordfreqs_and_tones.jpeg"), style={'width': '100%', 'height': '100%', 'border': '3px solid black'}),style=column) 
                                                                                                                                                                                                                                                                ], style=row) ]), 
                                            doc.Tab(label='Tones Classification', 
                                                    selected_style=selected_tab_style, 
                                                    style=tab_style, 
                                                    children=html.Div(children=[doc.Textarea(id='textarea-text', value='Please Enter Your Text Here', style={**text_area_style, **{'font-size': '150%', 'fontWeight': 'bold', }}), 
                                                                                html.Div(id='textarea-classification', style={'whitespace': 'pre-line', 'fontWeight': 'bold', 'font-size':'250%'}) 
                                                                               ], style={'padding-top':'15%', 'padding-bottom': '10%'}
                                                                     ) 
                                                    )
                                        ]),
                               ], 
                               style={'background-image':'url(https://wallpapercave.com/wp/wp2071777.jpg)', 
                                      'height': '100%'
                                     }
                    )

@app.callback(Output('textarea-classification', 'children'), 
              Input('textarea-text', 'value')
)
def classify_tones(text):
    
    if not text:
        return "Please Input Your Text Here"

    results: Dict = pd.DataFrame( rf_clf.predict_proba( tfidf.transform(vect.transform(sent_tokenize(text)))), columns=['Analytical', 'Joy', 'Sadness', 'Fear', 'Confident', 'Anger', 'Tentative']).mean(axis=0).sort_values(ascending=False).to_dict()  
    response: str = "" 
    for key,value in results.items(): 
        if key == 'Analytical':
            response += f"\n{key} {chr(129300)}: {value * 100:.2f} % \n\n"

        elif key == 'Joy':
            response += f"\n{key} {chr(128514)}: {value * 100:.2f} % \n\n"

        elif key == 'Confident':
            response += f"{key} {chr(128548)}: {value * 100:.2f} % \n\n"

        elif key == 'Tentative': 
            response += f"{key} {chr(129300)}: {value * 100:.2f} % \n\n"

        elif key == 'Anger':
            response += f"{key} {chr(128574)}: {value * 100:.2f} % \n\n"

        elif key == 'Fear':
            response += f"{key} {chr(128534)}: {value * 100:.2f} % \n\n"

        elif key == 'Sadness': 
            response += f"{key} {chr(128557)}: {value * 100:.2f} % \n\n"

    return response

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="80", threaded=True) #ssl_context="adhoc")
