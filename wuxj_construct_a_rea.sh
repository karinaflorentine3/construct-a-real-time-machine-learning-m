#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Activate virtual environment
source venv/bin/activate

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go

# Load dataset
df = pd.read_csv('machine_learning_data.csv')

# Preprocess data
X = df.drop(['target_variable'], axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Create Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1('Real-time Machine Learning Model Dashboard'),
    html.Div([
        html.P('Select a feature to analyze:'),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': i, 'value': i} for i in X.columns],
            value=X.columns[0]
        )
    ]),
    dcc.Graph(id='feature-graph'),
    html.Div([
        html.P('Model Accuracy:'),
        html.P(id='accuracy-p')
    ]),
    html.Div([
        html.P('Model Classification Report:'),
        html.P(id='report-p')
    ]),
    html.Div([
        html.P('Model Confusion Matrix:'),
        html.P(id='matrix-p')
    ])
])

# Define callback functions
@app.callback(
    Output('feature-graph', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_feature_graph(selected_feature):
    fig = px.histogram(df, x=selected_feature, title='Feature Distribution')
    return fig

@app.callback(
    Output('accuracy-p', 'children'),
    [Input('feature-dropdown', 'value')]
)
def update_accuracy(selected_feature):
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return f'Model Accuracy: {accuracy:.3f}'

@app.callback(
    Output('report-p', 'children'),
    [Input('feature-dropdown', 'value')]
)
def update_report(selected_feature):
    y_pred = rfc.predict(X_test)
    report = classification_report(y_test, y_pred)
    return f'Model Classification Report:\n{report}'

@app.callback(
    Output('matrix-p', 'children'),
    [Input('feature-dropdown', 'value')]
)
def update_matrix(selected_feature):
    y_pred = rfc.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    return f'Model Confusion Matrix:\n{matrix}'

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)