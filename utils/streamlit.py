import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import streamlit_shadcn_ui as ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from utils.helper import wrap_label, get_date_range, handle_remove_outliers


SELECT_COLUMN = {
    'Company Group': 'Company Group',
    'Company': 'Company Code - Name',
    'Department': 'Department',
    'Request Type Group': 'Request Type Group',
    'Request Type': 'Request Type',
    'MDM': 'Processed By'
}

SELECT_METRIC = {
    'Request Duration (Hr)': ('Processed Duration (Hours)', 'mean'),
    'Task Duration (min)': ('Processed Duration per Task', 'mean'),
    'Request Count': ('Request Number', 'count'),
    'Task Count': ('Total Task', 'sum')
}

TEMPLATE = 'plotly_white'
COLOR = ['#005F60']  # Blue Teal color

# --- Table ---
def load_overview_table(data, column):
    metrics = [metric[0] for metric in SELECT_METRIC.values()]  # Get metric names
    overview_table = data.groupby(column)[metrics].agg(
        {metric[0]: metric[1] 
         for metric in SELECT_METRIC.values()}).reset_index()
    overview_table.set_index(column, inplace=True)
    
    # Rename columns in overview_table based on SELECT_METRIC keys
    overview_table.rename(columns={metric[0]: key for key, metric in SELECT_METRIC.items()}, inplace=True)

    # Adjust values to two decimal points if they are floats
    # overview_table = overview_table.applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)
    overview_table = overview_table.style.format("{:.2f}")
    st.dataframe(
        overview_table.highlight_max(axis=0),
        use_container_width=True
        )  

def load_metric_table(data, column, group, metric):
    # Create a pivot table for individual durations by company, filling empty cells with zero
    pivot_data = data.pivot_table(index=column, 
                                  columns=group, 
                                  values=metric[0],  # Use the metric name from the tuple
                                  aggfunc=metric[1],  # Use the aggregation method from the tuple
                                  fill_value=0)  # Name for the subtotal

    # Format the pivot table to display maximum value with two decimal places
    formatted_pivot_data = pivot_data.style.format("{:.2f}")
    st.dataframe(
        formatted_pivot_data.highlight_max(axis=0),
        use_container_width=True)

# --- Multiselect ---
def load_multiselect(data, title):
    return st.sidebar.multiselect(
        title, 
        options=sorted(list(data.dropna().unique())), 
        )

# --- Selectbox ---
def load_column_selectbox(key, default_index=None, title=""):
    options = list(SELECT_COLUMN.keys())
    selected_option = st.selectbox(title, 
                                   key=key,
                                   label_visibility="visible" if title!="" else "collapsed", 
                                   options=options, 
                                   index=default_index,
                                   placeholder="Select grouped by")
    
    if not selected_option:
        return None
    
    return SELECT_COLUMN[selected_option]

def load_metric_selectbox(key, default_index=None, title="", hide_count=False):
    options = list(SELECT_METRIC.keys())
    
    if hide_count:
        options = [option for option in options 
                   if SELECT_METRIC[option][1] != 'count']
    selected_option = st.selectbox(title, 
                                   key=key,
                                   label_visibility="visible" if title!="" else "collapsed", 
                                   options=options, 
                                   index=default_index,
                                   placeholder="Select Metric")
    
    if not selected_option:
        return None
    
    return SELECT_METRIC[selected_option]

# --- DATE PICKER ---

def load_range_date_picker(range):    
    date_range = st.date_input("Pick a date range", range)
    
    if len(date_range) > 1:
        start_date, end_date = sorted(pd.to_datetime(date_range))
    elif len(date_range) == 1:
        start_date = end_date = pd.to_datetime(date_range[0])
        
    start_date = start_date.replace(hour=0, minute=0, second=1)
    end_date = end_date.replace(hour=23, minute=59, second=59)
    return start_date, end_date

def load_date_picker():
    options = ["Today", "Yesterday", "This Week", "Last 30 Days", "Custom Date"]
    
    cols = st.columns(3)

    with cols[0]:
        selection = st.selectbox("Select date range", options, index=4)

        if selection and selection != "Custom Date":
            start_date, end_date = get_date_range(selection)
            return start_date, end_date
        
        #Load custom range date picker
        if selection == "Custom Date":
            range = get_date_range(selection)
            with cols[1]:
                start_date, end_date = load_range_date_picker(range)
                return start_date, end_date

# --- CHART ---

def load_bar_chart(
    data, 
    xaxis_title=None, 
    yaxis_title=None,
    title=None, 
    max_label_length=15,
    vertical=True,
    chart_height=450,  # Parameter for fixed chart height
    chart_width=None,   # Parameter for fixed chart width
    visible_range=5,    # Number of items to display at a time
    enable_sorting=False  # Parameter to enable sorting buttons
):
    if data.empty:
        return None
    
    data.sort_values(ascending=True, inplace=True)
    # Apply line breaks to labels
    data.index = data.index.map(lambda x: wrap_label(x, max_label_length))
    
    # Prepare the figure
    fig = go.Figure()

    if enable_sorting:
        # Define buttons for ascending and descending sorting
        buttons = [
            dict(
                label="Highest",
                method="update",
                args=[{
                    "x": [data.iloc[-visible_range:].values] if vertical else [data.iloc[-visible_range:].index],
                    "y": [data.iloc[-visible_range:].index] if vertical else [data.iloc[-visible_range:].values],
                    "text": [data.iloc[-visible_range:].values] if vertical else [data.iloc[-visible_range:].index]
                }]
            ),
            dict(
                label="Lowest",
                method="update",
                args=[{
                    "x": [data.iloc[:visible_range].sort_values(ascending=False).values] if vertical else [data.iloc[:visible_range].sort_values(ascending=False).index],
                    "y": [data.iloc[:visible_range].sort_values(ascending=False).index] if vertical else [data.iloc[:visible_range].sort_values(ascending=False).values],
                    "text": [data.iloc[:visible_range].sort_values(ascending=False).values] if vertical else [data.iloc[:visible_range].sort_values(ascending=False).index]
                }]
            )
        ]

        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="right", x=0, y=1.15, showactive=True, buttons=buttons)]
        )

    data = data.tail(visible_range) if vertical else data

    # Add the bar trace with text labels
    fig.add_trace(go.Bar(
        x=data.values if vertical else data.index,
        y=data.index if vertical else data.values,
        orientation='h' if vertical else 'v',
        text=data.values,
        texttemplate="%{value:.2f}",
        textposition='auto',
        insidetextanchor='start' if vertical else 'end',
        hovertemplate="%{y}: %{x:.2f}<extra></extra>" if vertical else "%{x}: %{y:.2f}<extra></extra>",
        marker_color=COLOR[0]  # Use Blue Teal color
    ))

    # Update layout
    fig.update_layout(
        title=title if title else "",
        xaxis_title=xaxis_title if xaxis_title else None,
        yaxis_title=yaxis_title if yaxis_title else None,
        height=chart_height,
        width=chart_width,
        margin=dict(l=10, r=10, t=0, b=10),
        hovermode="y unified" if vertical else "x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.25,
            xanchor="center",
            x=0.4
        )
    )

    if not visible_range:
        return fig
    
    return fig

def load_pie_chart(
    data,
    title=None,
    labels=None,
):
    # Handle different input data types
    if isinstance(data, pd.DataFrame):
        # Flatten data values and calculate counts
        values = data.values.flatten()
        value_counts = pd.Series(values).value_counts()
        
        # Prepare labels if not provided
        if labels is None:
            labels = [f"{index} ({count})" for index, count in zip(value_counts.index, value_counts.values)]
        
        # Use counts for the pie chart
        values = value_counts.values
    else:
        # If not DataFrame, assume it's a Series or list-like object
        value_counts = pd.Series(data).value_counts()
        if labels is None:
            labels = [f"{index} ({count})" for index, count in zip(value_counts.index, value_counts.values)]
        values = value_counts.values
    
    # Create pie chart
    fig = px.pie(
        values=values,
        names=labels,
    )
    
    # Update layout
    fig.update_layout(
        title=title if title else "",
        showlegend=True,
        margin=dict(l=10, r=10, t=100, b=10),
        legend=dict(
            orientation="v",      # Vertical legend
            yanchor="middle",     # Anchor the legend to the middle
            y=0.5,
            xanchor="left",       # Anchor to the left
            x=-0                  # Position legend outside to the right
        ),
    )
    
    # Update trace settings for better visualization
    fig.update_traces(
        textinfo='percent+value+label',  # Show percentage, value, and label
        hovertemplate="%{label}<br>Value: %{value}<br>Percent: %{percent}<extra></extra>",
        insidetextorientation='radial'  # Orient text radially
    )
    
    return fig


def load_line_chart(
    data,
    xaxis_title=None, 
    yaxis_title=None, 
    title=None
):
    # Check if data has MultiIndex
    if isinstance(data.index, pd.MultiIndex):
        # Handle MultiIndex case
        fig = px.line(
            x=data.index.get_level_values(1),
            y=data.values,
            color=data.index.get_level_values(0),
            # color_discrete_sequence=[COLOR[0]]  # Use Blue Teal color
        )
    else:
        # Handle SingleIndex case
        fig = px.line(
            x=data.index,
            y=data.values,
            color_discrete_sequence=[COLOR[0]]  # Use Blue Teal color
        )
    
    # Update layout to add x-axis and y-axis titles if provided
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(l=10, r=15, t=10, b=10),
        title=title if title else "",
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",     # Anchor the legend to the top
            y=1.25,            # Position it below the plot
            xanchor="center",  # Center the legend horizontally
            x=0.4              # Center the legend relative to the chart
        ),
    )
    
    # Update hover template
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}<extra></extra>")

    return fig


def load_empty_chart(text:str=None):
    default_text = "No data to show."
    
    fig = go.Figure()
    fig.add_annotation(
        text= text if text else default_text,
        xref="paper", 
        yref="paper",
        x=0.5,
        y=0.5, 
        showarrow=False,
        font=dict(size=20, color=COLOR[0])  # Use Blue Teal color
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    return fig

def load_histogram(
    data, 
    xaxis_title=None, 
    yaxis_title=None,
    title=None, 
    bin_size=None,         # Size of the bins for the histogram
    chart_height=450,     # Parameter for fixed chart height
    chart_width=None,     # Parameter for fixed chart width
    visible_range=None    # Number of bins to display at a time
):
    if data.empty:
        return None
    
    # Create the histogram with optional bin_size
    fig = px.histogram(
        data,
        nbins=bin_size if bin_size else '100%',  # Set the number of bins or use default
        title=title if title else None,
        template='plotly_white',
        color_discrete_sequence=[COLOR[0]]  # Use Blue Teal color
    )
    
    fig.update_layout(
        xaxis_title=xaxis_title if xaxis_title else None, 
        yaxis_title=yaxis_title if yaxis_title else None,
        height=chart_height,
        width=chart_width,
        margin=dict(l=10, r=20, t=20, b=20)
    )
    
    # Enable scrollbars by limiting the visible range of bins
    if visible_range:
        x_range = fig.layout.xaxis.range
        step = (x_range[1] - x_range[0]) / len(data)
        start = max(x_range[0], x_range[1] - (step * visible_range))
        end = min(x_range[1], x_range[0] + (step * visible_range))
        fig.update_xaxes(
            range=[start, end],  # Show only a portion of the data
            fixedrange=False  # Allow scrolling
        )
    
    # Update hover template
    fig.update_traces(hovertemplate="Count: %{y}<br>Value: %{x:.2f}<extra></extra>")
    
    return fig

def load_bar_chart_with_trend(
    bar_data, 
    trend_data, 
    title="", 
    xaxis_title="", 
    yaxis_title="", 
    y_axis_title_trend="Duration (Hours)",
    group_by=None,
    show_trend=True
):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if group_by is None:
        # Add bar chart
        fig.add_trace(
            go.Bar(x=bar_data.index, y=bar_data.values, name="Daily Count", text=bar_data.values, textposition='auto',
                   hovertemplate="%{x}: %{y}<extra></extra>", marker_color=COLOR[0]),  # Use Blue Teal color
            secondary_y=False,
        )
        
        # Add trend line if show_trend is True
        if show_trend:
            fig.add_trace(
                go.Scatter(x=trend_data.index, y=trend_data.values, name=xaxis_title, line=dict(color='#FF9912'), 
                           text=trend_data.values.round(2), textposition='top center', mode='lines+markers+text', 
                           textfont=dict(color='#FF9912'), hovertemplate="%{x}: %{y:.2f}<extra></extra>"),
                secondary_y=True,
            )
    else:
        # Add grouped bar chart
        for i, person in enumerate(bar_data.columns):
            fig.add_trace(
                go.Bar(x=bar_data.index, y=bar_data[person], name=person,
                       hovertemplate=f"{person}: %{{y}}<extra></extra>"),
                secondary_y=False,
            )
            if show_trend:
                fig.add_trace(
                    go.Scatter(x=trend_data.index, y=trend_data[person], name=f"{person}", 
                               hovertemplate=f"{person}: %{{y:.2f}}<extra></extra>"),
                    secondary_y=True,
                )
                
        # Add text annotations for the total value of each stack
        total_values = bar_data.sum(axis=1)
        fig.add_trace(
            go.Scatter(
                x=bar_data.index,
                y=total_values,
                mode='text',
                text=total_values,
                textposition='top center',
                showlegend=False,
                hoverinfo='skip'
            ),
            secondary_y=False
        )
    
    # Set x-axis title
    fig.update_xaxes(title_text=xaxis_title)
    
    # Set y-axes titles
    fig.update_yaxes(title_text=yaxis_title, secondary_y=False)
    if show_trend:
        fig.update_yaxes(title_text=y_axis_title_trend, secondary_y=True)
    
    # Update layout
    fig.update_layout(
        title_text=title,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.25,
            xanchor="center",
            x=0.4
        ),
        barmode='stack' if group_by else 'relative'
    )
    
    return fig


def load_boxplot(
    data,
    x_column,
    y_column,
    xaxis_title=None,
    yaxis_title=None,
    title=None,
    chart_height=450,
    chart_width=None,
    visible_range=10,
    remove_outliers=True,
    max_label_length=15,
    aggregation_method='median'  # New parameter for aggregation method
):
    if data.empty:
        return None

    # Sort the data by specified aggregation method
    grouped = data.groupby(x_column)[y_column].agg(aggregation_method).sort_values(ascending=False)  # Updated line
    sorted_categories = grouped.index[:visible_range]

    # Filter data to include only the top categories
    filtered_data = data[data[x_column].isin(sorted_categories)]

    fig = go.Figure()

    if remove_outliers:
        # Remove outliers from the data
        filtered_data = handle_remove_outliers(filtered_data, y_column)

    fig.add_trace(go.Box(
        x=filtered_data[y_column],
        y=filtered_data[x_column].apply(lambda x: wrap_label(x, max_label_length)),
        boxmean=True,  # Show mean as a dashed line
        marker_color=COLOR[0],  # Use Blue Teal color
        line_color=COLOR[0],    # Use Blue Teal color
        fillcolor='rgba(0,128,128,0.5)',  # Use Blue Teal color with transparency
        hovertemplate=(
            "<b>%{y}</b><br><br>"
            "Middle value (Median): %{median}<br>"
            "Average value (Mean): %{mean}<br>"
            "25% of data falls below: %{q1}<br>"
            "75% of data falls below: %{q3}<br>"
            "Lowest value: %{lowerfence}<br>"
            "Highest value: %{upperfence}<br>"
            "<extra></extra>"
        ),
        orientation='h'  # Set orientation to horizontal
    ))

    fig.update_layout(
        title=title if title else "",
        xaxis_title=yaxis_title if yaxis_title else y_column,
        yaxis_title=xaxis_title if xaxis_title else x_column,
        height=chart_height,
        width=chart_width,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    return fig