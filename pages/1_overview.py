import streamlit as st
import streamlit_shadcn_ui as ui
from local_components import card_container
from utils.helper import get_column_counts
from utils.streamlit import (
    load_bar_chart, 
    load_column_selectbox, 
    load_line_chart, 
    load_empty_chart,
    )

AGGREGATION_MAP = {
    '1 hour': '1h',
    '1 day': '1D',
    '7 days': '7D',
    '30 days': '30D'
}


filtered_data = st.session_state.filtered_data

#--- VISUALIZATION ---
#--- METRICS CARD ---
col1, col2, col3, col4 = st.columns(4)

with col1: 
    # ui.metric_card("Total Count", f"{st.session_state.total_requests:,}", "Requests")
    ui.metric_card("Total Count", f"{len(filtered_data):,}", "Requests")
with col2: 
    ui.metric_card("Total Expired", f"{st.session_state.expired_count:,}", "Requests")
with col3: 
    ui.metric_card("Average Count", f"{st.session_state.avg_requests_per_day:.2f}", "Request/Day")
with col4:
    ui.metric_card("Overdue Count", f"{st.session_state.filtered_data_overdue.shape[0]:}", "Requests")

#--- LINE CHART ---
st.header("Request Trend", help="Visualize trend over time")
agg_col = list(AGGREGATION_MAP.keys())

col1, _, col2 = st.columns([0.4, 0.1, 0.5])
with col1:
    chart_option = load_column_selectbox("line_chart_options", title="Filtered By")
    
with col2:
    aggregation_option = ui.tabs(
        options=agg_col,
        default_value=agg_col[0]
    )
    
resample_freq = AGGREGATION_MAP[aggregation_option]
#Filtering Data based on selected
grouped_data = filtered_data.set_index('Timestamp')
if chart_option is not None:
    grouped_data = grouped_data.groupby([chart_option], as_index=True)
grouped_data = grouped_data.resample(resample_freq).size()    

if filtered_data.empty :
    fig = load_empty_chart()
else:
    fig = load_line_chart(grouped_data)
        
with card_container(key='chart_main'):
    st.plotly_chart(fig, use_container_width=True)

#--- BAR CHART ---
if len(filtered_data) > 0:
    st.header('Request Breakdown', help="Visualize number of request based on selected group")
    col1, col2 = st.columns(2)
    with col1:
        tab_left = ui.tabs(options=['Company Group', 'Company', 'Department'], default_value='Company')
            
        with card_container(key='chart_left'):
            column_name = 'Company Code - Name' if tab_left == 'Company' else tab_left
            counts_df = get_column_counts(filtered_data, column_name)
            fig = load_bar_chart(counts_df, 
                                 xaxis_title=column_name.replace(" - ", " "),
                                 yaxis_title="Number of Requests",
                                 visible_range=7,
                                 enable_sorting=True,)
            st.plotly_chart(fig, use_container_width=True)    
        
    with col2:
        tab_right = ui.tabs(options=['Processed By','Request', 'Request Type'], default_value='Request')
        
        with card_container(key='chart_right'):
            column_name = 'Request Type Group' if tab_right == 'Request' else tab_right
            counts_df = get_column_counts(filtered_data, column_name)
            fig = load_bar_chart(counts_df, 
                                 xaxis_title=column_name,
                                 yaxis_title="Number of Requests",
                                 visible_range=7,
                                 enable_sorting=True)
            st.plotly_chart(fig, use_container_width=True)

#--- Overdue Request ---

st.header(
    "Overdue Requests",
    help="These are requests MDM has received but not yet taken; duration is in work-day minutes"
)

st.dataframe(st.session_state.filtered_data_overdue[[
    'Request Number',
    'Company Code - Name',
    'Request Type',
    'Timestamp MDM Received',
    'Total Task',
    'Processed By',
    'Taken Date'
]], use_container_width=True)

#--- Track error ---
with st.expander("Error Rows"):
    mdm_error_rows = st.session_state.mdm_error_rows
    if (mdm_error_rows.shape[0] > 0):
        st.dataframe(mdm_error_rows)
