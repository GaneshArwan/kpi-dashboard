import streamlit as st
import streamlit_shadcn_ui as ui
from local_components import card_container
import pandas as pd
from utils.helper import make_summary
from utils.streamlit import (
    load_bar_chart, 
    load_bar_chart_with_trend, 
    load_empty_chart, 
    load_column_selectbox, 
    load_metric_selectbox,
    load_overview_table,
    load_metric_table,
    load_boxplot,
    load_pie_chart
    )

AGGREGATION_MAP = {
    '30 minutes': '30min',
    '1 hour': '1h',
    '1 day': '1D'
}

filtered_data = st.session_state.mdm_filtered_data 

#--- VISUALIZATION ---
#-- METRIC --
col1, col2, col3, col4 = st.columns(4, gap='small', vertical_alignment='top')
with col1:
    ui.metric_card('Completed Count', f"{st.session_state.mdm_completed_count:}", description="Requests")    
    ui.metric_card('Completion Rate', f"{st.session_state.mdm_completed_rate_percentage:.2f}", description="Percent (%)")
    ui.metric_card('Processed Duration', f"{st.session_state.mdm_average_processed_duration:.2f}", description="Hr/Request")    
    
with col2:
    ui.metric_card('Partial Reject Count', f"{st.session_state.mdm_partial_rejected_count:}", description="Requests")    
    ui.metric_card('Partial Reject Rate', f"{st.session_state.mdm_partially_rejected_rate_percentage:.2f}", description="Percent (%)")
    ui.metric_card('Total Task', f"{st.session_state.mdm_total_task:.0f}", description="Tasks")
        
with col3:
    ui.metric_card('Rejected Count', f"{st.session_state.mdm_rejected_count:}", description="Requests")
    ui.metric_card('Rejected Rate', f"{st.session_state.mdm_rejected_rate_percentage:.2f}", description="Percent (%)")
    ui.metric_card('Average Task', f"{st.session_state.mdm_avg_task_per_request:.2f}", description="Tasks/Request")
    
with col4:
    ui.metric_card('Send Back Count', f"{st.session_state.mdm_send_back_count:}", description="Requests")
    ui.metric_card('Send Back Rate', f"{st.session_state.mdm_send_back_rate_percentage:.2f}", description="Percent (%)")
    ui.metric_card("On-Time Execution Rate", f"{(filtered_data['KPI Pass'].sum() / filtered_data['KPI Pass'].count()) * 100:.2f}", "Percent (%)")


#--- Summary Table ---
summary_df = make_summary(filtered_data)
with st.expander("Summary Table per MDM"):
    st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

 # --- BOX PLOT ---   
col1, col2, col3 = st.columns(3)
with col1: group_by = load_column_selectbox("box_plot_option", 0, 'Filtered By')
with col2: metric = load_metric_selectbox("agg_method", 0, hide_count=True, title='Select Metrics')
with col3: hide_outliers = st.checkbox("Hide Outliers", value=True)
# group_by = ui.tabs(options=['Request Type Group', 'Request Type', 'Processed By'], default_value='Request Type')
with card_container(key='chart_box_plot'):
    # Convert minutes to hours
    filtered_data['Processed Duration (Hours)'] = filtered_data['Processed Duration (Minutes)'] / 60
    
    if filtered_data.empty:
        fig = load_empty_chart()
    else:
        fig = load_boxplot(
            filtered_data,
            x_column=group_by,
            y_column=metric[0],
            xaxis_title=group_by,
            yaxis_title=metric[0],
            remove_outliers=hide_outliers,
            aggregation_method=metric[1]
        )
    st.plotly_chart(fig, use_container_width=True)

#-- Trend Chart --
if not filtered_data.empty:
    st.header("Request Processing Trend", help="MDM Incoming Request (Bar Chart) and Average Duration (Trend Line)")

    col1, col2, col3 = st.columns(3)

    with col1: 
        chart_option = load_column_selectbox("processing_trend", title="Filtered By")
    with col2:
        metric_option = load_metric_selectbox("metric_chart", 
                                              default_index=0,
                                              hide_count=True,
                                              title="Select Metrics")
    with col3:
        show_trend = st.checkbox("Show Duration Trend", value=True)

    # Filtering Data based on selected
    grouped_data = filtered_data.set_index('Processed Date')
        
    # Resample and count for bar chart
    bar_data = grouped_data.resample('1D').size()
    # Resample and calculate average duration for trend line
    # trend_data = grouped_data[metric_option[0]].resample('1D').agg(metric_option[1]) / 60  # Convert to hours
    trend_data = grouped_data[metric_option[0]].resample('1D').agg(metric_option[1])   # Convert to hours
    non_empty_dates = bar_data[bar_data > 0].index
    bar_data = bar_data.loc[non_empty_dates]
    trend_data = trend_data.loc[non_empty_dates]

    if chart_option is not None:
        bar_data = grouped_data.groupby([pd.Grouper(freq='D'), chart_option]).size().unstack(fill_value=0)
        trend_data = grouped_data.groupby([pd.Grouper(freq='D'), chart_option])[metric_option[0]].agg(metric_option[1]).unstack(fill_value=0)

    if filtered_data.empty:
        fig = load_empty_chart()
    else:
        fig = load_bar_chart_with_trend(bar_data, trend_data, 
                                        group_by= chart_option, 
                                        show_trend=show_trend,
                                        xaxis_title=metric_option[0])

    with card_container(key='chart_trend_chart'):
        st.plotly_chart(fig, use_container_width=True)   
        
    #-- BAR CHART--
    
    st.header("Request and Task Duration")
    col1, col2, col3, = st.columns(3)
    with col1: request_group_box = load_column_selectbox("request_group_box", default_index=0, title="Filtered By")
    
    col1, col2 = st.columns(2)
    with col1:
        with card_container(key='chart_bar_chart_col_one'):
            request_by_group = filtered_data.groupby(request_group_box)['Processed Duration (Minutes)'].mean() / 60
            fig = load_bar_chart(request_by_group,
                                xaxis_title="Average Duration per Req (Hours)",
                                yaxis_title=request_group_box,
                                visible_range=7, 
                                enable_sorting=True)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        with card_container(key='chart_bar_chart_col_two'):
            request_by_group = filtered_data.groupby(request_group_box)['Processed Duration per Task'].mean() / 60
            fig = load_bar_chart(request_by_group,
                                xaxis_title="Average Duration per Task (Hours)",
                                yaxis_title=request_group_box,
                                visible_range=7, 
                                enable_sorting=True)
            st.plotly_chart(fig, use_container_width=True)


# --- Service-Level Agreement (SLA) MDM ---
st.header("Service-Level Agreement (SLA)")
day_value_counts = filtered_data['Processed Duration (Day)'].value_counts()
col1, col2 = st.columns([0.8, 0.3])
with col1:
    with card_container(key='pie_chart'):
        fig = load_pie_chart(filtered_data['Processed Duration (Day)'],
                            labels = day_value_counts.index)
        st.plotly_chart(fig, use_container_width=True)
with col2:
    with card_container(key='SLA Count Data'):
        st.table(day_value_counts)
        

# --- Bar Chart SLA
st.header("Performance Breakdown", help="Fail to Achieve SLA (Left) and Status Rejected (Right)")

# 1) Let user pick any column to group by
sla_group_box = load_column_selectbox(
    title="Group SLA By",
    default_index=0,
    key="sla_group_box"
)

# 2) Define your SLA threshold (you could also make this a number_input if you want)
sla_threshold_days = 1

col1_sla, col2_sla = st.columns([0.5, 0.5])

# Failed-SLA chart
with col1_sla:
    with card_container(key='Failed to Pass SLA'):
        failed_sla = filtered_data[
            filtered_data['Processed Duration (Day)'] > sla_threshold_days
        ]

        # count per selected group
        failed_by_group = (
            failed_sla[sla_group_box]
            .value_counts()
            .nlargest(5)
        )
        
        if failed_by_group.empty:
            fig = load_empty_chart()
        else:
            fig = load_bar_chart(
                failed_by_group,
                xaxis_title="Above SLA Count ",
                yaxis_title=sla_group_box,
                visible_range=5,
                enable_sorting=True
            )

        st.plotly_chart(fig, use_container_width=True)

# Rejected-requests chart
with col2_sla:
    with card_container(key='Rejected by SLA Group'):
        rejected = filtered_data[
            filtered_data['Process Status'] == 'Rejected'
        ]
        rejected_by_group = (
            rejected[sla_group_box]
            .value_counts()
            .nlargest(5)
        )
        if rejected_by_group.empty:
            fig = load_empty_chart()
        else :
            fig = load_bar_chart(
                rejected_by_group,
                xaxis_title="Rejected Request Count",
                yaxis_title=sla_group_box,
                visible_range=5,
                enable_sorting=True
            )
        st.plotly_chart(fig, use_container_width=True, 
                        key="Rejected Request Count"
                        )

with st.expander("Performance Data Detail"):
    st.dataframe(failed_sla[[
        'Request Number',
        'Company Code - Name',
        'Request Type',
        'Timestamp MDM Received',
        'Processed By',
        'Process Status',
        'Processed Date',
        'Total Task',
        'Processed Duration (Day)'
    ]])

# --- COMPARSION TABLE ---
st.header("Comparison Table")
col1, col2, col3 = st.columns(3)
with col1 : selected_col = load_column_selectbox(
                                    default_index=0, 
                                    title="Select Column",
                                    key="select_column")
with col2 : selected_group = load_column_selectbox(
                                    default_index=1, 
                                    title="Select Group",
                                    key="select_group")
with col3 : selected_metrics = load_metric_selectbox(default_index=0, 
                                    title="Select Metric",
                                    key="select_metric")

load_metric_table(
                filtered_data,
                selected_col, 
                selected_group,
                selected_metrics)

load_overview_table(filtered_data,
                    selected_col)
# -- Data Overview ---
with st.expander("Data Overview"):
    st.dataframe(filtered_data) 
