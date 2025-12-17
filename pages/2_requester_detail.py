import streamlit_shadcn_ui as ui
import streamlit as st
from local_components import card_container
from utils.streamlit import load_bar_chart, load_empty_chart, load_histogram


filtered_data = st.session_state.filtered_data

if len(filtered_data) > 0:

    #-- Creation Rate
    st.header('Request Completion Rate')
    first_tab = ui.tabs(['Bar Chart', 'Histogram'], default_value='Bar Chart', key='first_tab')
    col1, col2 = st.columns([0.65, 0.35], vertical_alignment='center')
    with col1:
        with card_container(key='chart'):
            if first_tab == 'Bar Chart':    
                fig = load_bar_chart(st.session_state.requester_creation_duration_count, vertical=False, visible_range=None)
            else:
                fig = load_histogram(st.session_state.requester_filtered_data['Creation Duration (Minutes)'])

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sub1, sub2 = st.columns(2, gap='small')
        with sub1:
            ui.metric_card("Completed Count", f"{st.session_state.completed_count}", "Requests")
            ui.metric_card("Expired Requests", f"{st.session_state.expired_count}", "Requests")
            ui.metric_card("Unique Request", f"{st.session_state.unique_requests}", "Requesters")
        with sub2:
            ui.metric_card("Creation Rate", f"{st.session_state.completion_rate_percentage:.2f}", "% Created")
            ui.metric_card("Creation Duration", f"{st.session_state.requester_avg_creation_duration:.2f}", "Min/Created")
            
            
    #-- Approval Rate --
    st.header('Approval Completion Rate')
    second_tab = ui.tabs(['Bar Chart', 'Histogram'], default_value='Bar Chart', key='sceond_tab')
    col1, col2 = st.columns([0.65, 0.35], vertical_alignment='center')
    with col1:
        with card_container(key='chart2'):
            if second_tab == 'Bar Chart':    
                fig = load_bar_chart(st.session_state.approver_duration_count, vertical=False, visible_range=None)
            else:
                fig = load_histogram(st.session_state.approver_filtered_data['Approval Duration (Minutes)'])
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        sub1, sub2 = st.columns(2, gap='small')
        with sub1:
            ui.metric_card("Approved Count", f"{st.session_state.approved_count}", "Requests")
            ui.metric_card("Rejected Count", f"{st.session_state.rejected_count}", "Requests")
            ui.metric_card("No Approver", f"{st.session_state.no_approver_count}", "Requests")
        with sub2:
            ui.metric_card("Approval Rate", f"{st.session_state.approval_rate_percentage:.2f}", "% Approved")
            ui.metric_card("Approval Duration", f"{st.session_state.approver_avg_approval_duration:.2f}", "Min/Approve")
            
else:
    st.plotly_chart(load_empty_chart(), use_container_width=True)