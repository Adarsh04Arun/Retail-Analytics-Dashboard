# -*- coding: utf-8 -*-
"""
Enhanced Online Retail Analytics Dashboard with Streamlit
"""
import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import json

# Plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Other viz
import seaborn as sns
import matplotlib.pyplot as plt

# Runtime Configuration Parameters for Matplotlib
plt.rcParams['font.family'] = 'Verdana'
plt.style.use('ggplot')

# Warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For Customer Segmentation
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

# Constants
DATA_FILE = 'OnlineRetail.csv'

# Custom color palette
COLOR_PALETTE = px.colors.qualitative.Vivid
BG_COLOR = "#f0f2f6"
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"

@st.cache_data
def load_data():
    """Load and cache the retail data"""
    return pd.read_csv(DATA_FILE)

def generate_co_occurrence(data):
    """Generate product recommendations based on co-occurrence"""
    try:
        # Group products by invoice
        transactions = data.groupby('InvoiceNo')['Description'].apply(list).reset_index()
        
        # Create co-occurrence dictionary
        co_occurrence = {}
        for products in transactions['Description']:
            for i in range(len(products)):
                if products[i] not in co_occurrence:
                    co_occurrence[products[i]] = {}
                
                for j in range(len(products)):
                    if i != j:
                        if products[j] not in co_occurrence[products[i]]:
                            co_occurrence[products[i]][products[j]] = 0
                        co_occurrence[products[i]][products[j]] += 1
        
        # Sort and get top recommendations
        item_sets = {}
        for product in co_occurrence:
            sorted_items = sorted(co_occurrence[product].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
            item_sets[product] = [item[0] for item in sorted_items]
            
        return item_sets
    
    except Exception as e:
        st.warning(f"Could not generate co-occurrence: {str(e)}")
        return {}

def choose_country(country="all", data=None):
    """Filter data by country"""
    if country == "all":
        return data.copy()
    temp_df = data.loc[data["Country"] == country].copy()
    temp_df.reset_index(drop=True, inplace=True)
    return temp_df

def cluster_plot(data_frame):
    """Create 3D cluster plot"""
    fig = px.scatter_3d(data_frame, 
                       x='Recency', y='Frequency', z='Monetary',
                       color='Clusters', opacity=0.8, 
                       width=800, height=700,
                       color_discrete_sequence=COLOR_PALETTE,
                       template="plotly_white",
                       hover_name='CustomerID',
                       title="Customer Segments Visualization")
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency',
            zaxis_title='Monetary Value ($)'
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def perform_kmeans(rfm_new):
    """Perform K-means clustering on RFM data"""
    scaler = StandardScaler()
    rfm_scaled = rfm_new[['Recency','Frequency','Monetary','RFM_Score']].copy()
    rfm_scaled = scaler.fit_transform(rfm_scaled)
    rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Recency','Frequency','Monetary','RFM_Score'])
    
    kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=50, verbose=0)
    kmeans.fit(rfm_scaled)
    
    rfm_new['Clusters'] = kmeans.labels_
    return rfm_new

def plot_pcts(df, string, title=""):
    """Create pie chart visualization"""
    fig = go.Figure(data=[go.Pie(labels=df.index,
                                values=df[string],
                                hole=.3,
                                marker_colors=COLOR_PALETTE)])
    fig.update_layout(
        showlegend=True,
        height=400,
        margin={'l': 20, 'r': 20, 't': 40, 'b': 20},
        title={
            'text': title,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def group_sales_quantity(df, feature):
    """Group sales data by specified feature"""
    return df[[feature, 'Quantity', 'Sales Revenue']].groupby([feature]).sum()\
           .sort_values(by='Sales Revenue', ascending=False).reset_index()

def setup_rfm_analysis(country_df):
    """Setup RFM analysis for the selected country"""
    country_df = country_df.copy()
    country_df['InvoiceDate'] = pd.to_datetime(country_df['InvoiceDate'])
    ref_date = country_df['InvoiceDate'].max() + timedelta(days=1)
    
    # Remove 'Guest Customer' using .loc
    country_df = country_df.loc[country_df['CustomerID'] != "Guest Customer"].copy()
    
    # Aggregating over CustomerID
    rfm_new = country_df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': lambda x: x.nunique(),
        'Sales Revenue': lambda x: x.sum()
    }).copy()
    
    # Calculate quantiles
    rfm_new.columns = ['Recency', 'Frequency', 'Monetary']
    rfm_new["R"] = pd.qcut(rfm_new['Recency'].rank(method="first"), 4, labels=[4, 3, 2, 1])
    rfm_new["F"] = pd.qcut(rfm_new['Frequency'].rank(method="first"), 4, labels=[1, 2, 3, 4])
    rfm_new["M"] = pd.qcut(rfm_new['Monetary'].rank(method="first"), 4, labels=[1, 2, 3, 4])
    
    # Calculate RFM Score
    rfm_new['RFM_Score'] = (rfm_new['R'].astype(int) + 
                            rfm_new['F'].astype(int) + 
                            rfm_new['M'].astype(int))
    
    return rfm_new.reset_index()

def product_recommendation_page(retail_data):
    """Render the product recommendation page with co-occurrence based recommendations"""
    st.title("🛍️ Product Recommendation Engine")
    st.markdown("Discover products that are frequently purchased together using market basket analysis.")
    
    # Custom container styling
    st.markdown(
        f"""
        <style>
            .recommendation-box {{
                padding: 15px;
                border-radius: 10px;
                background-color: {BG_COLOR};
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Country selection with better UI
    country_list = retail_data['Country'].unique().tolist()
    selected_country = st.selectbox(
        '🌍 Choose Country:', 
        country_list,
        help="Select a country to analyze purchase patterns"
    )
    country_data = choose_country(selected_country, retail_data)
    
    # Display data in an expandable card
    with st.expander("📊 View Country Data", expanded=False):
        st.dataframe(country_data.head(100))
    
    # Generate recommendations section
    st.subheader("🔍 Generate Product Recommendations")
    st.markdown("Select a product to see what other products customers frequently purchase with it.")
    
    # Two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Generate recommendations button with custom styling
        if st.button(
            "✨ Analyze Purchase Patterns", 
            help="Click to analyze product co-occurrence patterns",
            use_container_width=True
        ):
            st.session_state.generate_recs = True
    
    if st.session_state.get('generate_recs', False):
        with st.spinner("🔍 Analyzing purchase patterns..."):
            item_sets = generate_co_occurrence(country_data)
            
            # Product selection in a nice card
            with st.container():
                st.markdown("### 🎯 Select a Product")
                product_catalog = country_data['Description'].unique().tolist()
                selected_product = st.selectbox(
                    'Choose a product:', 
                    product_catalog,
                    key="product_select"
                )
                
                # Display recommendations in a styled box
                if selected_product in item_sets and item_sets[selected_product]:
                    st.markdown("### 🛒 People Also Bought...")
                    for i, item in enumerate(item_sets[selected_product], 1):
                        st.markdown(
                            f"""
                            <div class="recommendation-box">
                                <b>{i}. {item}</b>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.error("No recommendations available. Try selecting a more popular product.")

def customer_segmentation_page(retail_data):
    """Render the customer segmentation page"""
    st.title("👥 Customer Segmentation Analysis")
    st.markdown("Segment customers based on their purchasing behavior using RFM analysis.")
    
    # Explanation section
    with st.expander("📚 What is RFM Analysis?", expanded=False):
        st.markdown("""
        RFM is a powerful customer segmentation technique:
        - **Recency (R)**: Days since last purchase
        - **Frequency (F)**: Number of transactions
        - **Monetary Value (M)**: Total revenue generated
        """)
    
    # Country selection
    country_list = retail_data['Country'].unique().tolist()
    selected_country = st.selectbox(
        '🌍 Choose Country:', 
        country_list, 
        key='seg_country'
    )
    
    try:
        country_data = choose_country(selected_country, retail_data)
        
        # Check if there's enough data
        if len(country_data) < 10:
            st.warning("Not enough data for segmentation in this country")
            return
            
        rfm_data = setup_rfm_analysis(country_data)
        clustered_data = perform_kmeans(rfm_data)
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Visualization", "📊 Statistics", "🧐 Raw Data"])
        
        with tab1:
            st.plotly_chart(cluster_plot(clustered_data), use_container_width=True)
            
            st.markdown("### 🏷️ Cluster Descriptions")
            cols = st.columns(4)
            descriptions = [
                "High-Value Customers",
                "At-Risk Customers", 
                "New Customers",
                "Loyal Low-Spenders"
            ]
            
            for i, col in enumerate(cols):
                count = len(clustered_data[clustered_data['Clusters'] == i])
                with col:
                    st.metric(f"Cluster {i}", f"{count} customers", descriptions[i])
        
        with tab2:
            # Calculate statistics - using correct column names
            stats = clustered_data.groupby("Clusters").agg({
                'Recency': 'mean',
                'Frequency': 'mean', 
                'Monetary': 'mean',
                'RFM_Score': ['mean', 'count']
            }).copy()
            
            # Flatten multi-index columns
            stats.columns = [
                'Avg_Recency', 
                'Avg_Frequency',
                'Avg_Revenue',
                'Avg_RFM_Score',
                'Customer_Count'
            ]
            
            # Format the display
            display_stats = stats.style.format({
                'Avg_Recency': '{:.1f} days',
                'Avg_Revenue': '${:,.2f}',
                'Avg_RFM_Score': '{:.1f}'
            })
            
            st.dataframe(display_stats)
            
            # Pie charts
            st.subheader("Cluster Distribution")
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    stats.reset_index(),
                    names='Clusters',
                    values='Customer_Count',
                    title='Customer Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    stats.reset_index(),
                    names='Clusters',
                    values='Avg_Revenue',
                    title='Revenue Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.dataframe(clustered_data)
    
    except Exception as e:
        st.error(f"Error in RFM analysis: {str(e)}")

def dashboard_page(retail_data):
    """Render the dashboard page"""
    st.title("📊 Retail Analytics Dashboard")
    st.markdown("Comprehensive overview of sales performance and customer behavior.")
    
    # Custom metrics styling
    st.markdown(
        f"""
        <style>
            .metric-box {{
                padding: 15px;
                border-radius: 10px;
                background-color: {BG_COLOR};
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .metric-title {{
                font-size: 14px;
                color: #555;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: {PRIMARY_COLOR};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Country selection with better UI
    country_list = retail_data['Country'].unique().tolist()
    selected_country = st.selectbox(
        '🌍 Choose Country:', 
        country_list, 
        key='dashboard_country',
        help="Select a country to view analytics"
    )
    country_data = choose_country(selected_country, retail_data)
    
    # Key metrics row
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = country_data['Sales Revenue'].sum()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Total Revenue</div>
                <div class="metric-value">${total_revenue:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        total_transactions = country_data['InvoiceNo'].nunique()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Total Transactions</div>
                <div class="metric-value">{total_transactions:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        avg_order_value = total_revenue / total_transactions
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Avg Order Value</div>
                <div class="metric-value">${avg_order_value:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        unique_customers = country_data['CustomerID'].nunique()
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">Unique Customers</div>
                <div class="metric-value">{unique_customers:,}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Customer insights section
    st.subheader("👥 Customer Insights")
    top_customers = country_data.loc[country_data["CustomerID"] != "Guest Customer"]\
                   .groupby("CustomerID")["InvoiceNo"].nunique()\
                   .sort_values(ascending=False).reset_index().head(10)
    
    fig = px.bar(top_customers, x="CustomerID", y="InvoiceNo", 
                 color='InvoiceNo', 
                 title="Top Customers by Number of Purchases",
                 color_continuous_scale=px.colors.sequential.Blues)
    fig.update_layout(
        xaxis_title="Customer ID",
        yaxis_title="Number of Purchases",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Product performance section with tabs
    st.subheader("📦 Product Performance")
    tab1, tab2 = st.tabs(["By Quantity Sold", "By Revenue Generated"])
    
    with tab1:
        top_products_qty = group_sales_quantity(country_data, 'Description')\
                          .sort_values(by="Quantity", ascending=False)\
                          .drop('Sales Revenue', axis=1).head(10)
        
        fig = px.bar(top_products_qty, x='Description', y='Quantity',
                     color='Quantity',
                     title="Top Products by Quantity Sold",
                     color_continuous_scale=px.colors.sequential.Mint)
        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Quantity Sold",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        top_products_revenue = group_sales_quantity(country_data, 'Description')\
                              .sort_values(by="Sales Revenue", ascending=False)\
                              .drop('Quantity', axis=1).head(10)
        
        fig = px.bar(top_products_revenue, x='Description', y='Sales Revenue',
                     color='Sales Revenue',
                     title="Top Products by Revenue Generated",
                     color_continuous_scale=px.colors.sequential.Oranges)
        fig.update_layout(
            xaxis_title="Product",
            yaxis_title="Revenue ($)",
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis with date range selector
    st.subheader("📅 Sales Trend Analysis")
    
    try:
        country_data['InvoiceDate'] = pd.to_datetime(country_data['InvoiceDate'])
        
        # Date range selector
        min_date = country_data['InvoiceDate'].min().date()
        max_date = country_data['InvoiceDate'].max().date()
        
        date_range = st.date_input(
            "Select Date Range:",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range"
        )
        
        if len(date_range) == 2:
            filtered_data = country_data.loc[
                (country_data['InvoiceDate'].dt.date >= date_range[0]) & 
                (country_data['InvoiceDate'].dt.date <= date_range[1])
            ].copy()
            
            # Daily and monthly tabs
            tab1, tab2 = st.tabs(["Daily Trend", "Monthly Trend"])
            
            with tab1:
                daily_sales = filtered_data.set_index('InvoiceDate')['Sales Revenue'].resample('D').sum()
                fig = px.line(daily_sales, 
                             title="Daily Sales Trend",
                             color_discrete_sequence=[PRIMARY_COLOR])
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Sales Revenue ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                monthly_sales = filtered_data.set_index('InvoiceDate')['Sales Revenue'].resample('M').sum()
                fig = px.line(monthly_sales, 
                             title="Monthly Sales Trend",
                             color_discrete_sequence=[SECONDARY_COLOR])
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Sales Revenue ($)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Could not generate time series: {str(e)}")

def main():
    """Main application function"""
    # Configure page with custom theme
    st.set_page_config(
        layout="wide", 
        page_title="Retail Analytics Dashboard",
        page_icon="🛍️"
    )
    
    # Custom CSS for the entire app
    st.markdown(
        f"""
        <style>
            /* Main app styling */
            .stApp {{
                background-color: white;
            }}
            
            /* Sidebar styling */
            .css-1d391kg {{
                background-color: {BG_COLOR} !important;
            }}
            
            /* Button styling */
            .stButton>button {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                border: none;
            }}
            
            .stButton>button:hover {{
                background-color: {SECONDARY_COLOR};
                color: white;
            }}
            
            /* Selectbox styling */
            .stSelectbox>div>div {{
                border: 1px solid #ddd;
                border-radius: 8px;
            }}
            
            /* Tab styling */
            .stTabs>div>div>button {{
                color: #555;
                font-weight: bold;
            }}
            
            .stTabs>div>div>button[aria-selected="true"] {{
                color: {PRIMARY_COLOR};
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Load data
    retail_data = load_data()
    
    # Add sales revenue column if not present
    if 'Sales Revenue' not in retail_data.columns:
        retail_data['Sales Revenue'] = retail_data['Quantity'] * retail_data['UnitPrice']
    
    # Initialize session state
    if 'generate_recs' not in st.session_state:
        st.session_state.generate_recs = False
    
    # Navigation sidebar with icons
    with st.sidebar:
        st.title("🛍️ Retail Analytics")
        st.markdown("---")
        
        page = st.radio(
            "Navigate to:",
            options=[
                "📊 Dashboard",
                "🛍️ Product Recommendations", 
                "👥 Customer Segmentation"
            ],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides insights into:
        - Sales performance
        - Customer behavior
        - Product recommendations
        """)
        
        st.markdown("---")
        st.markdown("### Made by: Adarsh Arun")
    
    # Page routing
    if page == "🛍️ Product Recommendations":
        product_recommendation_page(retail_data)
    elif page == "👥 Customer Segmentation":
        customer_segmentation_page(retail_data)
    else:
        dashboard_page(retail_data)

if __name__ == "__main__":
    main()