# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 24px; color: #1f77b4; font-weight: bold;}
    .section-header {font-size: 20px; color: #ff7f0e; margin-top: 20px;}
    .info-text {font-size: 14px; color: #2ca02c;}
    .warning-text {font-size: 14px; color: #d62728;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'file_type' not in st.session_state:
    st.session_state.file_type = None

# Function to load data based on file type
def load_data(uploaded_file, file_type):
    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "excel":
            df = pd.read_excel(uploaded_file)
        elif file_type == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Function to display data summary
def display_data_summary(df):
    st.markdown('<p class="section-header">Data Summary</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", missing_values)
    with col4:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicates)
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)

# Function for data cleaning options
def data_cleaning_options(df):
    st.markdown('<p class="section-header">Data Cleaning Options</p>', unsafe_allow_html=True)
    
    cleaned_df = df.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Handle missing values
        st.subheader("Handle Missing Values")
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if missing_cols:
            selected_missing_col = st.selectbox("Select column to handle missing values", 
                                               options=["All Columns"] + missing_cols)
            
            missing_option = st.radio("Method for handling missing values",
                                     options=["Remove rows", "Fill with mean", "Fill with median", 
                                              "Fill with mode", "Fill with custom value"])
            
            if st.button("Apply Missing Value Treatment"):
                if selected_missing_col == "All Columns":
                    cols_to_fix = missing_cols
                else:
                    cols_to_fix = [selected_missing_col]
                
                for col in cols_to_fix:
                    if missing_option == "Remove rows":
                        cleaned_df = cleaned_df.dropna(subset=[col])
                    elif missing_option == "Fill with mean" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
                    elif missing_option == "Fill with median" and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    elif missing_option == "Fill with mode":
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                    elif missing_option == "Fill with custom value":
                        custom_val = st.text_input(f"Custom value for {col}", value="0")
                        try:
                            # Try to convert to number first
                            custom_val = float(custom_val)
                        except ValueError:
                            # Keep as string if not convertible
                            pass
                        cleaned_df[col] = cleaned_df[col].fillna(custom_val)
                
                st.success("Missing values handled successfully!")
        else:
            st.info("No missing values found in the dataset.")
    
    with col2:
        # Remove duplicates
        st.subheader("Remove Duplicates")
        if st.button("Remove Duplicate Rows"):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            final_count = len(cleaned_df)
            st.success(f"Removed {initial_count - final_count} duplicate rows.")
        
        # Rename columns
        st.subheader("Rename Columns")
        col_to_rename = st.selectbox("Select column to rename", options=df.columns)
        new_name = st.text_input("New column name", value=col_to_rename)
        
        if st.button("Rename Column") and new_name != col_to_rename:
            cleaned_df = cleaned_df.rename(columns={col_to_rename: new_name})
            st.success(f"Column renamed from '{col_to_rename}' to '{new_name}'")
    
    return cleaned_df

# Function to create visualizations
def create_visualizations(df):
    st.markdown('<p class="section-header">Data Visualizations</p>', unsafe_allow_html=True)
    
    # Visualization type selection
    viz_type = st.selectbox("Select Visualization Type", 
                           options=["Auto Visualize", "Histogram", "Box Plot", "Bar Chart", 
                                   "Scatter Plot", "Line Chart", "Pie Chart", "Heatmap", "Pair Plot"])
    
    # Column selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = df.columns.tolist()
    
    # Auto visualization logic
    if viz_type == "Auto Visualize":
        if not numeric_cols and not categorical_cols:
            st.warning("No suitable columns for automatic visualization.")
            return
        
        # For datasets with both numeric and categorical columns
        if numeric_cols and categorical_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                num_col = st.selectbox("Select Numeric Column", options=numeric_cols)
            with col2:
                cat_col = st.selectbox("Select Categorical Column", options=categorical_cols)
            
            # Create a combined visualization
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'Distribution of {num_col}', f'{num_col} by {cat_col}')
            )
            
            # Histogram
            fig.add_trace(go.Histogram(x=df[num_col], name="Distribution"), row=1, col=1)
            
            # Box plot
            for category in df[cat_col].unique():
                subset = df[df[cat_col] == category]
                fig.add_trace(go.Box(y=subset[num_col], name=str(category)), row=1, col=2)
            
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # For datasets with only numeric columns
        elif numeric_cols and not categorical_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                col_x = st.selectbox("Select X Axis Column", options=numeric_cols)
            with col2:
                col_y = st.selectbox("Select Y Axis Column", options=numeric_cols)
            
            fig = px.scatter(df, x=col_x, y=col_y, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation heatmap
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        # For datasets with only categorical columns
        elif categorical_cols and not numeric_cols:
            col = st.selectbox("Select Column", options=categorical_cols)
            
            value_counts = df[col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Manual visualization selection
    else:
        if viz_type in ["Histogram", "Box Plot"] and numeric_cols:
            col = st.selectbox("Select Column", options=numeric_cols)
            if viz_type == "Histogram":
                fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            else:
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart" and categorical_cols:
            col = st.selectbox("Select Column", options=categorical_cols)
            value_counts = df[col].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                         title=f"Bar Chart of {col}", labels={'x': col, 'y': 'Count'})
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X Axis", options=numeric_cols)
            with col2:
                y_col = st.selectbox("Select Y Axis", options=numeric_cols)
            
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color by (optional)", options=["None"] + categorical_cols)
                if color_col == "None":
                    color_col = None
            
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart" and numeric_cols:
            # Try to find a date/time column
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if date_cols:
                x_col = st.selectbox("Select Date Column", options=date_cols)
                y_col = st.selectbox("Select Value Column", options=numeric_cols)
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over Time")
            else:
                x_col = st.selectbox("Select X Axis", options=all_cols)
                y_col = st.selectbox("Select Y Axis", options=numeric_cols)
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart" and categorical_cols:
            col = st.selectbox("Select Column", options=categorical_cols)
            value_counts = df[col].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Heatmap" and numeric_cols:
            st.subheader("Correlation Heatmap")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pair Plot" and len(numeric_cols) >= 2:
            st.subheader("Pair Plot")
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color by", options=["None"] + categorical_cols)
                if color_col == "None":
                    color_col = None
            
            fig = px.scatter_matrix(df, dimensions=numeric_cols[:4], color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No suitable columns for selected visualization type.")

# Function for filtering data
def filter_data(df):
    st.markdown('<p class="section-header">Data Filtering</p>', unsafe_allow_html=True)
    
    filtered_df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    st.write("Apply filters to your data:")
    
    # Numeric filters
    if numeric_cols:
        st.subheader("Numeric Filters")
        num_col = st.selectbox("Select numeric column to filter", options=["None"] + numeric_cols)
        
        if num_col != "None":
            min_val = float(df[num_col].min())
            max_val = float(df[num_col].max())
            
            selected_range = st.slider(
                f"Select range for {num_col}",
                min_val, max_val, (min_val, max_val)
            )
            
            filtered_df = filtered_df[
                (filtered_df[num_col] >= selected_range[0]) & 
                (filtered_df[num_col] <= selected_range[1])
            ]
    
    # Categorical filters
    if categorical_cols:
        st.subheader("Categorical Filters")
        cat_col = st.selectbox("Select categorical column to filter", options=["None"] + categorical_cols)
        
        if cat_col != "None":
            unique_vals = df[cat_col].unique().tolist()
            selected_vals = st.multiselect(f"Select values for {cat_col}", options=unique_vals, default=unique_vals)
            
            if selected_vals:
                filtered_df = filtered_df[filtered_df[cat_col].isin(selected_vals)]
    
    st.write(f"Filtered data: {len(filtered_df)} rows × {len(filtered_df.columns)} columns")
    return filtered_df

# Main app function
def main():
    st.markdown('<p class="main-header">📊 Professional Data Visualization Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar for file upload and navigation
    with st.sidebar:
        st.header("Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data file", 
            type=["csv", "xlsx", "xls", "json"]
        )
        
        if uploaded_file is not None:
            # Determine file type
            if uploaded_file.name.endswith('.csv'):
                file_type = 'csv'
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                file_type = 'excel'
            elif uploaded_file.name.endswith('.json'):
                file_type = 'json'
            else:
                st.error("Unsupported file format")
                return
            
            # Load data
            if st.session_state.df is None or st.session_state.file_type != file_type:
                with st.spinner("Loading data..."):
                    df = load_data(uploaded_file, file_type)
                    if df is not None:
                        st.session_state.df = df
                        st.session_state.df_cleaned = df.copy()
                        st.session_state.file_type = file_type
                        st.success("Data loaded successfully!")
        
        # Navigation
        st.header("Navigation")
        page_options = [
            "Data Overview", 
            "Data Cleaning", 
            "Data Filtering", 
            "Visualizations", 
            "Export Data"
        ]
        selected_page = st.radio("Go to", page_options)
    
    # Main content area
    if st.session_state.df is not None:
        df = st.session_state.df
        df_cleaned = st.session_state.df_cleaned
        
        if selected_page == "Data Overview":
            st.header("Data Overview")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display data summary
            display_data_summary(df)
        
        elif selected_page == "Data Cleaning":
            st.header("Data Cleaning")
            st.session_state.df_cleaned = data_cleaning_options(df)
            
            # Show cleaned data preview
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.df_cleaned.head(), use_container_width=True)
            
            # Show cleaning summary
            st.subheader("Cleaning Summary")
            original_rows, original_cols = df.shape
            cleaned_rows, cleaned_cols = st.session_state.df_cleaned.shape
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows Removed", original_rows - cleaned_rows)
            col2.metric("Columns Changed", original_cols - cleaned_cols)
            col3.metric("Missing Values Removed", 
                       df.isnull().sum().sum() - st.session_state.df_cleaned.isnull().sum().sum())
        
        elif selected_page == "Data Filtering":
            st.header("Data Filtering")
            filtered_df = filter_data(df_cleaned)
            
            # Show filtered data
            st.subheader("Filtered Data Preview")
            st.dataframe(filtered_df.head(), use_container_width=True)
            
            # Update the cleaned dataframe with filtered data
            if st.button("Apply Filters to Dataset"):
                st.session_state.df_cleaned = filtered_df
                st.success("Filters applied to dataset!")
        
        elif selected_page == "Visualizations":
            st.header("Data Visualizations")
            create_visualizations(df_cleaned)
        
        elif selected_page == "Export Data":
            st.header("Export Data")
            
            # Export cleaned/filtered data
            st.subheader("Export Processed Data")
            export_format = st.radio("Select export format", options=["CSV", "Excel"])
            
            if export_format == "CSV":
                csv = df_cleaned.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
            else:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_cleaned.to_excel(writer, index=False, sheet_name='Processed Data')
                st.download_button(
                    label="Download data as Excel",
                    data=output.getvalue(),
                    file_name="processed_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
            
            # Export visualization
            st.subheader("Export Current Visualization")
            # Note: In a real implementation, you would capture the current figure
            # and provide export options. This is a placeholder.
            st.info("To export a visualization, use the camera icon that appears when hovering over a chart.")
    
    else:
        # Show instructions if no data is uploaded
        st.info("👈 Please upload a data file to get started.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Supported File Formats")
            st.markdown("""
            - **CSV**: Comma-separated values
            - **Excel**: .xlsx or .xls files
            - **JSON**: JavaScript Object Notation
            """)
        
        with col2:
            st.subheader("Getting Started")
            st.markdown("""
            1. Upload your data file using the sidebar
            2. Explore your data in the Data Overview section
            3. Clean and filter your data as needed
            4. Create visualizations based on your data
            5. Export your processed data or charts
            """)

# Run the app
if __name__ == "__main__":
    main()