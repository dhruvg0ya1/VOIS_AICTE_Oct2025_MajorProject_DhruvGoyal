import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Netflix Dataset Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #E50914;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #564d4d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #E50914;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: white;
        margin: 0.25rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🎬 Netflix Dataset Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Content Trends Analysis for Strategic Recommendations</p>', unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load and preprocess the Netflix dataset from file path"""
    try:
        file_path = "Netflix Dataset.csv"
        
        df = pd.read_csv(file_path)
        
        # Map your column names to standard names
        column_mapping = {
            'Show_Id': 'show_id',
            'Category': 'type',
            'Title': 'title',
            'Director': 'director',
            'Cast': 'cast',
            'Country': 'country',
            'Release_Date': 'date_added',
            'Rating': 'rating',
            'Duration': 'duration',
            'Type': 'listed_in',
            'Description': 'description'
        }
        
        # Rename columns if they exist
        df.rename(columns=column_mapping, inplace=True)
        
        # Data preprocessing
        # Convert date_added to datetime
        df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
        df['year_added'] = df['date_added'].dt.year
        df['month_added'] = df['date_added'].dt.month
        
        # Extract release year from date_added if release_year doesn't exist
        if 'release_year' not in df.columns:
            df['release_year'] = df['year_added']
        
        # Clean country data (take first country if multiple)
        df['country'] = df['country'].fillna('Unknown')
        df['primary_country'] = df['country'].apply(lambda x: str(x).split(',')[0].strip())
        
        # Clean duration data
        df['duration'] = df['duration'].fillna('Unknown')
        
        # Clean rating
        df['rating'] = df['rating'].fillna('Not Rated')
        
        # Clean listed_in (genres)
        df['listed_in'] = df['listed_in'].fillna('Unknown')
        
        return df
    
    except FileNotFoundError:
        st.error(f"❌ File not found: {file_path}")
        st.info("Please update the file_path variable in the load_data() function with the correct path to your Netflix dataset.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

# Load the data
with st.spinner("Loading Netflix dataset..."):
    df = load_data()

# Display dataset info
st.sidebar.header("📊 Dataset Information")
st.sidebar.metric("Total Records", len(df))
st.sidebar.metric("Features", len(df.columns))
st.sidebar.metric("Date Range", f"{df['year_added'].min():.0f} - {df['year_added'].max():.0f}" if 'year_added' in df.columns else "N/A")

# Sidebar filters
st.sidebar.header("🔍 Filters")
content_type = st.sidebar.multiselect(
    "Content Type",
    options=df['type'].unique().tolist(),
    default=df['type'].unique().tolist()
)

if 'year_added' in df.columns and df['year_added'].notna().any():
    min_year = int(df['year_added'].min())
    max_year = int(df['year_added'].max())
    year_range = st.sidebar.slider(
        "Year Added",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
    filtered_df = df[
        (df['type'].isin(content_type)) & 
        (df['year_added'] >= year_range[0]) & 
        (df['year_added'] <= year_range[1])
    ]
else:
    filtered_df = df[df['type'].isin(content_type)]

# Country filter
top_countries = df['primary_country'].value_counts().head(20).index.tolist()
selected_countries = st.sidebar.multiselect(
    "Select Countries (Top 20)",
    options=['All'] + top_countries,
    default=['All']
)

if 'All' not in selected_countries:
    filtered_df = filtered_df[filtered_df['primary_country'].isin(selected_countries)]

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", 
    "📈 Content Trends", 
    "🌍 Geographic Analysis", 
    "🎭 Genre Analysis",
    "📅 Release Patterns",
    "💡 Recommendations"
])

# TAB 1: OVERVIEW
with tab1:
    st.header("Dataset Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Titles", f"{len(filtered_df):,}")
    
    with col2:
        movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
        st.metric("Movies", f"{movies_count:,}", f"{movies_count/len(filtered_df)*100:.1f}%")
    
    with col3:
        tv_shows_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
        st.metric("TV Shows", f"{tv_shows_count:,}", f"{tv_shows_count/len(filtered_df)*100:.1f}%")
    
    with col4:
        countries_count = filtered_df['primary_country'].nunique()
        st.metric("Countries", countries_count)
    
    st.markdown("---")
    
    # Content Type Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Type Distribution")
        type_counts = filtered_df['type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            color_discrete_sequence=['#E50914', '#564d4d'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Ratings Distribution")
        rating_counts = filtered_df['rating'].value_counts().head(10)
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            color=rating_counts.values,
            color_continuous_scale='Reds',
            labels={'x': 'Rating', 'y': 'Count'}
        )
        fig.update_layout(height=400, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Top 5 Countries")
        top_5_countries = filtered_df['primary_country'].value_counts().head(5)
        for country, count in top_5_countries.items():
            st.write(f"**{country}**: {count:,} titles")
    
    with col2:
        st.subheader("Top 5 Ratings")
        top_5_ratings = filtered_df['rating'].value_counts().head(5)
        for rating, count in top_5_ratings.items():
            st.write(f"**{rating}**: {count:,} titles")
    
    with col3:
        st.subheader("Content Statistics")
        st.write(f"**Avg Release Year**: {filtered_df['release_year'].mean():.0f}")
        st.write(f"**Oldest Content**: {filtered_df['release_year'].min():.0f}")
        st.write(f"**Newest Content**: {filtered_df['release_year'].max():.0f}")
    
    st.markdown("---")
    
    # Sample data
    st.subheader("Sample Data Preview")
    st.dataframe(
        filtered_df[['type', 'title', 'primary_country', 'release_year', 'rating', 'duration']].head(20),
        use_container_width=True
    )

# TAB 2: CONTENT TRENDS
with tab2:
    st.header("Content Trends Over Time")
    
    if 'year_added' in filtered_df.columns:
        # Content added over years
        st.subheader("Content Added by Year")
        yearly_data = filtered_df.groupby(['year_added', 'type']).size().reset_index(name='count')
        
        fig = px.line(
            yearly_data,
            x='year_added',
            y='count',
            color='type',
            markers=True,
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
            labels={'year_added': 'Year', 'count': 'Number of Titles', 'type': 'Content Type'}
        )
        fig.update_layout(height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Year over year growth
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Year-over-Year Growth")
            yearly_total = filtered_df.groupby('year_added').size().reset_index(name='count')
            yearly_total['growth'] = yearly_total['count'].pct_change() * 100
            
            fig = px.bar(
                yearly_total,
                x='year_added',
                y='growth',
                color='growth',
                color_continuous_scale='RdYlGn',
                labels={'year_added': 'Year', 'growth': 'Growth Rate (%)'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cumulative Content Growth")
            yearly_total['cumulative'] = yearly_total['count'].cumsum()
            
            fig = px.area(
                yearly_total,
                x='year_added',
                y='cumulative',
                color_discrete_sequence=['#E50914'],
                labels={'year_added': 'Year', 'cumulative': 'Total Titles'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly trends
        st.subheader("Monthly Addition Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_data = filtered_df.groupby('month_added').size().reset_index(name='count')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_data['month_name'] = monthly_data['month_added'].apply(
                lambda x: month_names[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else 'Unknown'
            )
            
            fig = px.bar(
                monthly_data,
                x='month_name',
                y='count',
                color='count',
                color_continuous_scale='Reds',
                labels={'month_name': 'Month', 'count': 'Number of Titles'}
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_month = filtered_df.groupby(['month_added', 'type']).size().reset_index(name='count')
            type_month['month_name'] = type_month['month_added'].apply(
                lambda x: month_names[int(x)-1] if pd.notna(x) and 1 <= x <= 12 else 'Unknown'
            )
            
            fig = px.bar(
                type_month,
                x='month_name',
                y='count',
                color='type',
                barmode='stack',
                color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
                labels={'month_name': 'Month', 'count': 'Number of Titles', 'type': 'Content Type'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Date information not available in the dataset.")

# TAB 3: GEOGRAPHIC ANALYSIS
with tab3:
    st.header("Geographic Content Distribution")
    
    # Top countries
    st.subheader("Top 20 Countries by Content Production")
    country_counts = filtered_df['primary_country'].value_counts().head(20)
    
    fig = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation='h',
        color=country_counts.values,
        color_continuous_scale='Reds',
        labels={'x': 'Number of Titles', 'y': 'Country'},
        text=country_counts.values
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Content Type by Top 10 Countries")
        top_countries = filtered_df['primary_country'].value_counts().head(10).index
        country_type_data = filtered_df[filtered_df['primary_country'].isin(top_countries)]
        country_type = country_type_data.groupby(['primary_country', 'type']).size().reset_index(name='count')
        
        fig = px.bar(
            country_type,
            x='primary_country',
            y='count',
            color='type',
            barmode='group',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
            labels={'primary_country': 'Country', 'count': 'Number of Titles', 'type': 'Content Type'}
        )
        fig.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Regional Content Distribution")
        # Regional categorization
        region_map = {
            'United States': 'North America',
            'Canada': 'North America',
            'Mexico': 'Latin America',
            'Brazil': 'Latin America',
            'Argentina': 'Latin America',
            'United Kingdom': 'Europe',
            'France': 'Europe',
            'Germany': 'Europe',
            'Spain': 'Europe',
            'Italy': 'Europe',
            'India': 'Asia',
            'Japan': 'Asia',
            'South Korea': 'Asia',
            'China': 'Asia',
            'Singapore': 'Asia',
            'Thailand': 'Asia',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania',
            'Egypt': 'Africa',
            'South Africa': 'Africa',
            'Nigeria': 'Africa'
        }
        
        filtered_df['region'] = filtered_df['primary_country'].map(region_map).fillna('Other')
        region_counts = filtered_df['region'].value_counts()
        
        fig = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            color_discrete_sequence=px.colors.sequential.Reds,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Country growth over time
    if 'year_added' in filtered_df.columns:
        st.subheader("Content Growth by Top 5 Countries Over Time")
        top_5_countries = filtered_df['primary_country'].value_counts().head(5).index
        country_year = filtered_df[filtered_df['primary_country'].isin(top_5_countries)]
        country_year_data = country_year.groupby(['year_added', 'primary_country']).size().reset_index(name='count')
        
        fig = px.line(
            country_year_data,
            x='year_added',
            y='count',
            color='primary_country',
            markers=True,
            labels={'year_added': 'Year', 'count': 'Number of Titles', 'primary_country': 'Country'}
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# TAB 4: GENRE ANALYSIS
with tab4:
    st.header("Genre Analysis")
    
    # Extract genres
    all_genres = []
    for genres in filtered_df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in str(genres).split(',')])
    
    genre_df = pd.Series(all_genres).value_counts().head(25)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top 25 Genres/Categories")
        fig = px.bar(
            x=genre_df.values,
            y=genre_df.index,
            orientation='h',
            color=genre_df.values,
            color_continuous_scale='Reds',
            labels={'x': 'Number of Titles', 'y': 'Genre'},
            text=genre_df.values
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Genre Distribution")
        top_10_genres = genre_df.head(10)
        fig = px.pie(
            values=top_10_genres.values,
            names=top_10_genres.index,
            color_discrete_sequence=px.colors.sequential.Reds
        )
        fig.update_traces(textposition='inside', textinfo='percent')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Genre Statistics")
        st.metric("Total Unique Genres", len(pd.Series(all_genres).unique()))
        st.metric("Most Popular Genre", genre_df.index[0])
        st.metric("Titles in Top Genre", f"{genre_df.values[0]:,}")
    
    # Content Type vs Genre
    st.subheader("Content Type Distribution in Top 10 Genres")
    top_genres = genre_df.head(10).index.tolist()
    
    genre_type_data = []
    for idx, row in filtered_df.iterrows():
        if pd.notna(row['listed_in']):
            genres = [g.strip() for g in str(row['listed_in']).split(',')]
            for genre in genres:
                if genre in top_genres:
                    genre_type_data.append({'genre': genre, 'type': row['type']})
    
    genre_type_df = pd.DataFrame(genre_type_data)
    genre_type_counts = genre_type_df.groupby(['genre', 'type']).size().reset_index(name='count')
    
    fig = px.bar(
        genre_type_counts,
        x='genre',
        y='count',
        color='type',
        barmode='stack',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
        labels={'genre': 'Genre', 'count': 'Number of Titles', 'type': 'Content Type'},
        text='count'
    )
    fig.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre trends over time
    if 'year_added' in filtered_df.columns:
        st.subheader("Genre Trends Over Time (Top 5 Genres)")
        top_5_genres = genre_df.head(5).index.tolist()
        
        genre_year_data = []
        for idx, row in filtered_df.iterrows():
            if pd.notna(row['listed_in']) and pd.notna(row['year_added']):
                genres = [g.strip() for g in str(row['listed_in']).split(',')]
                for genre in genres:
                    if genre in top_5_genres:
                        genre_year_data.append({'genre': genre, 'year': row['year_added']})
        
        genre_year_df = pd.DataFrame(genre_year_data)
        genre_year_counts = genre_year_df.groupby(['year', 'genre']).size().reset_index(name='count')
        
        fig = px.line(
            genre_year_counts,
            x='year',
            y='count',
            color='genre',
            markers=True,
            labels={'year': 'Year', 'count': 'Number of Titles', 'genre': 'Genre'}
        )
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: RELEASE PATTERNS
with tab5:
    st.header("Release Patterns Analysis")
    
    # Release year distribution
    st.subheader("Content by Release Year")
    release_year_counts = filtered_df['release_year'].value_counts().sort_index()
    
    fig = px.bar(
        x=release_year_counts.index,
        y=release_year_counts.values,
        color=release_year_counts.values,
        color_continuous_scale='Reds',
        labels={'x': 'Release Year', 'y': 'Number of Titles'}
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Release Year by Content Type")
        # Group by decade for better visualization
        filtered_df['decade'] = (filtered_df['release_year'] // 10) * 10
        decade_type = filtered_df.groupby(['decade', 'type']).size().reset_index(name='count')
        
        fig = px.bar(
            decade_type,
            x='decade',
            y='count',
            color='type',
            barmode='group',
            color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
            labels={'decade': 'Decade', 'count': 'Number of Titles', 'type': 'Content Type'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age of Content on Platform")
        if 'year_added' in filtered_df.columns:
            filtered_df['content_age'] = filtered_df['year_added'] - filtered_df['release_year']
            age_dist = filtered_df['content_age'].value_counts().sort_index()
            
            fig = px.histogram(
                filtered_df,
                x='content_age',
                nbins=50,
                color_discrete_sequence=['#E50914'],
                labels={'content_age': 'Years Between Release and Addition', 'count': 'Number of Titles'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year added information not available.")
    
    # Rating patterns
    st.subheader("Rating Distribution by Content Type")
    rating_type = filtered_df.groupby(['rating', 'type']).size().reset_index(name='count')
    top_ratings = filtered_df['rating'].value_counts().head(10).index
    rating_type_filtered = rating_type[rating_type['rating'].isin(top_ratings)]
    
    fig = px.bar(
        rating_type_filtered,
        x='rating',
        y='count',
        color='type',
        barmode='group',
        color_discrete_map={'Movie': '#E50914', 'TV Show': '#564d4d'},
        labels={'rating': 'Rating', 'count': 'Number of Titles', 'type': 'Content Type'}
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# TAB 6: RECOMMENDATIONS
with tab6:
    st.header("Strategic Recommendations")
    
    # Calculate insights
    total_content = len(filtered_df)
    movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
    tv_shows_count = len(filtered_df[filtered_df['type'] == 'TV Show'])
    movies_pct = (movies_count / total_content * 100) if total_content > 0 else 0
    tv_shows_pct = (tv_shows_count / total_content * 100) if total_content > 0 else 0
    
    top_country = filtered_df['primary_country'].value_counts().index[0]
    top_country_count = filtered_df['primary_country'].value_counts().values[0]
    top_country_pct = (top_country_count / total_content * 100) if total_content > 0 else 0
    
    # Extract top genre
    all_genres = []
    for genres in filtered_df['listed_in'].dropna():
        all_genres.extend([g.strip() for g in str(genres).split(',')])
    top_genre = pd.Series(all_genres).value_counts().index[0]
    top_genre_count = pd.Series(all_genres).value_counts().values[0]
    
    st.subheader("📊 Executive Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown(f"""
        ### Dataset Overview
        - **Total Content Analyzed**: {total_content:,} titles
        - **Time Period**: {filtered_df['release_year'].min():.0f} - {filtered_df['release_year'].max():.0f}
        - **Countries Represented**: {filtered_df['primary_country'].nunique()}
        - **Unique Genres**: {len(pd.Series(all_genres).unique())}
        """)
    
    with summary_col2:
        st.markdown(f"""
        ### Key Findings
        - **Content Mix**: {movies_pct:.1f}% Movies, {tv_shows_pct:.1f}% TV Shows
        - **Top Market**: {top_country} ({top_country_pct:.1f}% of content)
        - **Most Popular Genre**: {top_genre} ({top_genre_count:,} titles)
        - **Dominant Rating**: {filtered_df['rating'].mode()[0]}
        """)
    
    st.markdown("---")
    
    # Visual insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📺 Content Balance</h3>
            <p><strong>Movies:</strong> {movies_pct:.1f}%</p>
            <p><strong>TV Shows:</strong> {tv_shows_pct:.1f}%</p>
            <p style="font-size: 0.9em; color: #666;">Ratio: {movies_count/tv_shows_count if tv_shows_count > 0 else 'N/A':.2f}:1</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌍 Leading Market</h3>
            <p><strong>{top_country}</strong></p>
            <p>{top_country_count:,} titles ({top_country_pct:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎭 Top Genre</h3>
            <p><strong>{top_genre}</strong></p>
            <p>{top_genre_count:,} titles</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("💡 Strategic Recommendations")
    
    recommendations = [
        {
            "title": "1. Content Portfolio Optimization",
            "description": f"Current split: {movies_pct:.1f}% Movies vs {tv_shows_pct:.1f}% TV Shows. {'Netflix should consider increasing TV Show production as they typically offer better viewer retention and longer engagement.' if movies_pct > 65 else 'The current balance is healthy. Maintain focus on quality over quantity.'}",
            "action": """
            **Immediate Actions:**
            - Analyze completion rates: Movies vs TV Shows
            - Identify which format drives more subscriptions
            - Invest in limited series (6-10 episodes) to balance flexibility and engagement
            
            **Long-term Strategy:**
            - Develop flagship TV series with strong brand identity
            - Create interconnected universes for sustained engagement
            - Balance between binge-worthy series and weekly releases
            """
        },
        {
            "title": "2. Geographic Diversification Strategy",
            "description": f"{top_country} dominates with {top_country_pct:.1f}% of content. While strong in key markets, Netflix should expand into underrepresented regions for growth.",
            "action": """
            **Emerging Market Focus:**
            - Increase local content production in Asia-Pacific, Africa, and Middle East
            - Partner with regional production houses and streaming platforms
            - Create culturally relevant content with global appeal (think Squid Game, Money Heist)
            
            **Content Localization:**
            - Invest in dubbing and subtitling for top 20 languages
            - Develop region-specific recommendation algorithms
            - Create co-productions between countries for wider appeal
            """
        },
        {
            "title": "3. Genre Strategy & Innovation",
            "description": f"{top_genre} leads the catalog. While leveraging successful genres, Netflix must innovate to stay ahead of competitors.",
            "action": """
            **Double Down on Winners:**
            - Increase budget for top 5 performing genres
            - Create spin-offs and franchises in successful categories
            - Develop genre-specific content hubs on the platform
            
            **Explore New Territories:**
            - Hybrid genres (e.g., Sci-Fi Comedy, Horror Documentary)
            - Interactive content and choose-your-own-adventure stories
            - Short-form content (15-30 min) for mobile-first audiences
            - Leverage trending genres from competitor analysis
            """
        },
        {
            "title": "4. Release Timing & Content Calendar Optimization",
            "description": "Strategic timing of releases can maximize impact and maintain year-round engagement.",
            "action": """
            **Peak Season Strategy:**
            - Schedule major releases during high-engagement months
            - Align premieres with cultural events and holidays
            - Stagger big releases to avoid cannibalization
            
            **Content Pipeline Management:**
            - Maintain steady flow: at least 2-3 major releases per month
            - Balance between instant-hit potential and slow-burn series
            - Create anticipation through strategic teaser campaigns
            - Use data on viewing patterns to optimize release days/times
            """
        },
        {
            "title": "5. Quality Over Quantity Approach",
            "description": "With increased competition, focus on creating must-watch content rather than volume.",
            "action": """
            **Premium Content Investment:**
            - Allocate higher budgets to flagship productions
            - Partner with acclaimed directors and showrunners
            - Develop award-worthy content for brand prestige
            - Create exclusive partnerships with top talent
            
            **Content Curation:**
            - Regular catalog audits to remove underperforming titles
            - Focus on completion rates and re-watch value
            - Invest in evergreen content with long-term appeal
            - Build franchises that generate sustained buzz (e.g., Stranger Things model)
            """
        },
        {
            "title": "6. Data-Driven Content Acquisition",
            "description": "Leverage viewer data to guide production and licensing decisions.",
            "action": """
            **Analytics-First Approach:**
            - Identify content gaps in popular categories
            - Analyze competitor catalogs for opportunities
            - Track emerging trends on social media and streaming platforms
            - Use predictive modeling for content success probability
            
            **Performance Metrics:**
            - Monitor: Completion rate, viewing hours, social engagement, subscription impact
            - A/B test thumbnails, titles, and trailers
            - Regional preference analysis for targeted content
            - ROI tracking for all productions
            """
        },
        {
            "title": "7. Audience Segmentation & Personalization",
            "description": "Different demographics consume content differently. Tailor strategy accordingly.",
            "action": """
            **Demographic Targeting:**
            - Family-friendly content for household subscriptions
            - Young adult content (18-34) for growth segment
            - International content for diverse communities
            - Documentary and educational content for premium subscribers
            
            **Personalization Enhancement:**
            - Improve recommendation algorithms
            - Create user profiles with better preference tracking
            - Develop themed collections and curated lists
            - Implement smart notifications for new relevant content
            """
        }
    ]
    
    for rec in recommendations:
        with st.expander(rec["title"], expanded=False):
            st.write(rec["description"])
            st.markdown("**Action Plan:**")
            st.markdown(rec["action"])
    
    st.markdown("---")
    
    st.subheader("📈 Future Outlook & Growth Opportunities")
    
    outlook_col1, outlook_col2 = st.columns(2)
    
    with outlook_col1:
        st.markdown("""
        ### Short-Term Priorities (6-12 months)
        1. **Content Gap Analysis**: Identify underrepresented but high-demand genres
        2. **Regional Expansion**: Focus on 3-5 emerging markets with pilot productions
        3. **Quality Upgrades**: Increase production values for top-performing series
        4. **Marketing Optimization**: Data-driven campaigns for new releases
        5. **Platform Features**: Enhance discovery and personalization tools
        """)
    
    with outlook_col2:
        st.markdown("""
        ### Long-Term Vision (2-5 years)
        1. **Global Content Hub**: Become the go-to platform for international content
        2. **Original IP Dominance**: Own franchises that rival traditional studios
        3. **Interactive Entertainment**: Pioneer new forms of storytelling
        4. **Strategic Partnerships**: Collaborate with top global creators
        5. **Sustainable Growth**: Balance subscriber acquisition with retention
        """)
    
    st.markdown("---")
    
    st.subheader("⚠️ Risk Factors & Mitigation")
    
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("""
        ### Key Risks
        - **Increased Competition**: Disney+, Amazon, Apple TV+ gaining market share
        - **Content Costs**: Rising production and licensing expenses
        - **Market Saturation**: Slower growth in mature markets
        - **Economic Factors**: Subscription cancellations during downturns
        - **Content Misses**: High-budget failures impact bottom line
        """)
    
    with risk_col2:
        st.markdown("""
        ### Mitigation Strategies
        - **Differentiation**: Focus on exclusive, can't-miss content
        - **Cost Management**: Optimize production budgets, explore co-productions
        - **International Growth**: Aggressive expansion in Asia, Africa, Latin America
        - **Tiered Pricing**: Flexible plans to retain price-sensitive subscribers
        - **Data Analytics**: Reduce risk through better predictive modeling
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Key Performance Indicators (KPIs) to Track")
    
    kpi_metrics = [
        "**Content Engagement**: Average viewing hours per subscriber",
        "**Completion Rate**: Percentage of users finishing series/movies",
        "**Subscriber Retention**: Month-over-month churn rate",
        "**Content ROI**: Revenue generated vs production cost",
        "**Market Share**: Position vs competitors in key regions",
        "**Brand Sentiment**: Social media mentions and sentiment analysis",
        "**Release Efficiency**: Time from production to platform",
        "**Catalog Turnover**: Rate of new content addition vs removal"
    ]
    
    col1, col2 = st.columns(2)
    for i, kpi in enumerate(kpi_metrics):
        if i % 2 == 0:
            col1.markdown(f"✓ {kpi}")
        else:
            col2.markdown(f"✓ {kpi}")
    
    st.markdown("---")
    
    st.subheader("🔍 Conclusion")
    
    st.markdown(f"""
    Based on comprehensive analysis of {total_content:,} titles spanning {filtered_df['primary_country'].nunique()} countries, 
    Netflix has established a strong content foundation with {movies_pct:.1f}% Movies and {tv_shows_pct:.1f}% TV Shows.
    
    **Key Strengths:**
    - Dominant position in {top_country} market
    - Strong genre diversity with focus on {top_genre}
    - Consistent content addition over the years
    - Wide geographic representation
    
    **Areas for Improvement:**
    - Geographic diversification beyond top markets
    - Balance between Movies and TV Shows for optimal engagement
    - Strategic genre exploration and innovation
    - Enhanced content quality over quantity approach
    
    **Bottom Line:**
    By implementing these strategic recommendations, Netflix can strengthen its competitive position, 
    expand into high-growth markets, and continue to deliver compelling content that drives subscriber 
    growth and retention. The focus should be on creating differentiated, high-quality content while 
    maintaining operational efficiency and leveraging data analytics for informed decision-making.
    
    **Success will depend on:**
    1. Agile response to market trends
    2. Strategic investments in emerging markets
    3. Continuous innovation in content formats
    4. Strong partnerships with global creators
    5. Data-driven optimization of the content catalog
    """)
    
    # Download report option
    st.markdown("---")
    st.info("💾 **Pro Tip**: Use your browser's print function (Ctrl+P) to save this analysis as a PDF report.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #564d4d; padding: 20px;'>
        <p><strong>Netflix Dataset Analysis Project</strong></p>
        <p>Built with Streamlit & Plotly | Data-Driven Insights for Strategic Decision Making</p>
        <p style='font-size: 0.8em; margin-top: 10px;'>
            📊 Total Records Analyzed: {total} | 🌍 Countries: {countries} | 🎭 Genres: {genres}
        </p>
    </div>
""".format(
    total=f"{len(df):,}",
    countries=df['primary_country'].nunique(),
    genres=len(pd.Series([g.strip() for genres in df['listed_in'].dropna() for g in str(genres).split(',')]).unique())
), unsafe_allow_html=True)