import os
import tempfile
import zipfile
import requests
import pandas as pd
import numpy as np
import json
import geopandas as gpd
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from bokeh.models import GeoJSONDataSource
from bokeh.models import Div, HoverTool, Title
from bokeh.models import LogScale, LogTicker, LogTickFormatter, NumeralTickFormatter
from bokeh.models import MultiChoice, Slider, RadioButtonGroup , Spacer
from bokeh.models import ColorBar, CategoricalColorMapper, LogColorMapper
from bokeh.palettes import Category10, Reds8

url_co2gdp_data = 'https://drive.switch.ch/index.php/s/cxW0xrmQXdGL1VJ/download'
url_geo_data = 'https://drive.switch.ch/index.php/s/bfb1TrwoIrXGAfM/download'

# Load the dataset
def load_data():
    try:
        return pd.read_csv(url_co2gdp_data) #, sep=';'
    except Exception as e:
        # Create a sample dataframe for demonstration if file is not found
        sample_df = pd.DataFrame({
            'country': ['United States', 'China', 'India', 'Germany', 'Brazil'],
            'region': ['North America', 'Asia', 'Asia', 'Europe', 'South America'],
            'year': [2000, 2000, 2000, 2000, 2000],
            'co2': [20.2, 2.7, 0.9, 10.1, 1.9],
            'gdp': [36330, 959, 452, 23635, 3739]
        })
        return sample_df  

df = load_data()


# --------------------------------------
# Create title
# --------------------------------------
title = Div(text="""<h1 style='text-align: center; color: #3366cc;'>Sample Dashboard on the CO2 Emissions Dataset</h1>""",
            width=800, height=50)

# --------------------------------------
# DATASET OVERVIEW SECTION
# --------------------------------------
overview_title = Div(text="""<h2 style='color: #3366cc;'>Dataset Overview</h2>""",
                    width=800, height=30)

# Table of columns with datatype
column_types = pd.DataFrame({
    'Column': df.columns,
    'Data Type': [str(df[col].dtype) for col in df.columns]
})
source_column_types = ColumnDataSource(column_types)
columns = [
    TableColumn(field="Column", title="Column Name"),
    TableColumn(field="Data Type", title="Data Type")
]
data_table = DataTable(source=source_column_types, columns=columns, width=400, height=150, index_position=None)


# Overview section - Layout elements
overview_section = column(
    overview_title,
    data_table,
    row(
        Div(text="""<div style="display: flex; align-items: center; height: 40px;"><h3 style="margin: 0;">Number of Rows:</h3></div>""", width=150, height=40),
        Div(text=f"""<div style="display: flex; align-items: center; height: 40px;"><span style='color: #3366cc; font-size: 1.5em; margin: 0;'>{len(df)}</span></div>""", width=100, height=40),
        Spacer(width=20),
        Div(text="""<div style="display: flex; align-items: center; height: 40px;"><h3 style="margin: 0;">Year Range:</h3></div>""", width=150, height=40),
        Div(text=f"""<div style="display: flex; align-items: center; height: 40px;"><span style='color: #3366cc; font-size: 1.5em; margin: 0;'>{df['year'].min()} - {df['year'].max()}</span></div>""", width=200, height=40),
        Spacer(width=20),
        Div(text="""<div style="display: flex; align-items: center; height: 40px;"><h3 style="margin: 0;">Number of Countries:</h3></div>""", width=150, height=40),
        Div(text=f"""<div style="display: flex; align-items: center; height: 40px;"><span style='color: #3366cc; font-size: 1.5em; margin: 0;'>{len(df['country'].unique())}</span></div>""", width=100, height=40),
        sizing_mode="scale_width"
    ), 
    sizing_mode="scale_width"
)

# --------------------------------------
# CO2 EMISSIONS UNIVARIATE ANALYSIS SECTION
# --------------------------------------
co2_title = Div(text="""<h2 style='color: #3366cc;'>Univariate Analysis: CO2</h2>""",
                width=800, height=50)

# 6. CO2 boxplot (Vertical)
co2_stats = df['co2'].describe()
co2_quartiles = co2_stats[['25%', '50%', '75%']].values
co2_iqr = co2_quartiles[2] - co2_quartiles[0]

# Create a single-row dataframe for the boxplot
co2_box_data = pd.DataFrame({
    'category': ['CO2'],
    'lower': [co2_quartiles[0] - 1.5 * co2_iqr],
    'q1': [co2_quartiles[0]],
    'q2': [co2_quartiles[1]],
    'q3': [co2_quartiles[2]],
    'upper': [co2_quartiles[2] + 1.5 * co2_iqr],
    'outlier_min': [df['co2'].min()],
    'outlier_max': [df['co2'].max()]
})

source_co2_box = ColumnDataSource(co2_box_data)

# Create vertical boxplot with no x-ticks
p_co2_box = figure(title="CO2 Emissions (metric tons per capita)",
                   height=400, width=300,
                   # sizing_mode="scale_height",
                   x_range=['CO2'], toolbar_location=None)

# Remove x-axis ticks and labels
p_co2_box.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
p_co2_box.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
p_co2_box.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
p_co2_box.xaxis.visible = False  # This should hide the entire axis

# stems
p_co2_box.segment('category', 'lower', 'category', 'q1', line_width=2, line_color="black", source=source_co2_box)
p_co2_box.segment('category', 'q3', 'category', 'upper', line_width=2, line_color="black", source=source_co2_box)

# boxes
p_co2_box.vbar(x='category', width=0.5, top='q3', bottom='q1', line_width=2, line_color="black", 
               fill_color="#3366cc", alpha=0.6, source=source_co2_box)

# whiskers (almost-0 height rects simpler than segments)
p_co2_box.rect(x='category', y='lower', width=0.1, height=0.1, line_color="black", source=source_co2_box)
p_co2_box.rect(x='category', y='upper', width=0.1, height=0.1, line_color="black", source=source_co2_box)

# median
p_co2_box.rect(x='category', y='q2', width=0.5, height=0.1, line_color="black", fill_color="white", source=source_co2_box)

# Add hover
hover_co2_box = HoverTool()
hover_co2_box.tooltips = [
    ("Min", "@outlier_min{0,0.00}"),
    ("Lower Quartile", "@q1{0,0.00}"),
    ("Median", "@q2{0,0.00}"),
    ("Upper Quartile", "@q3{0,0.00}"),
    ("Max", "@outlier_max{0,0.00}")
]
p_co2_box.add_tools(hover_co2_box)

# CO2 histogram
# Filter out extreme outliers for better visualization
co2_filtered = df[df['co2'] <= co2_quartiles[2] + 3 * co2_iqr]
hist, edges = np.histogram(co2_filtered['co2'], bins=20)
co2_hist_data = pd.DataFrame({
    'count': hist,
    'left': edges[:-1],
    'right': edges[1:]
})
co2_hist_data['interval'] = [f'{left:0.2f} - {right:0.2f}' for left, right in zip(co2_hist_data['left'], co2_hist_data['right'])]

source_co2_hist = ColumnDataSource(co2_hist_data)

p_co2_hist = figure(title="",
                    height=400, width=600,
                    # sizing_mode="scale_height", 
                    x_range=(co2_hist_data['left'].min(), co2_hist_data['right'].max()),
                    toolbar_location=None)

p_co2_hist.quad(top='count', bottom=0, left='left', right='right', source=source_co2_hist,
                fill_color="#3366cc", line_color="white", alpha=0.6)

# Add hover for histogram
hover_co2_hist = HoverTool()
hover_co2_hist.tooltips = [
    ("Interval", "@interval"),
    ("Count", "@count")
]
p_co2_hist.add_tools(hover_co2_hist)

p_co2_hist.xaxis.axis_label = "CO2 Emissions (metric tons per capita)"
p_co2_hist.xaxis.axis_label_text_font_style = 'normal'

p_co2_hist.y_range.start = 0
p_co2_hist.yaxis.axis_label = "Count"
p_co2_hist.yaxis.axis_label_text_font_style = 'normal'

# CO2 min/max table
co2_min_idx = df['co2'].idxmin()
co2_max_idx = df['co2'].idxmax()
co2_extremes = df.loc[[co2_min_idx, co2_max_idx]].copy()
co2_extremes['type'] = ['Minimum CO2', 'Maximum CO2']

source_co2_extremes = ColumnDataSource(co2_extremes)

co2_extreme_columns = [
    TableColumn(field="type", title="Type"),
    TableColumn(field="country", title="Country"),
    TableColumn(field="region", title="Region"),
    TableColumn(field="year", title="Year"),
    TableColumn(field="co2", title="CO2"),
    TableColumn(field="gdp", title="GDP")
]

co2_extremes_table = DataTable(source=source_co2_extremes, columns=co2_extreme_columns, 
                              width=800, height=100, index_position=None)


# CO2 Section - Layout elements
# CO2 box plot to the left of histogram
co2_plots = row(p_co2_box, p_co2_hist, sizing_mode="stretch_width")
co2_section = column(co2_title, co2_plots, co2_extremes_table, sizing_mode="stretch_width")

# --------------------------------------
# GDP UNIVARIATE ANALYSIS SECTION
# --------------------------------------
gdp_title = Div(text="""<h2 style='color: #3366cc;'>Univariate Analysis: GDP</h2>""",
                width=800, height=50)

# 4. GDP boxplot (Vertical)
gdp_stats = df['gdp'].describe()
gdp_quartiles = gdp_stats[['25%', '50%', '75%']].values
gdp_iqr = gdp_quartiles[2] - gdp_quartiles[0]

# Create a single-row dataframe for the boxplot
gdp_box_data = pd.DataFrame({
    'category': ['GDP'],
    'lower': [gdp_quartiles[0] - 1.5 * gdp_iqr],
    'q1': [gdp_quartiles[0]],
    'q2': [gdp_quartiles[1]],
    'q3': [gdp_quartiles[2]],
    'upper': [gdp_quartiles[2] + 1.5 * gdp_iqr],
    'outlier_min': [df['gdp'].min()],
    'outlier_max': [df['gdp'].max()]
})

source_gdp_box = ColumnDataSource(gdp_box_data)

# Create a vertical boxplot with no x-ticks
p_gdp_box = figure(title="GDP (USD per capita)", height=400, width=300,
                   # sizing_mode="scale_height", 
                   x_range=['GDP'], 
                   toolbar_location=None)

# Remove x-axis ticks and labels
p_gdp_box.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
p_gdp_box.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
p_gdp_box.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
p_gdp_box.xaxis.axis_label = ""  # Remove x-axis label
p_gdp_box.xaxis.visible = False  # This should hide the entire axis

p_gdp_box.yaxis.formatter = NumeralTickFormatter(format='0 a')

# stems
p_gdp_box.segment('category', 'lower', 'category', 'q1', line_width=2, line_color="black", source=source_gdp_box)
p_gdp_box.segment('category', 'q3', 'category', 'upper', line_width=2, line_color="black", source=source_gdp_box)

# boxes
p_gdp_box.vbar(x='category', width=0.5, top='q3', bottom='q1', line_width=2, line_color="black", 
               fill_color="#3366cc", alpha=0.6, source=source_gdp_box)

# whiskers (almost-0 height rects simpler than segments)
p_gdp_box.rect(x='category', y='lower', width=0.1, height=0.1, line_color="black", source=source_gdp_box)
p_gdp_box.rect(x='category', y='upper', width=0.1, height=0.1, line_color="black", source=source_gdp_box)

# median
p_gdp_box.rect(x='category', y='q2', width=0.5, height=0.1, line_color="black", fill_color="white", source=source_gdp_box)

# Add hover
hover_gdp_box = HoverTool()
hover_gdp_box.tooltips = [
    ("Min", "@outlier_min{0,0.00}"),
    ("Lower Quartile", "@q1{0,0.00}"),
    ("Median", "@q2{0,0.00}"),
    ("Upper Quartile", "@q3{0,0.00}"),
    ("Max", "@outlier_max{0,0.00}")
]
p_gdp_box.add_tools(hover_gdp_box)

# GDP histogram
# Filter out extreme outliers for better visualization
gdp_filtered = df[df['gdp'] <= gdp_quartiles[2] + 3 * gdp_iqr]
hist, edges = np.histogram(gdp_filtered['gdp'], bins=20)
gdp_hist_data = pd.DataFrame({
    'count': hist,
    'left': edges[:-1],
    'right': edges[1:]
})
gdp_hist_data['interval'] = [f'{left:0.1f} - {right:0.1f}' for left, right in zip(gdp_hist_data['left'], gdp_hist_data['right'])]

source_gdp_hist = ColumnDataSource(gdp_hist_data)

p_gdp_hist = figure(title="", height=400, width=600,
                    # sizing_mode="scale_height", 
                    x_range=(gdp_hist_data['left'].min(), gdp_hist_data['right'].max()),
                    toolbar_location=None)

p_gdp_hist.quad(top='count', bottom=0, left='left', right='right', source=source_gdp_hist,
                fill_color="#3366cc", line_color="white", alpha=0.6)

# Add hover for histogram
hover_gdp_hist = HoverTool()
hover_gdp_hist.tooltips = [
    ("Interval", "@interval"),
    ("Count", "@count")
]
p_gdp_hist.add_tools(hover_gdp_hist)

p_gdp_hist.xaxis.formatter = NumeralTickFormatter(format='0 a')
p_gdp_hist.xaxis.axis_label = "GDP (USD per capita)"
p_gdp_hist.xaxis.axis_label_text_font_style = 'normal'

p_gdp_hist.y_range.start = 0
p_gdp_hist.yaxis.axis_label = "Count"
p_gdp_hist.yaxis.axis_label_text_font_style = 'normal'

# GDP min/max table
gdp_min_idx = df['gdp'].idxmin()
gdp_max_idx = df['gdp'].idxmax()
gdp_extremes = df.loc[[gdp_min_idx, gdp_max_idx]].copy()
gdp_extremes['type'] = ['Minimum GDP', 'Maximum GDP']

source_gdp_extremes = ColumnDataSource(gdp_extremes)

gdp_extreme_columns = [
    TableColumn(field="type", title="Type"),
    TableColumn(field="country", title="Country"),
    TableColumn(field="region", title="Region"),
    TableColumn(field="year", title="Year"),
    TableColumn(field="gdp", title="GDP"),
    TableColumn(field="co2", title="CO2")
]

gdp_extremes_table = DataTable(source=source_gdp_extremes, columns=gdp_extreme_columns, 
                              width=800, height=100, index_position=None)

# GDP Section - Layout elements
# GDP box plot to the left of histogram
gdp_plots = row(p_gdp_box, p_gdp_hist, sizing_mode="stretch_width")
gdp_section = column(gdp_title, gdp_plots, gdp_extremes_table, sizing_mode="stretch_width")


# --------------------------------------
# TIMESERIES SECTION
# --------------------------------------

# Section title
development_title = Div(text="""<h2 style='color: #3366cc;'>Development of CO2 and GDP over Time by Country</h2>""",
                      width=800, height=50)

# Get all unique countries
all_countries = sorted(df['country'].unique().tolist())

# Create MultiChoice widget for country selection
country_selector = MultiChoice(
    title="Select Countries to Highlight:",
    value=[],
    options=all_countries,
    width=400,
    height=100
)

# Create sliders for year selection
years = sorted(df['year'].unique().tolist())
min_year, max_year = min(years), max(years)

year_start_slider = Slider(
    title="Start Year for Slopegraph",
    value=min_year,
    start=min_year,
    end=max_year-1,
    step=1,
    width=400
)

year_end_slider = Slider(
    title="End Year for Slopegraph",
    value=max_year,
    start=min_year+1,
    end=max_year,
    step=1,
    width=400
)

# Ensure initial valid values (end > start)
year_end_slider.value = max(year_end_slider.value, year_start_slider.value + 1)

# Create ColumnDataSource for line plots (initialized empty)
source_lines_co2 = ColumnDataSource(data=dict(
    xs=[], ys=[], country=[], color=[]
))

source_lines_gdp = ColumnDataSource(data=dict(
    xs=[], ys=[], country=[], color=[]
))

# Create ColumnDataSource for selected countries' lines
source_selected_co2 = ColumnDataSource(data=dict(
    xs=[], ys=[], country=[], color=[]
))

source_selected_gdp = ColumnDataSource(data=dict(
    xs=[], ys=[], country=[], color=[]
))

source_co2_points = ColumnDataSource(data=dict(
    x=[], y=[], country=[], color=[]
))

source_gdp_points = ColumnDataSource(data=dict(
    x=[], y=[], country=[], color=[]
))

# Add hover tool for line charts
hover_line = HoverTool(
    tooltips=[
        ("Country", "@country"),
        ("Year", "$x{0}"),
        ("Value", "$y{0.00}")
    ],
    mode="mouse",
    point_policy="follow_mouse",
    line_policy="nearest"
)

# Create line plots

# Create additional ColumnDataSources for end labels
source_co2_end_labels = ColumnDataSource(data=dict(
    x=[], y=[], country=[], color=[]
))

source_gdp_end_labels = ColumnDataSource(data=dict(
    x=[], y=[], country=[], color=[]
))

p_co2_time = figure(title="", height=400, width=800,
                   x_range=(min_year, max_year+6),
                   toolbar_location="above", sizing_mode="scale_width")

# Grey background lines with increased transparency
p_co2_time.multi_line(xs='xs', ys='ys', source=source_lines_co2,
                    line_color='gray', line_alpha=0.1, line_width=1)

# Thicker colored lines for selected countries
p_co2_time.multi_line(xs='xs', ys='ys', source=source_selected_co2,
                    line_color='color', line_width=3)

p_co2_time.scatter(x='x', y='y', source=source_co2_points,
                fill_color='color', line_color='color',
                size=6, alpha=0.7)

# Add text glyphs to the line charts for labels
p_co2_time.text(x='x', y='y', text='country', 
               source=source_co2_end_labels,
               text_font_size='8pt', text_align='left', 
               x_offset=5, y_offset=5,
               text_color='color')

p_co2_time.add_tools(hover_line)
p_co2_time.xaxis.axis_label = "Year"
p_co2_time.yaxis.axis_label_text_font_style = 'normal'
p_co2_time.yaxis.axis_label = "CO2 Emissions (metric tons per capita)"
p_co2_time.xaxis.axis_label_text_font_style = 'normal'

p_gdp_time = figure(title="", height=400, width=800,
                   x_range=(min_year, max_year+6),
                   toolbar_location="above", sizing_mode="scale_width")

p_gdp_time.yaxis.axis_label_text_font_style = 'normal'
p_gdp_time.yaxis.formatter = NumeralTickFormatter(format='0 a')

# Grey background lines with increased transparency
p_gdp_time.multi_line(xs='xs', ys='ys', source=source_lines_gdp,
                    line_color='gray', line_alpha=0.1, line_width=1)

# Thicker colored lines for selected countries
p_gdp_time.multi_line(xs='xs', ys='ys', source=source_selected_gdp,
                    line_color='color', line_width=3)

p_gdp_time.scatter(x='x', y='y', source=source_gdp_points,
                fill_color='color', line_color='color',
                size=6, alpha=0.7)

p_gdp_time.text(x='x', y='y', text='country', 
               source=source_gdp_end_labels,
               text_font_size='8pt', text_align='left', 
               x_offset=5, y_offset=5,
               text_color='color')

p_gdp_time.add_tools(hover_line)
p_gdp_time.xaxis.axis_label = "Year"
p_gdp_time.xaxis.axis_label_text_font_style = 'normal'
p_gdp_time.yaxis.axis_label = "GDP (USD per capita)"
p_gdp_time.yaxis.axis_label_text_font_style = 'normal'

# Create ColumnDataSource for slopegraphs
source_slope_co2 = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[], is_selected=[]
))

source_slope_gdp = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[], is_selected=[]
))

source_slope_co2_selected = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[]
))

source_slope_co2_nonselected = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[]
))

source_slope_gdp_selected = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[]
))

source_slope_gdp_nonselected = ColumnDataSource(data=dict(
    left_year=[], right_year=[], 
    left_value=[], right_value=[],
    country=[], color=[]
))

# Add hover tool for slopegraphs
hover_slope = HoverTool(
    tooltips=[
        ("Country", "@country"),
        ("Start Value", "@left_value{0.00}"),
        ("End Value", "@right_value{0.00}"),
        ("Change", "@{right_value}{+0.00;-0.00} (@{right_value}{+0.0%})")
    ]
)

# Create slopegraph plots
p_slope_co2 = figure(title="", height=300, width=400,
                    toolbar_location="above", sizing_mode="scale_width",
                    y_axis_type="log")  # Log scale for better visualization

p_slope_co2.yaxis.axis_label = "CO2 Emissions (metric tons per capita)"
p_slope_co2.yaxis.axis_label_text_font_style = 'normal'

# Non-selected countries (gray lines)
p_slope_co2.segment(x0=0.1, y0='left_value', x1=0.9, y1='right_value', 
                  line_color='gray', line_alpha=0.1, line_width=1,
                  source=source_slope_co2_nonselected)

# Selected countries (colored lines)
p_slope_co2.segment(x0=0.1, y0='left_value', x1=0.9, y1='right_value', 
                  line_color='color', line_alpha=1, line_width=3,
                  source=source_slope_co2_selected)

p_slope_co2.scatter(x=0.1, y='left_value', source=source_slope_co2_selected,
                fill_color='color', line_color='color',
                size=10, alpha=1.0)

p_slope_co2.scatter(x=0.9, y='right_value', source=source_slope_co2_selected,
                fill_color='color', line_color='color',
                size=10, alpha=1.0)

# Only add labels for selected countries
p_slope_co2.text(x=0.1, y='left_value', text='country', 
               source=source_slope_co2_selected,
               text_font_size='8pt', text_align='right', x_offset=-10,
               text_color='color')

p_slope_co2.text(x=0.9, y='right_value', text='country', 
               source=source_slope_co2_selected,
               text_font_size='8pt', text_align='left', x_offset=10,
               text_color='color')

p_slope_co2.add_tools(hover_slope)

# Remove x-axis as it's not needed
p_slope_co2.xaxis.visible = False
p_slope_co2.x_range.start = -0.1
p_slope_co2.x_range.end = 1.1

# Create GDP slopegraph
p_slope_gdp = figure(title="", height=300, width=400,
                    toolbar_location="above", sizing_mode="scale_width",
                    y_axis_type="log")  # Log scale for better visualization

p_slope_gdp.yaxis.axis_label = "GDP (USD per capita)"
p_slope_gdp.yaxis.axis_label_text_font_style = 'normal'

# Non-selected countries (gray lines)
p_slope_gdp.segment(x0=0.1, y0='left_value', x1=0.9, y1='right_value', 
                  line_color='gray', line_alpha=0.1, line_width=1,
                  source=source_slope_gdp_nonselected)

# Selected countries (colored lines)
p_slope_gdp.segment(x0=0.1, y0='left_value', x1=0.9, y1='right_value', 
                  line_color='color', line_alpha=1, line_width=3,
                  source=source_slope_gdp_selected)

p_slope_gdp.scatter(x=0.1, y='left_value', source=source_slope_gdp_selected,
                fill_color='color', line_color='color',
                size=10, alpha=1.0)

p_slope_gdp.scatter(x=0.9, y='right_value', source=source_slope_gdp_selected,
                fill_color='color', line_color='color',
                size=10, alpha=1.0)

# Only add labels for selected countries
p_slope_gdp.text(x=0.1, y='left_value', text='country', 
               source=source_slope_gdp_selected,
               text_font_size='8pt', text_align='right', x_offset=-10,
               text_color='color')

p_slope_gdp.text(x=0.9, y='right_value', text='country', 
               source=source_slope_gdp_selected,
               text_font_size='8pt', text_align='left', x_offset=10,
               text_color='color')

p_slope_gdp.add_tools(hover_slope)

# Remove x-axis as it's not needed
p_slope_gdp.xaxis.visible = False
p_slope_gdp.x_range.start = -0.1
p_slope_gdp.x_range.end = 1.1

# Create Div elements to display change values

co2_increase = Div(text="", width_policy='max', height=120, sizing_mode="stretch_width")
co2_decrease = Div(text="", width_policy='max', height=120, sizing_mode="stretch_width")

gdp_increase = Div(text="", width_policy='max', height=120, sizing_mode="stretch_width")
gdp_decrease = Div(text="", width_policy='max', height=120, sizing_mode="stretch_width")

# Function to update data based on selection
def update_data():
    global co2_view, gdp_view, non_selected_co2_view, non_selected_gdp_view
    # Colors for selected countries
    colors = Category10[10]
    
    # Prepare all country lines (gray)
    xs_co2 = []
    ys_co2 = []
    countries_co2 = []
    
    xs_gdp = []
    ys_gdp = []
    countries_gdp = []
    
    # Prepare selected country lines (colored)
    xs_sel_co2 = []
    ys_sel_co2 = []
    countries_sel_co2 = []
    colors_sel_co2 = []
    
    xs_sel_gdp = []
    ys_sel_gdp = []
    countries_sel_gdp = []
    colors_sel_gdp = []
    
    # Get selected countries
    selected_countries = country_selector.value
    
    # Dictionary to store country to color mapping
    country_colors = {}
    for i, country in enumerate(selected_countries):
        color_idx = i % len(colors)
        country_colors[country] = colors[color_idx]
    
    # Prepare data for all countries
    for country in all_countries:
        country_data = df[df['country'] == country]
        
        if len(country_data) > 0:
            # Sort by year
            country_data = country_data.sort_values('year')
            
            # CO2 data
            x_co2 = country_data['year'].tolist()
            y_co2 = country_data['co2'].tolist()
            
            # GDP data
            x_gdp = country_data['year'].tolist()
            y_gdp = country_data['gdp'].tolist()
            
            if country in selected_countries:
                # Add to selected sources with color
                xs_sel_co2.append(x_co2)
                ys_sel_co2.append(y_co2)
                countries_sel_co2.append(country)
                colors_sel_co2.append(country_colors[country])
                
                xs_sel_gdp.append(x_gdp)
                ys_sel_gdp.append(y_gdp)
                countries_sel_gdp.append(country)
                colors_sel_gdp.append(country_colors[country])
            else:
                # Add to gray sources
                xs_co2.append(x_co2)
                ys_co2.append(y_co2)
                countries_co2.append(country)
                
                xs_gdp.append(x_gdp)
                ys_gdp.append(y_gdp)
                countries_gdp.append(country)
    
    # Update line plot sources
    source_lines_co2.data = {
        'xs': xs_co2,
        'ys': ys_co2,
        'country': countries_co2
    }
    
    source_lines_gdp.data = {
        'xs': xs_gdp,
        'ys': ys_gdp,
        'country': countries_gdp
    }
    
    source_selected_co2.data = {
        'xs': xs_sel_co2,
        'ys': ys_sel_co2,
        'country': countries_sel_co2,
        'color': colors_sel_co2
    }
    
    source_selected_gdp.data = {
        'xs': xs_sel_gdp,
        'ys': ys_sel_gdp,
        'country': countries_sel_gdp,
        'color': colors_sel_gdp
    }

    # Flatten multi-line data into points for the line charts
    co2_points_x = []
    co2_points_y = []
    co2_points_country = []
    co2_points_color = []

    gdp_points_x = []
    gdp_points_y = []
    gdp_points_country = []
    gdp_points_color = []

    # Process selected countries for points
    for i, country in enumerate(countries_sel_co2):
        for j in range(len(xs_sel_co2[i])):
            if j < len(ys_sel_co2[i]):  # Check to avoid index errors
                co2_points_x.append(xs_sel_co2[i][j])
                co2_points_y.append(ys_sel_co2[i][j])
                co2_points_country.append(country)
                co2_points_color.append(colors_sel_co2[i])

    for i, country in enumerate(countries_sel_gdp):
        for j in range(len(xs_sel_gdp[i])):
            if j < len(ys_sel_gdp[i]):  # Check to avoid index errors
                gdp_points_x.append(xs_sel_gdp[i][j])
                gdp_points_y.append(ys_sel_gdp[i][j])
                gdp_points_country.append(country)
                gdp_points_color.append(colors_sel_gdp[i])

    # Update the points sources
    source_co2_points.data = {
        'x': co2_points_x,
        'y': co2_points_y,
        'country': co2_points_country,
        'color': co2_points_color
    }

    source_gdp_points.data = {
        'x': gdp_points_x,
        'y': gdp_points_y,
        'country': gdp_points_country,
        'color': gdp_points_color
    }

    # Prepare data for end labels on line charts
    co2_end_x = []
    co2_end_y = []
    co2_end_countries = []
    co2_end_colors = []

    gdp_end_x = []
    gdp_end_y = []
    gdp_end_countries = []
    gdp_end_colors = []

    for i, country in enumerate(countries_sel_co2):
        if len(xs_sel_co2[i]) > 0:
            # Get the last point for each line
            co2_end_x.append(xs_sel_co2[i][-1])
            co2_end_y.append(ys_sel_co2[i][-1])
            co2_end_countries.append(country)
            co2_end_colors.append(colors_sel_co2[i])

    for i, country in enumerate(countries_sel_gdp):
        if len(xs_sel_gdp[i]) > 0:
            # Get the last point for each line
            gdp_end_x.append(xs_sel_gdp[i][-1])
            gdp_end_y.append(ys_sel_gdp[i][-1])
            gdp_end_countries.append(country)
            gdp_end_colors.append(colors_sel_gdp[i])

    # Update the end label sources
    source_co2_end_labels.data = {
        'x': co2_end_x,
        'y': co2_end_y,
        'country': co2_end_countries,
        'color': co2_end_colors
    }

    source_gdp_end_labels.data = {
        'x': gdp_end_x,
        'y': gdp_end_y,
        'country': gdp_end_countries,
        'color': gdp_end_colors
    }
    
    # Get years for slopegraph
    start_year = year_start_slider.value
    end_year = year_end_slider.value
    
    # Update slopegraph titles with selected years
    p_slope_co2.title.text = f"CO2 Emissions Change from {start_year} to {end_year}"
    p_slope_gdp.title.text = f"GDP Change from {start_year} to {end_year}"
    
    # Prepare slopegraph data
    left_year_co2 = []
    right_year_co2 = []
    left_value_co2 = []
    right_value_co2 = []
    countries_slope_co2 = []
    colors_slope_co2 = []
    is_selected_co2 = []
    
    left_year_gdp = []
    right_year_gdp = []
    left_value_gdp = []
    right_value_gdp = []
    countries_slope_gdp = []
    colors_slope_gdp = []
    is_selected_gdp = []
    
    for country in all_countries:
        country_data = df[df['country'] == country]
        
        # Check if data exists for both years
        start_data = country_data[country_data['year'] == start_year]
        end_data = country_data[country_data['year'] == end_year]
        
        if len(start_data) > 0 and len(end_data) > 0:
            # CO2 data
            co2_start = start_data['co2'].values[0]
            co2_end = end_data['co2'].values[0]
            
            # GDP data
            gdp_start = start_data['gdp'].values[0]
            gdp_end = end_data['gdp'].values[0]
            
            # Skip if values are zero or NaN (would cause issues with log scale)
            if co2_start > 0 and co2_end > 0 and not np.isnan(co2_start) and not np.isnan(co2_end):
                left_year_co2.append(start_year)
                right_year_co2.append(end_year)
                left_value_co2.append(co2_start)
                right_value_co2.append(co2_end)
                countries_slope_co2.append(country)
                
                # Set color and selected flag
                if country in selected_countries:
                    colors_slope_co2.append(country_colors[country])
                    is_selected_co2.append(1)  # Selected
                else:
                    colors_slope_co2.append('gray')
                    is_selected_co2.append(0)  # Not selected
            
            # Skip if values are zero or NaN (would cause issues with log scale)
            if gdp_start > 0 and gdp_end > 0 and not np.isnan(gdp_start) and not np.isnan(gdp_end):
                left_year_gdp.append(start_year)
                right_year_gdp.append(end_year)
                left_value_gdp.append(gdp_start)
                right_value_gdp.append(gdp_end)
                countries_slope_gdp.append(country)
                
                # Set color and selected flag
                if country in selected_countries:
                    colors_slope_gdp.append(country_colors[country])
                    is_selected_gdp.append(1)  # Selected
                else:
                    colors_slope_gdp.append('gray')
                    is_selected_gdp.append(0)  # Not selected
    
    # Split data into selected and non-selected for CO2
    selected_co2_data = {
        'left_year': [],
        'right_year': [],
        'left_value': [],
        'right_value': [],
        'country': [],
        'color': []
    }
    
    nonselected_co2_data = {
        'left_year': [],
        'right_year': [],
        'left_value': [],
        'right_value': [],
        'country': [],
        'color': []
    }

    for i, country in enumerate(countries_slope_co2):
        if is_selected_co2[i] == 1:
            selected_co2_data['left_year'].append(left_year_co2[i])
            selected_co2_data['right_year'].append(right_year_co2[i])
            selected_co2_data['left_value'].append(left_value_co2[i])
            selected_co2_data['right_value'].append(right_value_co2[i])
            selected_co2_data['country'].append(country)
            selected_co2_data['color'].append(colors_slope_co2[i])
        else:
            nonselected_co2_data['left_year'].append(left_year_co2[i])
            nonselected_co2_data['right_year'].append(right_year_co2[i])
            nonselected_co2_data['left_value'].append(left_value_co2[i])
            nonselected_co2_data['right_value'].append(right_value_co2[i])
            nonselected_co2_data['country'].append(country)
            nonselected_co2_data['color'].append(colors_slope_co2[i])
    
    source_slope_co2_selected.data = selected_co2_data
    source_slope_co2_nonselected.data = nonselected_co2_data

    selected_gdp_data = {
        'left_year': [],
        'right_year': [],
        'left_value': [],
        'right_value': [],
        'country': [],
        'color': []
    }
    
    nonselected_gdp_data = {
        'left_year': [],
        'right_year': [],
        'left_value': [],
        'right_value': [],
        'country': [],
        'color': []
    }
    
    for i, country in enumerate(countries_slope_gdp):
        if is_selected_gdp[i] == 1:
            selected_gdp_data['left_year'].append(left_year_gdp[i])
            selected_gdp_data['right_year'].append(right_year_gdp[i])
            selected_gdp_data['left_value'].append(left_value_gdp[i])
            selected_gdp_data['right_value'].append(right_value_gdp[i])
            selected_gdp_data['country'].append(country)
            selected_gdp_data['color'].append(colors_slope_gdp[i])
        else:
            nonselected_gdp_data['left_year'].append(left_year_gdp[i])
            nonselected_gdp_data['right_year'].append(right_year_gdp[i])
            nonselected_gdp_data['left_value'].append(left_value_gdp[i])
            nonselected_gdp_data['right_value'].append(right_value_gdp[i])
            nonselected_gdp_data['country'].append(country)
            nonselected_gdp_data['color'].append(colors_slope_gdp[i])
    
    source_slope_gdp_selected.data = selected_gdp_data
    source_slope_gdp_nonselected.data = nonselected_gdp_data

    # Find countries with strongest CO2 changes
    if len(countries_slope_co2) > 0:
        # Calculate percentage changes
        co2_changes = []
        for i in range(len(countries_slope_co2)):
            start_val = left_value_co2[i]
            end_val = right_value_co2[i]
            if start_val > 0:  # Avoid division by zero
                pct_change = ((end_val - start_val) / start_val) * 100
                absolute_change = end_val - start_val
                co2_changes.append({
                    'country': countries_slope_co2[i],
                    'start_val': start_val,
                    'end_val': end_val,
                    'pct_change': pct_change,
                    'abs_change': absolute_change
                })
        
        # Sort by percentage change
        co2_changes_sorted = sorted(co2_changes, key=lambda x: x['pct_change'])
        
        # Get largest decrease and increase
        if len(co2_changes_sorted) > 0:
            co2_largest_decrease = co2_changes_sorted[0]
            co2_largest_increase = co2_changes_sorted[-1]
            
            # Format the results
            co2_decrease.text = f"""
            <div style='background-color: #f8f8f8; padding: 10px; border-radius: 5px; border-left: 4px solid #000000;'>
            <h4>Largest CO2 Decrease from {start_year} to {end_year}:</h4>
            <p><b>{co2_largest_decrease['country']}</b>: {co2_largest_decrease['start_val']:.2f} to {co2_largest_decrease['end_val']:.2f} metric tons per capita</p>
            <p>Change: {'+' if co2_largest_decrease['abs_change'] > 0 else ''}{co2_largest_decrease['abs_change']:.2f} ({'+' if co2_largest_decrease['pct_change'] > 0 else ''}{co2_largest_decrease['pct_change']:.1f}%)</p>
            </div>
            """
            
            co2_increase.text = f"""
            <div style='background-color: #f8f8f8; padding: 10px; border-radius: 5px; border-left: 4px solid #000000;'>
            <h4>Largest CO2 Increase from {start_year} to {end_year}:</h4>
            <p><b>{co2_largest_increase['country']}</b>: {co2_largest_increase['start_val']:.2f} to {co2_largest_increase['end_val']:.2f} metric tons per capita</p>
            <p>Change: {'+' if co2_largest_increase['abs_change'] > 0 else ''}{co2_largest_increase['abs_change']:.2f} ({'+' if co2_largest_increase['pct_change'] > 0 else ''}{co2_largest_increase['pct_change']:.1f}%)</p>
            </div>
            """
        else:
            co2_decrease.text = "<p>No valid data available for the selected year range</p>"
            co2_increase.text = "<p>No valid data available for the selected year range</p>"
    else:
        co2_decrease.text = "<p>No data available for the selected year range</p>"
        co2_increase.text = "<p>No data available for the selected year range</p>"

    # Find countries with strongest GDP changes
    if len(countries_slope_gdp) > 0:
        # Calculate percentage changes
        gdp_changes = []
        for i in range(len(countries_slope_gdp)):
            start_val = left_value_gdp[i]
            end_val = right_value_gdp[i]
            if start_val > 0:  # Avoid division by zero
                pct_change = ((end_val - start_val) / start_val) * 100
                absolute_change = end_val - start_val
                gdp_changes.append({
                    'country': countries_slope_gdp[i],
                    'start_val': start_val,
                    'end_val': end_val,
                    'pct_change': pct_change,
                    'abs_change': absolute_change
                })
        
        # Sort by percentage change
        gdp_changes_sorted = sorted(gdp_changes, key=lambda x: x['pct_change'])
        
        # Get largest decrease and increase
        if len(gdp_changes_sorted) > 0:
            gdp_largest_decrease = gdp_changes_sorted[0]
            gdp_largest_increase = gdp_changes_sorted[-1]
            
            # Format the results
            gdp_decrease.text = f"""
            <div style='background-color: #f8f8f8; padding: 10px; border-radius: 5px; border-left: 4px solid #000000;'>
            <h4>Largest GDP Decrease from {start_year} to {end_year}:</h4>
            <p><b>{gdp_largest_decrease['country']}</b>: {gdp_largest_decrease['start_val']:.2f} to {gdp_largest_decrease['end_val']:.2f} USD</p>
            <p>Change: {'+' if gdp_largest_decrease['abs_change'] > 0 else ''}{gdp_largest_decrease['abs_change']:.2f} ({'+' if gdp_largest_decrease['pct_change'] > 0 else ''}{gdp_largest_decrease['pct_change']:.1f}%)</p>
            </div>
            """
            
            gdp_increase.text = f"""
            <div style='background-color: #f8f8f8; padding: 10px; border-radius: 5px; border-left: 4px solid #000000;'>
            <h4>Largest GDP Increase from {start_year} to {end_year}:</h4>
            <p><b>{gdp_largest_increase['country']}</b>: {gdp_largest_increase['start_val']:.2f} to {gdp_largest_increase['end_val']:.2f} USD</p>
            <p>Change: {'+' if gdp_largest_increase['abs_change'] > 0 else ''}{gdp_largest_increase['abs_change']:.2f} ({'+' if gdp_largest_increase['pct_change'] > 0 else ''}{gdp_largest_increase['pct_change']:.1f}%)</p>
            </div>
            """
        else:
            gdp_decrease.text = "<p>No valid data available for the selected year range</p>"
            gdp_increase.text = "<p>No valid data available for the selected year range</p>"
    else:
        gdp_decrease.text = "<p>No data available for the selected year range</p>"
        gdp_increase.text = "<p>No data available for the selected year range</p>"
    

# Set up callbacks for widgets
country_selector.on_change('value', lambda attr, old, new: update_data())
year_start_slider.on_change('value', lambda attr, old, new: update_data())
year_end_slider.on_change('value', lambda attr, old, new: update_data())

# Add callback to ensure end year is always greater than start year
def update_year_range(attr, old, new):
    if year_end_slider.value <= year_start_slider.value:
        year_end_slider.value = year_start_slider.value + 1

year_start_slider.on_change('value', update_year_range)

# Create a layout for the development section
year_slider_row = row(column(year_start_slider, width=400), Spacer(width=100), column(year_end_slider, width=400), sizing_mode="stretch_width")

# Timeseries section - Layout elements
development_section = column(
    development_title,
    country_selector,
    p_co2_time,
    p_gdp_time,
    year_slider_row,
    row(p_slope_co2, p_slope_gdp, sizing_mode="stretch_width"),
    row(
        column(
            co2_increase, 
            co2_decrease,
            width_policy='max', 
            sizing_mode="stretch_width"
        ),
        column(
            gdp_increase, 
            gdp_decrease,
            width_policy='max',
            sizing_mode="stretch_width"
        ), 
        sizing_mode="stretch_width"
    ),
    sizing_mode="stretch_width"
)

# Initialize the data
update_data()

# --------------------------------------
# THE BY YEAR SECTION
# --------------------------------------
# Section title
byyear_title = Div(text="""<h2 style='color: #3366cc;'>CO2 Emissions and GDP by Year</h2>""",
                      width=800, height=30)

# Create slider
byyear_slider = Slider(
    start=min_year, 
    end=max_year, 
    step=1, 
    value=min_year, 
    title='Year'
)

# Create radio button group for choropleth metric selection
metric_select = RadioButtonGroup(
    labels=["CO2 Emissions", "GDP"], 
    active=0,  # Default to CO2 emissions
)

# Create ColumnDataSource for line plots (initialized empty)
source_co2_gdp_byyear = ColumnDataSource(data=dict(
    co2=[], gdp=[], country=[], region=[]
))

# Create ColumnDataSource for the regional bar charts
source_co2_gdp_byregion = ColumnDataSource(data=dict(
    region=[], co2=[], gdp=[]
))

# Create empty GeoJSONDataSource for the choropleth map
geo_source = GeoJSONDataSource(geojson='{"type":"FeatureCollection","features":[]}')

# Get unique regions for categorical coloring
all_regions = df['region'].unique().tolist()

# Create a color mapper
regions_color_mapper = CategoricalColorMapper(
    factors=all_regions,
    palette=Category10[len(all_regions)]
)

# Color mapper for the choropleth
reversed_reds = list(reversed(Reds8))
cp_color_mapper = LogColorMapper(palette=reversed_reds, low=0.1)

# Create figure
p_co2_gdp_byyear = figure(
    height=400, width=800, sizing_mode="scale_width",
    # tools=[BoxZoomTool(), WheelZoomTool(), HoverTool(), ResetTool()]
)

# add a scatterplot
p_co2_gdp_byyear.scatter(source=source_co2_gdp_byyear, x='gdp', y='co2',
          size=10, fill_alpha=0.5,
          color={'field': 'region', 'transform': regions_color_mapper},  # Color by region
          legend_field='region'  # Add legend using region field
        )

# Initialise the title and set formatting
p_co2_gdp_byyear.title = Title(
    text='CO2 Emissions vs GDP in selected year',
    # align='center',  # Center the title
    # text_font_size='16pt'  # Set font size
)

# Axis labels...
p_co2_gdp_byyear.xaxis.axis_label = "GDP (in USD per capita)"

# Changing to log scale
p_co2_gdp_byyear.y_scale = LogScale()
# and formatting the tick labels
p_co2_gdp_byyear.yaxis.ticker = LogTicker()
p_co2_gdp_byyear.yaxis.formatter = LogTickFormatter()
# p_co2_gdp_byyear.yaxis.major_label_text_font_size = '14px'

p_co2_gdp_byyear.yaxis.axis_label = "CO2 emissions (metric tons per capita)"
# ... and styling:
p_co2_gdp_byyear.yaxis.axis_label_text_font_style = 'normal'
# p_co2_gdp_byyear.yaxis.axis_label_text_font_size = '18pt'


p_co2_gdp_byyear.xaxis.formatter = NumeralTickFormatter(format='0 a')
# p_co2_gdp_byyear.xaxis.major_label_text_font_size = '14px'
# p_co2_gdp_byyear.xaxis.axis_label_text_font_size = '18pt'
p_co2_gdp_byyear.xaxis.axis_label_text_font_style = 'normal'

# Customise the legend
p_co2_gdp_byyear.legend.title = "Region"
p_co2_gdp_byyear.legend.title_text_font_style = "bold"
p_co2_gdp_byyear.legend.location = "top_right"
p_co2_gdp_byyear.legend.label_text_font_size = "10pt"
p_co2_gdp_byyear.legend.background_fill_alpha = 0.5  # Semi-transparent background

# Add hover tool for better interactivity
hover_tool = HoverTool(tooltips=[
    ("Country", "@country"),
    ("Region", "@region"),
    ("CO2", "@co2{0.00}"),
    ("GDP", "@gdp{0,0.00}")
])
p_co2_gdp_byyear.add_tools(hover_tool)


# Create choropleth map if we have the geo data
p_choropleth = None
countries_renderer = None
color_bar = None

# Function to load and merge world GeoJSON data
def load_geo_data():
    """Load and prepare world GeoJSON data with proper handling of MultiPolygons"""
    
    try:
        # Create a temporary directory to store the downloaded and extracted files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download the zip file
            response = requests.get(url_geo_data)
            
            if response.status_code != 200:
                raise Exception(f"Failed to download: Status code {response.status_code}")
            
            # Save the zip file to the temporary directory
            zip_path = os.path.join(temp_dir, "geo_data.zip")
            with open(zip_path, "wb") as f:
                f.write(response.content)
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the .shp file in the extracted contents
            shapefile_path = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".shp"):
                        shapefile_path = os.path.join(root, file)
                        break
                if shapefile_path:
                    break
            
            if not shapefile_path:
                raise Exception("No .shp file found in the downloaded zip.")
            
            # Load the shapefile with GeoPandas
            world = gpd.read_file(shapefile_path)
            
            # Rename the country column if needed
            if 'NAME' in world.columns:
                world = world.rename(columns={'NAME': 'country'})
            elif 'name' in world.columns:
                world = world.rename(columns={'name': 'country'})
            
    except Exception as e:
        raise Exception(f"Error loading geo data: {e}")
    
    # Create empty lists for coordinates
    xs_list = []
    ys_list = []
    
    # Process each geometry to handle both Polygon and MultiPolygon types
    for idx, row in world.iterrows():
        geom = row.geometry
        
        # Handle different geometry types
        if geom.geom_type == 'Polygon':
            # For simple polygons, extract exterior coordinates
            xs, ys = list(geom.exterior.coords.xy[0]), list(geom.exterior.coords.xy[1])
        elif geom.geom_type == 'MultiPolygon':
            # For multipolygons, use the largest polygon (by area)
            largest_poly = max(geom.geoms, key=lambda p: p.area)
            xs, ys = list(largest_poly.exterior.coords.xy[0]), list(largest_poly.exterior.coords.xy[1])
        else:
            # Empty lists for other geometry types
            xs, ys = [], []
            
        xs_list.append(xs)
        ys_list.append(ys)
    
    # Add coordinate columns to the dataframe
    world['xs'] = xs_list
    world['ys'] = ys_list
    
    return world

# Try to load the geo data - initialize variables
world_geo = None
has_geo_data = False
geo_source = None

try:
    # Load the world geography data
    world_geo = load_geo_data()
    has_geo_data = True
    
    # Initialize the GeoJSONDataSource with a valid but empty GeoJSON structure
    geo_source = GeoJSONDataSource(geojson='{"type":"FeatureCollection","features":[]}')
except Exception as e:
    print(f"Warning: Could not load geographic data: {e}")
    print("Choropleth map will not be displayed.")

if has_geo_data:
    p_choropleth = figure(
        height=400, width=800, sizing_mode="scale_width",
        title=f"CO2 Emissions by Country in {min_year}",
        toolbar_location="above",
        x_axis_location=None, 
        y_axis_location=None
    )

    # Add country polygons to the choropleth
    countries_renderer = p_choropleth.patches(
        'xs', 'ys', 
        source=geo_source,
        fill_color={'field': 'value', 'transform': cp_color_mapper},
        line_color='black', 
        line_width=0.5, 
        fill_alpha=0.7
    )

    # Add hover for the choropleth
    choropleth_hover = HoverTool(renderers=[countries_renderer], tooltips=[
        ("Country", "@name"),
        ("Value", "@value{0.00}")
    ])
    p_choropleth.add_tools(choropleth_hover)

    # Add color bar to the choropleth
    color_bar = ColorBar(
        color_mapper=cp_color_mapper, 
        label_standoff=12, 
        border_line_color=None, 
        location=(0, 0)
    )
    p_choropleth.add_layout(color_bar, 'right')



# Create CO2 horizontal bar chart
p_co2_byregion = figure(
    height=300, width=600, sizing_mode="scale_width",
    title=f"Average CO2 Emissions by Region in {min_year}",
    x_axis_label="CO2 Emissions (Average in metric tons per capita)",
    y_range=all_regions,  # Will be dynamically updated
    toolbar_location=None
)

p_co2_byregion.xaxis.axis_label_text_font_style = 'normal'

p_co2_byregion.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
p_co2_byregion.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

# Add the horizontal bars for CO2
p_co2_byregion.hbar(
    y='region', 
    right='co2', 
    height=0.8, 
    source=source_co2_gdp_byregion,
    fill_color={'field': 'region', 'transform': regions_color_mapper},
    line_color="white",
    alpha=0.8
)

# Add hover tool for CO2 bar chart
co2_bar_hover = HoverTool(tooltips=[
    ("Region", "@region"),
    ("Avg CO2", "@co2{0.00}")
])
p_co2_byregion.add_tools(co2_bar_hover)


# Create GDP horizontal bar chart with same y_range to align with CO2 chart
p_gdp_byregion = figure(
    height=300, width=400, sizing_mode="scale_width",
    title=f"Average GDP by Region in {min_year}",
    x_axis_label="GDP (Average in USD per capita)",
    y_range=p_co2_byregion.y_range,  # Share y_range with CO2 chart
    toolbar_location=None
)

p_gdp_byregion.xaxis.axis_label_text_font_style = 'normal'

# Add the horizontal bars for GDP
p_gdp_byregion.hbar(
    y='region', 
    right='gdp', 
    height=0.8, 
    source=source_co2_gdp_byregion,
    fill_color={'field': 'region', 'transform': regions_color_mapper},
    line_color="white",
    alpha=0.8
)

# Add hover tool for GDP bar chart
gdp_bar_hover = HoverTool(tooltips=[
    ("Region", "@region"),
    ("Avg GDP", "@gdp{0,0}")
])
p_gdp_byregion.add_tools(gdp_bar_hover)

# Hide y-axis labels on GDP chart since they're the same as CO2 chart and regions are in the same order
p_gdp_byregion.yaxis.visible = False




def update_byyear_data(attr, old, new):
    """Update the scatter plot, choropleth and bar charts based on the selected year"""

    year = byyear_slider.value

    # Update the plot titles to show the selected year
    p_co2_gdp_byyear.title.text = f"GDP vs CO2 Emissions by Country in {year}"
    p_co2_byregion.title.text = f"Average CO2 Emissions by Region in {year}"
    p_gdp_byregion.title.text = f"Average GDP by Region in {year}"
    
    # Get the selected metric (0 = CO2, 1 = GDP)
    metric_idx = metric_select.active
    metric_name = "CO2 Emissions" if metric_idx == 0 else "GDP"


    # Prepare data for scatter plot
    co2_values = []
    gdp_values = []
    countries = []
    regions = []
    
    year_data = df[df['year'] == year]

    for idx, row in year_data.iterrows():
        co2_values.append(row['co2'])
        gdp_values.append(row['gdp'])
        countries.append(row['country'])
        regions.append(row['region'])
    
    # Update the scatter plot data source
    source_co2_gdp_byyear.data = {
        'co2': co2_values,
        'gdp': gdp_values,
        'country': countries,
        'region': regions
    }

    # Calculate regional averages
    region_data = {}
    for region in all_regions:
        region_rows = year_data[year_data['region'] == region]
        if len(region_rows) > 0:
            region_data[region] = {
                'co2': region_rows['co2'].mean(),
                'gdp': region_rows['gdp'].mean()
            }
        else:
            region_data[region] = {
                'co2': 0,
                'gdp': 0
            }
    
    # Convert to sorted lists (descending by CO2)
    sorted_regions = sorted(region_data.keys(), key=lambda r: region_data[r]['co2'], reverse=False)
    sorted_co2 = [region_data[r]['co2'] for r in sorted_regions]
    sorted_gdp = [region_data[r]['gdp'] for r in sorted_regions]
    
    # Update the bar charts data source
    source_co2_gdp_byregion.data = {
        'region': sorted_regions,
        'co2': sorted_co2,
        'gdp': sorted_gdp
    }
    
    # Update the y_range to reflect the new order of regions sorted by CO2
    p_co2_byregion.y_range.factors = sorted_regions

    # Update choropleth if geo data is available
    if has_geo_data and world_geo is not None and geo_source is not None:
        # Get the selected metric (0 = CO2, 1 = GDP)
        metric_idx = metric_select.active
        metric_name = "CO2 Emissions" if metric_idx == 0 else "GDP"
        p_choropleth.title.text = f"{metric_name} by Country in {year}"
        
        # Create a copy of the world GeoJSON data
        gdf = world_geo.copy()
        
        # Create a dictionary mapping country names to values
        country_to_value = {}
        for idx, row in year_data.iterrows():
            # Get the value based on selected metric
            value = row['co2'] if metric_idx == 0 else row['gdp']
            # Ensure all values are positive (required for log scale)
            country_to_value[row['country']] = max(value, 0.1)

        # Merge the data with the GeoJSON
        gdf['value'] = gdf['country'].apply(lambda c: country_to_value.get(c, 0))
        gdf['name'] = gdf['country']  # For the hover tooltip
        
        # Set the color mapper range based on the selected metric
        max_val = max(co2_values) if metric_idx == 0 else max(gdp_values)
        max_val = max(max_val, 1)  # Ensure we have a valid maximum

        # Update the mapper's high value
        cp_color_mapper.high = max_val

        # Convert to GeoJSON and update the source
        geo_json = json.loads(gdf.to_json())
        geo_source.geojson = json.dumps(geo_json)

def update_choropleth_metric(attr, old, new):
    """Update the choropleth based on the selected metric"""
    # This will trigger a full update including the choropleth
    update_byyear_data(attr, old, new)

# Set up callbacks for widgets - pass the function, don't call it
byyear_slider.on_change('value', update_byyear_data)
if has_geo_data:
    metric_select.on_change('active', update_choropleth_metric)

# Initialize the by year data
update_byyear_data(None, None, None)

# Create layout components
metric_select_row = None
choropleth_row = None

if has_geo_data:
    # Create section with the metric selector
    metric_select_row = row(
        Div(text="<b>Select choropleth metric:</b>", width=200, height=30),
        metric_select,
        sizing_mode="stretch_width"
    )
    
    # Create row for choropleth
    choropleth_row = row(p_choropleth, sizing_mode="stretch_width")

# Create layout for the bar charts
bar_charts_row = row(p_co2_byregion, p_gdp_byregion, sizing_mode="stretch_width")

# Create layout for the by-year section - add choropleth components only if available
components = [
    byyear_title,
    row(byyear_slider, sizing_mode="stretch_width"),
    row(p_co2_gdp_byyear, sizing_mode="stretch_width"),
]

if has_geo_data and metric_select_row is not None and choropleth_row is not None:
    components.append(metric_select_row)
    components.append(choropleth_row)

components.append(bar_charts_row)

byyear_section = column(
    *components,
    sizing_mode="stretch_width"
)

# --------------------------------------
# THE COMPLETE LAYOUT
# --------------------------------------
# Put everything together
dashboard = column(
    title,
    overview_section,
    co2_section,
    gdp_section,
    development_section,
    byyear_section,
    sizing_mode="stretch_width"
)

# Add the layout to the current document
curdoc().add_root(dashboard)
curdoc().title = "CO2 Emissions Dashboard"