#%% [markdown]
## Aspect Ratio STUDY 2019 
### Evolution of the Aspect ratio
#Source are :
# - archive of Para2000 
# - and gliderbase ( may 2019)
#
#
# Work in progress (sometime !).
# Develloped with python 3.7 with Anaconda.
# Used as a jupyter notebook , in visual studio code.
#
#Fred

#%% Change working directory 
import os
try:
	os.chdir(os.path.join(os.getcwd()))
	print(os.getcwd())
except:
	pass

#%%  Importing lib ...
import sys

import numpy as np

import pandas as pd
from pathlib import Path

import math

from bokeh.io import push_notebook, show, output_notebook
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool , LabelSet ,LegendItem ,Legend
from bokeh.palettes import Category20,Spectral11,cividis
from bokeh.transform import factor_cmap ,jitter

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns



#%% Defining usefull function
def print_group(grouped_df):
	for key, item in grouped_df:
		print(key)
		print(grouped_df.get_group(key), "\n\n")

#%% Files and Settings
#Main parameters :
df_csv_file_all = os.path.join(str(Path.cwd()) , "Data" , "final" , "df_all.csv") 

bool_notebook_version= True

if bool_notebook_version : output_notebook()  #Other wise export graph on file,

# General param for the graphs

jitter_width=0.1  # used to spread the points  for better visibily on  most of the graph
circle_size=12 
circle_alpha=0.5

plot_height_val=800
plot_width_val =700

# %matplotlib inline    #Not needed

#%% import data and make a copy 
dfi = pd.read_csv(df_csv_file_all,  na_values=np.nan)
dfi = dfi.copy()

#%% Adjust/clean data
def lower_case(data):
	return str(data).lower()

# all in lower case
dfi['manufacturer'] = dfi['manufacturer'].apply(lower_case)

dict_name_manufacturer = {"uturn": "u-turn"}
dfi.replace({"manufacturer": dict_name_manufacturer})

#define type of columns
dfi.certification = dfi.certification.astype(str)

cols = ['flat_AR', 'ptv_mini','ptv_maxi','flat_area','proj_area','year','weight' ]
dfi[cols] = dfi[cols].apply(pd.to_numeric, errors='coerce', axis=1)

#%% Input data frame ready
dfi.info()
print("Dataframe is ready")

#%% Global stats :
print("nb gliders from source: ", dfi.groupby('source').name.count())
print("")
print("nb of manufacturers : ", len(dfi.manufacturer.unique()))
print("")
#print("nb gliders from manufactuer: ", dfi.groupby('manufacturer').name.count())
print("nb gliders from manufacturer (top 20): ", dfi.groupby('manufacturer').name.count().sort_values(ascending=False).head(20))

#%% Graph number glider / year 
dfi.year.hist()

#%% Prepatre a lighter DataFrame for Aspect ratio:
df_Ar = dfi[['year', 'manufacturer','name','size','flat_AR','certification']]

#%%  PLOT ASPECT RATION ( YEAR )
# Group by name 
dfG_AR_Name= df_Ar.groupby("name")
nb_model = len(dfG_AR_Name.size())

dfG_AR_Name_mean = dfG_AR_Name.mean()
dfGyear = df_Ar.groupby("year")

#Median and quartile

qv1=0.1 
qv2=0.5
qv3=0.9

q1 = dfGyear.quantile(q=qv1)
q2 = dfGyear.quantile(q=qv2)
q3 = dfGyear.quantile(q=qv3)

date_start= str(int(q1.index[0]))
date_end=str(int(q1.index[-1]))

#Plot
p = figure(title = 
"Flat Aspect Ratio (year)  nb_model:" + str(nb_model) + '   ['+date_start+' - '+date_end+']' )
p.xaxis.axis_label = 'Year'
p.yaxis.axis_label = 'Flat Aspect ratio'

p.circle(dfG_AR_Name_mean["year"], dfG_AR_Name_mean["flat_AR"]
					, fill_alpha=0.2, size=5)

p.line(q1.index, q1["flat_AR"],legend=str(qv1), color='navy',alpha=0.8)
p.line(q2.index, q2["flat_AR"],legend='Median', color='navy' , line_width=4)
p.line(q3.index, q3["flat_AR"],legend=str(qv3), color='navy',alpha=0.8)

show(p)

#%% SIZE Selection 
# created a df with only one size per model
# Sort pick up for target PTV,
Target_weight=100   # kg

#Group by name
dfGname=dfi.groupby('name')

def extract_one_size(data , trag_weight):
	# print(data)
	# print(type(data))
	return data[data['ptv_maxi'] > trag_weight].min()


dfG_1size = dfGname.apply(extract_one_size , trag_weight=Target_weight)
df1 = dfG_1size.unstack() 
df1.drop(columns=['name'],inplace=True)
df1.certification.fillna(value=pd.np.nan, inplace=True)

cols = ['flat_AR', 'ptv_mini','ptv_maxi','flat_area','proj_area','year','weight','flat_span','proj_AR' ]
df1[cols] = df1[cols].apply(pd.to_numeric, errors='coerce', axis=1)


#%% Prepare for graph

unique_certif_order=['AFNOR_Biplace','AFNOR_Standard', 'AFNOR_Perf',  'AFNOR_Compet', 
	'DHV_1', 'DHV_2', 'DHV_3', 
'A', 'B', 'C', 'D','CCC',
'DUVL', 'DGAC',
 'Load', 'nan', 'pending']

#Common Graph tools
TOOLS="crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"


#%%    ## ASPECT RATIO / EN cat // Manufactuer
start_year = 2000
df1_recent  = df1.loc[(df1['year'] > start_year)]
print("Cuuting year : ", start_year)
print("nb of glider : ",len(df1_recent.index.unique()))

#Add color
df1_recent['manufacturer_colors']= "#0000FF"
# Change color for Ozone
df1_recent.loc[df1_recent.manufacturer == 'ozone', 'manufacturer_colors'] = "#ff0000"

source = ColumnDataSource(df1_recent)
#source = df1

if 1: 

	TITLE = "Glider Aspect Ratio by category  colored by manufacturer  V1"
	print("\n" + TITLE)

	p = figure(plot_height=plot_height_val,plot_width=plot_width_val,tools=TOOLS, toolbar_location="above",x_range=unique_certif_order, title=TITLE)
	p.toolbar.logo = "grey"
	p.background_fill_color = "#dddddd"
	p.xaxis.axis_label = "category"
	p.yaxis.axis_label = "AR"
	p.grid.grid_line_color = "white"

	p.title.text_font_size = "25px"

	hover = HoverTool()

	hover.tooltips = [
		("name", "@name"),
		("Manufacturer", "@manufacturer"),
		("Size", "@size"),
		("weight_range", "@ptv_mini - @ptv_maxi"),            
		("AR:", "@flat_AR"),
		("year", "@year"),
		("_______", "")
	]

	p.tools.append(hover)
	# jitter is used to  spread a little bit 

	p.circle(jitter('certification', width=jitter_width,distribution='normal', range=p.x_range), "flat_AR", size=circle_size, source=source,
			color='manufacturer_colors', line_color="black")

	#output_file(chart_1_AR, title="AR_cat_manufacturer")

	show(p)  # open a browser

#%% ## ASPECT RATION / EN CAT  Violin
# it is a try to have a better view on the distribution WIP
list_cert=['A','B','C','D','CCC']
min_year=2010
df_filt=df1.loc[(df1['year'] > min_year) & (df1['certification'].isin(list_cert) ) ]

width = 12
height = 12
plt.figure(figsize=(width, height))

#ax= sns.violinplot(data)

#ax = sns.violinplot(x="day", y="total_bill", data=tips)
ax = sns.violinplot(x=df_filt["certification"], y=df_filt["flat_AR"])



#%%    ## ASPECT RATIO / Year // Categorie
#Apply colors by categorie
palette =  Category20[20]
dict_certif_color={name:palette[idx] for idx,name in enumerate(unique_certif_order)}
list_temp=[]
for certif in df1['certification'] :
		if certif == 'nan':
			#print('Nan')
			list_temp.append('#000000')
		else:
			list_temp.append(dict_certif_color[str(certif)])

#list_temp=[dict_certif_color[certif] for certif in df1['certification']  ]
df1['certification_colors']= list_temp
source = ColumnDataSource(df1)

TITLE = "Glider Aspect Ratio by year  colored by Categorie  V1"
print("\n" + TITLE)

p = figure(plot_height=plot_height_val,plot_width=plot_width_val,tools=TOOLS, toolbar_location="above", title=TITLE)
p.toolbar.logo = "grey"
p.background_fill_color = "#dddddd"
p.xaxis.axis_label = "Year"
p.yaxis.axis_label = "AR"
p.grid.grid_line_color = "white"

p.title.text_font_size = "25px"


hover = HoverTool()

hover.tooltips = [
	("name", "@name"),
	("Manufacturer", "@manufacturer"),
	("Size", "@size"),
	("weight_range", "@ptv_mini - @ptv_maxi"),            
	("AR:", "@flat_AR"),
	("cert:", "@certification"),
	("year", "@year"),
	("_______", "")
]

p.tools.append(hover)

p.circle("year", "flat_AR", size=circle_size, source=source,
		color='certification_colors', line_color="black" , legend='certification')

p.legend.location = "top_left"
p.legend.click_policy="hide"
#output_file(chart_1_AR, title="AR_cat_manufacturer")

show(p)  # open a browser


#%% average Aspect ration / year ( colored by Cert)

df1_cert_long = df1.groupby(['year','certification'])['flat_AR'].median()
df1_cert = df1_cert_long.unstack()

df1_cert.drop(columns=['nan','Load','pending','DUVL','DGAC' ],inplace=True)

numlines_nb=len(df1_cert.columns)
mypalette=Category20[numlines_nb]

#verions multilines
# p = figure(width=plot_width_val, height=plot_height_val) 
# r=p.multi_line(xs=[df1_cert.index.values]*numlines_nb,
#                 ys=[df1_cert[name].values for name in df1_cert],
#                 line_color=mypalette,
#                 line_width=2
# 				)
# #legend = Legend(items=list_temp2)
# legend = Legend(items=
# [LegendItem(label=col, renderers=[r], index=idx) for idx, col in enumerate(list(df1_cert.columns))]
# )
# p.add_layout(legend)
# show(p)

p = figure(plot_width=plot_width_val, plot_height=plot_height_val ,tools=TOOLS)
p.title.text = 'Median value of Aspect ration  ( year ) , colored by certif'

for name, color in zip(df1_cert.columns, mypalette):
	df = pd.DataFrame(df1_cert[name])
	p.line(df.index,df[name], line_width=2, color=color, alpha=0.8, legend=name)

# change just some things about the grid
p.ygrid.minor_grid_line_color = 'navy'
p.ygrid.minor_grid_line_alpha = 0.2

p.xgrid.minor_grid_line_color = 'blue'
p.xgrid.minor_grid_line_alpha = 0.1

#Legend
p.legend.location = "top_left"
p.legend.click_policy="hide"

#output_file("interactive_legend.html", title="interactive_legend.py example")

show(p)


#%% Means Aspect ration / year ( colored by Manufacturer)
# a try to see the average position of the main manufacturer  on each categorie
# WIP : not clear and biais on manufacturer selection. TODO: to find a better solution to select the main manufacturers.

year_cut=2000
df1_cut_year=df1.loc[df1['year']>year_cut]


df1_cut_yr_avg= df1_cut_year.groupby('manufacturer')

df1_cut_yr_manu = df1_cut_year.groupby(['manufacturer','year','certif_EN'])
df1_cut_yr_manu_mean = df1_cut_yr_manu['flat_AR'].mean()
df1_cut_yr_manu_mean_df = df1_cut_yr_manu_mean.unstack()


df1_cut_yr = df1_cut_year.groupby(['year','certif_EN'])['flat_AR'].mean()
df1_cut_yr.drop(index=['nan','DUVL','pending','DGAC'], level=1,inplace=True)
df1_cut_yr_df = df1_cut_yr.unstack()

df1_minus = df1_cut_yr_manu_mean_df -  df1_cut_yr_df

df1_minus_avg =df1_minus.groupby('manufacturer').mean()

df1_minus_avg_T= df1_minus_avg.transpose()
df1_minus_avg_T.dropna(axis='columns' ,inplace=True)


numlines_nb=len(df1_minus_avg_T.columns)
mypalette=cividis(numlines_nb)


p = figure(plot_width=plot_width_val, plot_height=plot_height_val , x_range=list(df1_minus_avg_T.index),tools=TOOLS)
p.title.text = 'Means value of Aspect ration  ( year ) , colored by manufactuer'

for name, color in zip(df1_minus_avg_T.columns, mypalette):
	df = pd.DataFrame(df1_minus_avg_T[name])
	p.line(df.index,df[name], line_width=2, color=color, alpha=0.8, legend=name)

# change just some things about the grid
p.ygrid.minor_grid_line_color = 'navy'
p.ygrid.minor_grid_line_alpha = 0.2

p.xgrid.minor_grid_line_color = 'blue'
p.xgrid.minor_grid_line_alpha = 0.1

#Legend
p.legend.location = "top_left"
p.legend.click_policy="hide"

#output_file("interactive_legend.html", title="interactive_legend.py example")

show(p)



#%%
print("-- End --")



########################################################################
#  KEEP FOR ARCHIVE

# #%% average Aspect ration / year ( colored by Cert)
# year_cut=2000
# df1_cut_year=df1.loc[df1['year']>year_cut]

# nb_bigger_brand=1
# df1_nb_glider = df1_cut_year.groupby(['year','manufacturer'])['flat_AR'].count()
# df1_nb_glider_top = df1_nb_glider.groupby('year').nlargest(nb_bigger_brand)
# df1_nb_glider_top = df1_nb_glider_top.droplevel(level=0)
# df1_nb_glider_top = df1_nb_glider_top.to_frame()
# list_top_manufactuer = list(df1_nb_glider_top.index.get_level_values(1).unique())
# print("Nb_manufactuer:",len(list_top_manufactuer))

# df1_cut = df1_cut_year.loc[df1_cut_year['manufacturer'].isin(list_top_manufactuer)]




# df1_manufacturer_long = df1_cut.groupby(['year','manufacturer'])['flat_AR'].median()
# df1_manufacturer = df1_manufacturer_long.unstack()

# numlines_nb=len(df1_manufacturer.columns)
# mypalette=cividis(numlines_nb)


# p = figure(plot_width=plot_width_val, plot_height=plot_height_val ,tools=TOOLS)
# p.title.text = 'Median value of Aspect ration  ( year ) , colored by manufactuer'

# for name, color in zip(df1_manufacturer.columns, mypalette):
# 	df = pd.DataFrame(df1_manufacturer[name])
# 	p.line(df.index,df[name], line_width=2, color=color, alpha=0.8, legend=name)

# # change just some things about the grid
# p.ygrid.minor_grid_line_color = 'navy'
# p.ygrid.minor_grid_line_alpha = 0.2

# p.xgrid.minor_grid_line_color = 'blue'
# p.xgrid.minor_grid_line_alpha = 0.1

# #Legend
# p.legend.location = "top_left"
# p.legend.click_policy="hide"

# #output_file("interactive_legend.html", title="interactive_legend.py example")


# show(p)-/