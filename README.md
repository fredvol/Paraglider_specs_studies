
# Paraglider_specs_studies

## Database of a lot of pargliders and some scripts to analyse the datas

## Goals:

The goal was to agregate a maximun of glider specs.
Based on a archive version of Para2000.org and gliderbase.com info on Fev2019.
Datas has been collected, merged, removed duplicate and clean .

This data colection is used to proceed to multiple studies and analyses.
First one is : Aspect ratio evolution.

THIS IS A WORK IN PROGRESS ( sometime! )

## Notes:

* Some error might be present in the data.

## Files details :

 Source data in: Data/final/df_all.csv

#### Details 

* index	
* certif_AFNOR:  Certification according to AFNOR Standart
* certif_DHV:  Certification according to AFNOR Standart
* certif_EN	: Certification according to AFNOR Standart
* certif_MISC: Certification according to other standart ( CCC, DULV, DGAC)
* certification: Most recent of the certification : EN > DHV > AFNOR
* flat_AR Flat aspect ratio
* flat_area  : [m2]
* flat_span : [m]
* manufacturer : Manufacturer name
* name : Glider name
* proj_AR projected aspect ratio
* proj_area : [m2]
* proj_span : [m]]
* ptv_maxi : [kg]
* ptv_mini: [kg]
* size
* source
* weight: [kg]
* year

### Analyse Files details:

#### 3_AR_study_NB.py

Used to study the aspect ratio tendancy.

## Dependancy:

> * matplotlib==3.0.2
> * pandas==0.23.4
> * numpy==1.15.4
> * bokeh==1.2.0
> * seaborn==0.9.0   (use only for the violin graph)

See requirement.txt , to install :

`pip3 install -r requirements.txt`

## Instalation
Developed on linux with Anaconda Distributiuon  : **Python 3.7**

https://www.anaconda.com/distribution/


## License
No specs!

Thanks to **Para2000** and **gliderbase.com** for they amazing work of collecting data days after days, month after months and yers afters years.
