import pandas as pd
import io
import requests
import numpy as np
import urllib.request, urllib.parse, urllib.error
import urllib.request
import json
#from urllib.request import urlopen
#from pandas import DataFrame
import matplotlib.pyplot as plt
##########################################################################
################Cancer cleaning###########################################
##########################################################################

### Cancer Data with all sexes, races, ages and cancer types
url = 'https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1'
s = requests.get(url).content
cancer_data = pd.read_csv(io.StringIO(s.decode('windows-1252')), skiprows=8, skipfooter=27, engine='python')   
cancer_data['Category'] = 'All'
cancer_data['Group'] = 'Total'



def load_subdata(url, category, groupname, cancer_data):
    s = requests.get(url).content
    sub_data = pd.read_csv(io.StringIO(s.decode('windows-1252')), skiprows=8, skipfooter=27, engine='python', error_bad_lines=False)   
    sub_data['Category'] = category
    sub_data['Group'] = groupname
    return(cancer_data.append(sub_data, ignore_index = True))

cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=047&race=00&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Cancer Type', 'Lung Cancer', cancer_data)

cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=06&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Race/Ethnicity', 'White Hispanic', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=07&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Race/Ethnicity', 'White Non-Hispanic', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=02&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Race/Ethnicity', 'Black (includes Hispanic)', cancer_data)
# cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=03&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
#            'Race/Ethnicity', 'Amer. Indian/Alaskan Native (includes Hispanic)', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=04&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Race/Ethnicity', 'Asian or Pacific Islander (includes Hispanic)', cancer_data)
# cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=05&sex=0&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
#            'Race/Ethnicity', 'Hispanic (any race)', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=1&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Sex', 'Males', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=2&age=001&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Sex', 'Females', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=009&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Age', '<50', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=136&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Age', '50+', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=006&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Age', '<65', cancer_data)
cancer_data = load_subdata('https://www.statecancerprofiles.cancer.gov/incidencerates/index.php?stateFIPS=99&cancer=001&race=00&sex=0&age=157&type=incd&sortVariableName=rate&sortOrder=desc&output=1',
             'Age', '65+', cancer_data)

def cDataChecker(cancer_data):
    # run this before and after below clean data codes to see whether there 
    # are errors in the data
    cancer_data.info()
    cancer_data['Recent Trend'].value_counts()
    try:
        cancer_data['FIPS'].value_counts()
    except:
        print('\nRename the FIPS column.  The column name is messy')
    cancer_data['Age-Adjusted Incidence Rate(†) - cases per 100,000'].value_counts()
    cancer_data['Lower 95% Confidence Interval'].value_counts()
    cancer_data['Upper 95% Confidence Interval'].value_counts()
    cancer_data['Average Annual Count'].value_counts()
    cancer_data['Recent 5-Year Trend (‡) in Incidence Rates'].value_counts()
    cancer_data['Lower 95% Confidence Interval.1'].value_counts()
    cancer_data['Upper 95% Confidence Interval.1'].value_counts()


######################### clean the data#####################################
#############################################################################
# delete Met Healthy People
cDataChecker(cancer_data)
cancer_data = cancer_data.drop("Met Healthy People Objective of ***?", axis=1)
# change * to nan
cancer_data = cancer_data.replace('*', np.nan)
cancer_data = cancer_data.replace('* ', np.nan)
cancer_data = cancer_data.replace(' *', np.nan)
cancer_data = cancer_data.replace('**', np.nan)
cancer_data = cancer_data.replace(' **', np.nan)
cancer_data = cancer_data.replace('** ', np.nan)
cancer_data = cancer_data.replace('¶',np.nan)
cancer_data = cancer_data.replace('¶¶',np.nan)
cancer_data = cancer_data.replace('¶¶ ',np.nan)
cancer_data = cancer_data.replace('¶ ',np.nan)
cancer_data = cancer_data.replace(' ¶',np.nan)
cancer_data = cancer_data.replace(' ¶¶',np.nan)
# change §§§  to nan
cancer_data = cancer_data.replace('§§§',np.nan)
cancer_data = cancer_data.replace(' §§§',np.nan)
cancer_data = cancer_data.replace('§§§ ',np.nan)
cancer_data = cancer_data.replace('§§',np.nan)
cancer_data = cancer_data.replace('§§ ',np.nan)
cancer_data = cancer_data.replace(' §§',np.nan)
cancer_data = cancer_data.replace('&sect;&sect;&sect;',np.nan)
# drop FIPS=0
cancer_data = cancer_data[cancer_data[' FIPS'] != 0.0]
# delete the space before FIPS variable name
cancer_data.rename(columns={' FIPS': 'FIPS'}, inplace=True)
# replace fewer than 3 Average Annual Count into 3
cancer_data = cancer_data.replace('3 or fewer','3')

# change Recent 5-Year Trend (‡) in Incidence Rates &  Average Annual Count into numeric variable
cancer_data['Recent 5-Year Trend (‡) in Incidence Rates'] = cancer_data[
        'Recent 5-Year Trend (‡) in Incidence Rates'].astype('float')
cancer_data['Average Annual Count'] = cancer_data['Average Annual Count'].astype('float')
#split county, state, SEER, NPCR into separate columns
cancer_data['County'], cancer_data['State'] = cancer_data['County'].str.split(', ', 1).str
cancer_data['State'], cancer_data['SEER'] = cancer_data['State'].str.split('(', 1).str
cancer_data['SEER'], cancer_data['NPCR'] = cancer_data['SEER'].str.split(',', 1).str
cancer_data['NPCR'] = cancer_data['NPCR'].str.replace(")","")
# delete '#' 
cancer_data['Age-Adjusted Incidence Rate(†) - cases per 100,000'] = cancer_data[
        'Age-Adjusted Incidence Rate(†) - cases per 100,000'].str.replace('#',"")
# add leading zeros to FIPS < 5 digits
cancer_data['FIPS'] = cancer_data['FIPS'].astype(str).replace("000nan", "")
cancer_data['FIPS'], na = cancer_data['FIPS'].str.split('.', 1).str
del na
cancer_data['FIPS'] = cancer_data['FIPS'].apply(lambda x: '{0:0>5}'.format(x))

#Writing to csv file
cancer_data.to_csv('cleaned_cancer.csv', sep=',', encoding='utf-8', index = False)
cDataChecker(cancer_data)
###############################################################################
############## detecting cancer outliers#######################################
###############################################################################
cancer_data.hist(column='Average Annual Count',bins=5,range=[0, 4000])
extreme_cancer = cancer_data.loc[cancer_data['Average Annual Count'] >2500]

###############################################################################
######## water cleaning #######################################################
###############################################################################
url = "https://ephtracking.cdc.gov:443/apigateway/api/v1/getCoreHolder/441/2/ALL/ALL/2011,2012,2013,2013,2014,2015/0/0"

# Reading the json as a dict
with urllib.request.urlopen(url) as json_data:
    data = json.load(json_data)

# load from_dict
data = pd.DataFrame.from_dict(data['pmTableResultWithCWS'])  ### Datatype of dataValue is object, change it into numericals
data['dataValue'] = pd.to_numeric(data['dataValue'], errors='coerce')
### Checking missing values/typos/outliers in datasets  
### Datatype of dataValue is object, change it into numericals
### 1. get the means of all values for each county using "groupby" method
result = data.groupby(['title','year','geoId'])['dataValue'].mean()
### change "series" into "dataframe" datatype
        
### change "series" into "dataframe" datatype    
result = result.to_frame().reset_index()
result.columns = ['Location','Year','GeoId','Value']
    
### 2. split the location column into two column("county","state"), 
# expand=True to add these two column into dataframe
result[['County','State']] = result['Location'].str.split(',', n=1, expand=True)
### remove the location column
del result['Location']
#print (result.describe())
### the max value is 31.837500, which is lower than 50


def binQuality(result):
    ### 3. binning the mean into three categories
    ## According to documents in its original website:
    ### level < 1 means non-dect arsenic
    ### level in (1-10) means less than MCL == "no harm"
    ### level in (10-50) means "harmful"
    bins = [-1,1,10,60]
    labels=['Non Detect','Less than or equal MCL','More than MCL' ]   
    result['Quality']=pd.cut(result['Value'],bins,labels=labels)
    result.columns = ['Year','FIPS','Value','County','State','Quality']
    return result

### Code for Outliers check (Water Quality) ###
def waterOutlier(waterResult):
    ### check the outliers
    # For numerical column
    plt.figure(" Water Quality Outliers Check")
    plt.subplot(121)
    waterResult.boxplot(column="Value")
    plt.title("Boxplot for MCL Value data check")
    

    # For non-numerical columns, using histogram to check if there
    # Due to MCL-VALUE is a range for arsenic level, there is no higher limitation for this value
    # So we turn to check if there is any outliers(empty) in binned columns. 
    # If there is, it means the original dataset has outliers
    plt.subplot(122)
    waterResult['Quality'].isnull().sum()
    waterResult['Quality'].value_counts().plot(kind="bar")
    plt.title("Histogram for binned MCL Value data")
    plt.show()

    # if there is Nan (outliers for string type column), drop it
    waterResult['Quality'].dropna()

result = binQuality(result)
waterOutlier(result)
result.to_csv("water_clean.csv",index=False)

###############################################################################
##Air Data Cleaning Code######################################################
################################################################################
##This code inputs air quality data for each US County for 2011-2015
##This code cleans and adds the FIPS county and state codes to the dataframe
##This code adds new columns that normalize the “Good”, “Unhealthy” etc days 
#by dividing them by the number of days AQI was recorded for that county

##THis code checks for outliers in the numerical data
##This code removes outliers in the Max AQI column that are over 500.
data2015 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2015.zip')
data2014 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2014.zip')
data2013 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2013.zip')
data2012 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2012.zip')
data2011 = pd.read_csv('https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_2011.zip')
countyReference = pd.read_excel("https://www.schooldata.com/pdfs/US_FIPS_Codes.xls")

##fixing the header for CountyReference
new_header = countyReference.iloc[0]
countyReference = countyReference[1:]
countyReference.columns = new_header

##creating a single dataframe for air quality data and exporting to a csv file
allPollutionData = pd.concat([data2015, data2014, data2013, data2012, data2011])
allPollutionData = allPollutionData.reset_index(drop=True)
allPollutionData.to_csv('All_Pollution_Data.csv')


##There is no null rows
nullCount = allPollutionData.isnull().values.ravel().sum()
print("\nThe total number of rows with null values is: \n")
print(nullCount)


dataOnlyStateCounty = allPollutionData.loc[:,['State','County']]
referenceStateCounty = countyReference.loc[:,['State','County Name']]
##groupby('State') is grouping the dataframe by the unique values in the State column.
##The values in Country column are mapped to each unique value from State column.
##Tthen index for County column and turn the values that are mapped to the State column into 
##each distinct list groupings.
##Then turn each unique grouping to a dict
a = dataOnlyStateCounty.groupby('State')['County'].apply(list).to_dict()
b = referenceStateCounty.groupby('State')['County Name'].apply(list).to_dict()


columns = ['Year','Days with AQI','Good Days','Moderate Days','Max AQI','90th Percentile AQI',
           'Median AQI','Days CO','Days NO2','Days Ozone',
           'Days SO2','Days PM2.5','Days PM10']


##break statement is used to exit out of the nearest for loop
##The purpose of this loop is to determine invalid States and Counties
##The output has repitition because there are multiple state,county entries within some keys.
##These aren't duplicates because the same state,county is used across different years.
def stateCountyChecker(a,b):
    countCounty = 0
    countState = 0
    
    for key in a:
          for value in a[key]:
                try:
                      if value not in b[key]:
                            print(("The following state|county is a messy data entry: ", key,'|', value))
                            countCounty += 1
                except KeyError:
                      print((key,"is not a valid State or the entry is messy"))
                      countState += 1
                      break
                  
    print(('The number of messy States is',countState))
    print(('The number of messy Counties is',countCounty))


##Leap years have 366 days.  2012 and 2016 are leap years
##The purpose of this loop is to run through each column and determine the extent of messy data
def numericColumnChecker(allPollutionData,columns):
    for i in columns:
          c = allPollutionData[i].value_counts().sort_index()
          if i == 'Year':
                print("Expecting years to be between 2011 and 2017")
                indexLength = len(c)-1
                
                if c.index[0] < 2011 or c.index[indexLength] > 2017:
                      print("There are invalid entries in years column")
                      invalidEntries = c[(c.index < 2011) | (c.index > 2017)].sum()
                      print(("The number of invalid entries is", invalidEntries))
                else:
                      print(("No invalid entries for",i))
                      
          elif (i == 'Days with AQI' or i == 'Good Days' or i == 'Moderate Days' or i == 'Days CO'
                or i == 'Days NO2' or i == 'Days Ozone' or i == 'Days SO2' or i == 'Days PM2.5'
                or i == 'Days PM10'):
                print(("Expecting entries for",i,"column to between 0 and 366"))
                indexLength = len(c)-1
                
                if c.index[0] < 0 or c.index[indexLength] > 366:
                      print(("There are invalid intries in",i))
                      invalidEntries = c[(c.index < 0) | (c.index > 366)].sum()
                      print(("The number of invalid entries is", invalidEntries))
                else:
                      print(("No invalid entries for",i))
                      
          elif (i == 'Max AQI' or i == '90th Percentile AQI' or i == 'Median AQI'):
                print(("Expecting entries for",i,"column to be greater than 0"))
    
                if c.index[0] < 0:
                      print(("There are invalid intries in",i))
                      invalidEntries = c[c.index < 0].sum()
                      print(("The number of invalid entries is", invalidEntries))
                else:
                      print(("No invalid entries for",i))

# Determining the amount of dirt in data.
stateCountyChecker(a,b)
numericColumnChecker(allPollutionData,columns)

def hasPM25(allPollutionData):
    '''Function creates a bin for Days PM2.5 variable in the allPollutionData.
    0 means there are 0 days with PM2.5 and 1 means there are days with PM2.5.'''
    binLabels = [0,1]
    binRange = [0,1,367]
    allPollutionData['hasPM2.5'] = pd.cut(allPollutionData['Days PM2.5'],bins = binRange, 
                    right = False, labels = binLabels)
    return allPollutionData

def cleanPollutionData(allPollutionData):
    '''The purpose of this function is to clean the airPollutionData data frame
    by changing strings in the State and Country attributes, so it matches the 
    reference.  Function returns a cleaned dataframe, and is specific to the 
    allPollutionData data frame.'''
    allPollutionData[['State', 'County']] = allPollutionData[['State','County']].apply(
            lambda x: x.str.replace(".","").str.strip())
    
    toChange = ["DeKalb", "DuPage", "Saint Clair", "McLean", "LaPorte", "St John the Baptist",
                "Baltimore (City)", "Prince George's", "Saint Louis", "DeSoto", "Saint Charles",
                "Sainte Genevieve", "McKinley", "McKenzie", "McClain", "Fond du Lac",
                "Matanuska-Susitna","Yukon-Koyukuk"]
    newValue = ["De Kalb", "Du Page", "St Clair", "Mclean", "La Porte", "St John The Baptist"
                , "Baltimore", "Prince Georges", "St Louis", "De Soto", "St Charles"
                , "Ste Genevieve", "Mckinley", "Mckenzie", "Mcclain", "Fond Du Lac"
                , "Matanuska Susitna", "Yukon Koyukuk"]
    allPollutionData['County'] = allPollutionData['County'].replace(toChange, newValue)
    allPollutionData['State'] = allPollutionData['State'].replace('District Of Columbia','District of Columbia')
    
    filt = allPollutionData['State'].isin(['Tennessee', 'Virginia'])
    allPollutionData.loc[filt, ['County']] = allPollutionData.loc[filt, [
            'County']].replace('De Kalb', 'Dekalb').replace('Charles', 'Charles City')
    
    filt = (allPollutionData['State'] != 'Country Of Mexico') & (allPollutionData[
            'State'] != 'Puerto Rico') & (allPollutionData['State'] != 'Virgin Islands')
    allPollutionData = allPollutionData[filt]
    return allPollutionData

def addFips(cleanedData, countyReference):
    '''This function takes about 30 minutes to run.  It ends when the counter
    hits 5259.  Function's purpose is to add FIPS State and FIPS County to 
    cleanedData by matching county and state to countyReference.'''
    temp = countyReference['FIPS State'].unique()
    temp2 = countyReference['State'].unique()
    temp3 = pd.DataFrame({'State':temp2, 'FIPS State':temp})
    cleanedData['FIPS State'] = cleanedData['State'].map(temp3.set_index("State")['FIPS State'])
    
    # Naive solution
    # Did not have time to find a better solution.
    i=0
    countyRS = countyReference[['State','County Name','FIPS County']]
    cleanDS = cleanedData[['State','County']]
    for index, row in countyRS.iterrows():
        for index_d, row_d in cleanDS.iterrows():
            if row.State == row_d.State and row['County Name'] == row_d.County:
                cleanedData.loc[index_d, 'FIPS County'] = row['FIPS County']
                i+=1
                print(i)
    cleanedData['FIPS County'] = cleanedData['FIPS County']
    return cleanedData

def normalizeData(cdata):
    '''The data must be normalized by the number of days that the AQI values were taken'''
    cdata['Good Days_Norm'] = cdata['Good Days'] / cdata['Days with AQI']
    cdata['Moderate Days_Norm'] = cdata['Moderate Days'] / cdata[
            'Days with AQI']
    cdata['Unhealthy for Sensitive Groups Days_Norm'] = cdata[
            'Unhealthy for Sensitive Groups Days'] / cdata['Days with AQI']
    cdata['Unhealthy Days_Norm'] = cdata['Unhealthy Days'] / cdata[
            'Days with AQI']
    cdata['Very Unhealthy Days_Norm'] = cdata['Very Unhealthy Days'] / cdata[
            'Days with AQI']
    cdata['Hazardous Days_Norm'] = cdata['Hazardous Days'] / cdata[
            'Days with AQI']
    cdata['Days CO_Norm'] = cdata['Days CO'] / cdata['Days with AQI']
    cdata['Days NO2_Norm'] = cdata['Days NO2'] / cdata['Days with AQI']
    cdata['Days Ozone_Norm'] = cdata['Days Ozone'] / cdata['Days with AQI']
    cdata['Days SO2_Norm'] = cdata['Days SO2'] / cdata['Days with AQI']
    cdata['Days PM2.5_Norm'] = cdata['Days PM2.5'] / cdata['Days with AQI']
    cdata['Days PM10_Norm'] = cdata['Days PM10'] / cdata['Days with AQI']
    return cdata

def removeOutliers(cdata):
    '''Remove values over 500 for Max AQI days as these are obviously errors.'''
    '''The AQI scale ranges from 0-500'''
    clean_no_outlier = cdata[~(cdata['Max AQI'] >=500)] 
    return clean_no_outlier



cleanedData = cleanPollutionData(allPollutionData)
dataOnlyStateCounty = cleanedData.loc[:,['State','County']]
a = dataOnlyStateCounty.groupby('State')['County'].apply(list).to_dict()
stateCountyChecker(a,b)
numericColumnChecker(allPollutionData,columns)
cdata = addFips(cleanedData, countyReference)
cdata = normalizeData(cdata)
clean_no_outlier = removeOutliers(cdata)
#############Check for Outliers in numerical columns using a boxplot###################

cdata.boxplot(column='Good Days_Norm', by='Year')
plt.savefig('GoodDaysBP.jpg')
cdata.boxplot(column='Moderate Days_Norm', by='Year')
plt.savefig('ModDaysBP.jpg')
cdata.boxplot(column='Unhealthy for Sensitive Groups Days_Norm', by='Year')
plt.savefig('UFSGDaysBP.jpg')
cdata.boxplot(column='Unhealthy Days_Norm', by='Year')
plt.savefig('UnhealthyDaysBP.jpg')
cdata.boxplot(column='Very Unhealthy Days_Norm', by='Year')
plt.savefig('VUnhealthyDaysBP.jpg')
cdata.boxplot(column='Hazardous Days_Norm', by='Year')
plt.savefig('HazDaysBP.jpg')
cdata.boxplot(column = 'Max AQI', by = 'Year')
plt.savefig('MaxAQI.jpg')

##############################################################################
# NOTICE ME!!!!!!!!!
# fixing the FIPS entries, so excel will input it correctly.  
# This might cause problems in the analysis code if code is ran on apple computers.   
# The computer I am using is a Windows.
# Not 100% sure
##############################################################################
clean_no_outlier['FIPS State'] = clean_no_outlier['FIPS State'].apply('="{}"'.format)
clean_no_outlier['FIPS County'] = clean_no_outlier['FIPS County'].apply('="{}"'.format)
# Export clean data to csv file.
clean_no_outlier.to_csv('cleanPollutionData.csv', index=False)




