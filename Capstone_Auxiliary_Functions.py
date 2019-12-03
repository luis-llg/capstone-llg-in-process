''' Import of libraries'''

import glob
import csv
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
import traceback
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.dummy import DummyRegressor



def read_n_create_generation(csv_file):

    ''' Read a csv file and transform it in order to create a dataframe
    This DataFrame have the following columns:
    
    --- 'Fecha' : Date of the readings (one per file) string
    --- 'Fecha_completa': Date and hour in datetime
    --- 'Day':Day numeric
    --- 'Month': Month numeric
    --- 'Year':Year numeric
    --- 'Hour' : Hour of the reading (24 per file per 'Clave de nodo' or 'Region' 
    --- 'Zona de carga': Node code
    --- 'Cantidad total de energia asignada por Carga Directamente Modelada(MWh)' : Demand directly modelated
    --- 'Cantidad total de energia asignada por Carga Indirectamente Modelada(MWh)' : Demand indirectly modelated
    --- 'Cantidad total de energia asignada (MWh)' : Asignated demand

  
    (Intpus) : complete file address
    (output) : pandas Dataframe
    
    '''
    
    counter = 0
    features=[]
    data=[]
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            counter +=1
            if counter == 8:
                features=row
            if counter>8:
                data.append(row)
         
    df= pd.DataFrame(data, columns=features[:14])
    df['Fecha_string']=df[' Dia'].apply(lambda x: convert_string_to_date(x))
    df['Día']=df['Fecha_string'].apply(lambda x: x[0])
    df['Mes']=df['Fecha_string'].apply(lambda x: x[1])
    df['Año']=df['Fecha_string'].apply(lambda x: x[2])
    
    # Creation of feature "Fecha" as datetime
    df['Fecha']=df['Día']+'/'+df['Mes']+'/'+df['Año']
    df['Fecha']=df['Fecha'].apply(lambda x: datetime.strptime(join_list_date(convert_string_to_date(x)), "%d/%m/%Y"))
    df.columns=['Sistema','Dia','Hora','Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas','Fecha_string','Día','Mes','Año','Fecha']
    df.drop(['Sistema','Dia','Fecha_string'], axis=1, inplace=True)
    columns_order=['Fecha','Día','Mes','Año','Hora','Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas']
    df=df[columns_order]
    df[['Día','Mes','Año','Hora','Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas']]=df[['Día','Mes','Año','Hora','Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas']].apply(pd.to_numeric)

    return df

def create_or_load_dataframe_generation(Newfiles_list,Dpath_total,boolean_save__updated_csv=False):
    #Checking if there is a included csv files list
    fileorig=glob.glob(Dpath_total+'/*.csv')
    df_list=[]
    if check (fileorig):
        print("\n A file already exist, Loading data from that file")
        existingdf=reading_csv_Gen(Dpath_total)
        df_list.append(existingdf)
        print("Data loading completed")
    else:
        df_list=[]
        print('There is no previous data to load')
    list_of_errors=[]
    a=1  #Primer fichero que empieza en 1
    b=len(Newfiles_list)  #último fichero que termina en len()
    counter=a

    for file in Newfiles_list[a-1:b+1]:
        print('DataFrame nº {} of a total of {}:\n file: {}'.format(counter,len(Newfiles_list),Newfiles_list[counter-1]))
        counter+=1
        try:
            df_list.append(read_n_create_generation (file))
            print('Done\n')
        except Exception:
            df_list.append(pd.DataFrame(np.zeros(shape=(1,1))))
            list_of_errors.append(file)
            traceback.print_exc()
            print('Not pass\n')
            pass

    print('Range of files from {} to {} ok\n'.format(a,b))
    print('Wrong files:\n {}'.format(list_of_errors))
    print('Creating a unique dataframe')
    completedf=pd.concat(df_list, ignore_index=True)
    print('Saving dataframe to destination csv:{}'.format(Dpath_total+'/Saved_data.csv'))
    if b!=0:
        if boolean_save__updated_csv:
            completedf.to_csv(Dpath_total+'/Saved_data.csv')
            print ('Updated CSV file created\n Process complete :-)\n\n')
    else:
        print('A new file has not been created because we have not added new information to the existing one')
        
    return completedf

def merge_generation(list_of_generation_df):
    
    mergeddf=list_of_generation_df[0]
    expmergeddim=0
    for i in range(len(list_of_generation_df)-1):

        Li=mergeddf.shape[0]
        Liplusone=list_of_generation_df[i+1].shape[0]
        expmergeddim =Li+Liplusone

        if i == 3:
            mergeddf=pd.merge(left=mergeddf,right=list_of_generation_df[i+1], how='left', on=['Fecha','Día','Mes','Año','Hora'],indicator=True)
            for col in ['Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas']:
                mergeddf[col]=mergeddf[str(col+'_y')].fillna(mergeddf[str(col+'_x')])
            print('\nThis is and example to check if the correct value is taken\n',mergeddf[mergeddf['Fecha']=='2018-04-01'].iloc[0:4,[5,16,28]],'\nI modified manually one value of this dataframe for testing and corrected it to the original value afterwards')
            for col in ['Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas']:
                mergeddf.drop([str(col+'_y'),str(col+'_x')], axis=1,inplace=True)
            print('\nOriginal dataframe shape {},\nmerged with a dataframe of shape {}\nExpected merged shape should,in this case, have the shape of the original dataframe {} and is {}'.format(Li,Liplusone,Li,mergeddf.shape[0]))
        else:
            mergeddf=pd.merge(left=mergeddf,right=list_of_generation_df[i+1], how='outer', on=['Fecha','Día','Mes','Año','Hora','Eolica','Fotovoltaica','Biomasa','Carboelectrica','Ciclo Combinado','Combustion Interna','Geotermoelectrica','Hidroelectrica','Nucleoelectrica','Termica Convencional','Turbo Gas'],indicator=True)
            print('\nOriginal dataframe shape {},\nmerged with a dataframe of shape {}\nExpected merged shape should be {} and is {}'.format(Li,Liplusone,expmergeddim,mergeddf.shape[0]))

        r_only=mergeddf[mergeddf['_merge']=='right_only']['_merge'].count()
        l_only=mergeddf[mergeddf['_merge']=='left_only']['_merge'].count()
        both=mergeddf[mergeddf['_merge']=='both']['_merge'].count()

        if expmergeddim!=mergeddf.shape[0]:
            print('\nWe have added to the original dataframe of {}, {} new items from the merged dataframe'.format(Li,r_only))
            print('\n{} or ({} days x 24 hours) = New added values not included on the original dataframe \n{} = Items not updated from the original dataset\n{} = Equal items found in both datasets'.format(r_only,r_only/24,l_only,both))

        else:
            print('\nWe have added to the original dataframe of {}, {} new items from the merged dataframe'.format(l_only,r_only))

        print('\nmerged indicator column description\n')
        #print(mergeddf['_merge'])
        print(mergeddf['_merge'].describe())

        # Variable to know how much added data in daframe 4
        updated_values=mergeddf[mergeddf['Fecha'].isin(list_of_generation_df[4]['Fecha'].values)].shape[0]

        print('\n{} Items in common with the last version of dataframe'.format(updated_values))
        mergeddf=mergeddf.drop(['_merge'], axis=1)

        print('\n',mergeddf.sort_values(by=['Hora','Fecha'], ascending=True).head(), mergeddf.info(),'\n',mergeddf.describe() )

    return mergeddf


## Creation of new_csv_lists dataframes
def create_newfiles_list(Opath_total,Dpath_total):
    list_files=[]
    frame={}
    for i in range(len(Opath_total)):
        list_files.append(pd.Series(Opath_total[i], name=str(Dpath_total[i]).split('/')[-1:][0]))
        frame[str(Dpath_total[i]).split('/')[-1:][0]]=list_files[i]
    newfiles_list=pd.DataFrame(frame)
    return newfiles_list

def create_newfiles_list_n_save(listao,listad,dest_fich):
    lista_ficheros=create_newfiles_list(listao,listad)
    lista_ficheros.to_csv(dest_fich+'/data_included.csv')
    return lista_ficheros

### Creation of new files list 

def newfiles_list(file_list_total,path_destino_total,path_destino_ficheros):
    #Loading existing files list
    existing_files=create_newfiles_list(file_list_total,path_destino_total)
    #Loading the file that we already had (if exists)
    try:
        print('Reading files of csv already included in data')
        included_files = pd.read_csv(path_destino_ficheros+'/data_included.csv')
        included_files.drop(included_files.columns[0], axis=1, inplace=True)
        print('Done\n')
    except Exception:
        print('There is not a previous file created\n')
        list_files=[]
        frame={}
        for i in range(len(path_destino_total)):
            list_files.append(str(path_destino_total[i]).split('/')[-1:][0])
        included_files = pd.DataFrame(columns=list_files)
        traceback.print_exc()
        pass

    #Here we obtain the file including only the new elements
    listnewfiles=[]
    for i in included_files.columns:
        for val in existing_files[i].values:
            if val not in included_files[i].values:
                listnewfiles.append(val)
    listnewfiles = [elem for elem in listnewfiles if str(elem) != 'nan']
    
    # List is separated in several files regarding its destination folder
    a=[]
    for column in existing_files.columns:
        a.append(list(set(existing_files[column]).intersection(set(listnewfiles))))
    print('List of files created')
    return a

## Create or load dataframe 

def create_or_load_dataframe(Newfiles_list,Dpath_total,boolean_save__updated_csv=False):
    '''
    1 This function (function 1)checks if there is a previously created list of files
    2 if there is a list function 1 reads it and commence loading previously created csv
    3 if there is not a previously created list, function 1 the function creates a dataframe.
        3.1 function 1 calls a specific function to read csv's and defines final dataframe structure based on the kind of csv loaded (PML, PND or demand)
    4 function 1 concatenate all csv files included on the original path (same kind of csv's) '''
    
    #Checking if there is a included csv files list
    fileorig=glob.glob(Dpath_total+'/*.csv')
    df_list=[]
    if check (fileorig):
        print("\n A file already exist, Loading data from that file")
        existingdf=reading_csv(Dpath_total)
        df_list.append(existingdf)
        print("Data loading completed")
    else:
        df_list=[]
        print('There is no previous data to load')
    list_of_errors=[]
    a=1  #Primer fichero que empieza en 1
    b=len(Newfiles_list)  #último fichero que termina en len()
    counter=a

    for file in Newfiles_list[a-1:b+1]:
        print('DataFrame nº {} of a total of {}:\n file: {}'.format(counter,len(Newfiles_list),Newfiles_list[counter-1]))
        counter+=1
        try:
            df_list.append(read_n_create_files (file))
            print('Done\n')
        except Exception:
            df_list.append(pd.DataFrame(np.zeros(shape=(1,1))))
            list_of_errors.append(file)
            traceback.print_exc()
            print('Not pass\n')
            pass

    print('Range of files from {} to {} ok\n'.format(a,b))
    print('Wrong files:\n {}'.format(list_of_errors))
    print('Creating a unique dataframe')
    completedf=pd.concat(df_list, ignore_index=True)
    print('Saving dataframe to destination csv:{}'.format(Dpath_total+'/Saved_data.csv'))
    if b!=0:
        if boolean_save__updated_csv:
            completedf.to_csv(Dpath_total+'/Saved_data.csv')
            print ('Updated CSV file created\n Process complete :-)\n\n')
    else:
        print('A new file has not been created because we have not added new information to the existing one')
        
    return completedf

def read_commodities(commodities_csv_file):
    counter = 0
    features=[]
    data=[]
    
    with open(commodities_csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            counter +=1
            if counter == 1:
                features=row
            if counter>1:
                data.append(row)
         
    df= pd.DataFrame(data, columns=features[:6])
    df['Fecha_string']=df['Date'].apply(lambda x: convert_string_to_date(x))
    df['Día']=df['Fecha_string'].apply(lambda x: x[0])
    df['Mes']=df['Fecha_string'].apply(lambda x: x[1])
    df['Año']=df['Fecha_string'].apply(lambda x: x[2])
   
    # Creation of feature "Fecha" as datetime
    df['Fecha']=df['Día']+'/'+df['Mes']+'/'+df['Año']
    df['Fecha']=df['Fecha'].apply(lambda x: datetime.strptime(join_list_date(convert_string_to_date(x)), "%d/%m/%Y"))
    
    columns_order=['Fecha','Día','Mes','Año','TdC','API2','Brent','HH']
        
    df[['API2','Brent','HH','TdC','Día','Mes','Año']]=df[['API2','Brent','HH','TdC','Día','Mes','Año']].apply(pd.to_numeric)
    df=df[columns_order]
    
    return df
    
#### core function that creates the dataframe structure from the csv files

### Auxiliary functions called from create_or_load_dataframe

''' check
    reading_csv
    split_date
    join_list
    join_list_date
    convert_string_to_datetime
    convert_string_to_date'''
  
def check (fichero):
    if not fichero:
        a=False
    else:
        a=True
    return a

def reading_csv(Dpath):
    print(Dpath+'/Saved_data.csv')
    Dataf = pd.read_csv(Dpath+'/Saved_data.csv',parse_dates=['Fecha_completa','Fecha'],infer_datetime_format=True)
    Dataf.drop(Dataf.columns[0], axis=1, inplace=True)
    Dataf.sort_values(by=['Fecha_completa','Nodo o Region'], inplace=True)
    return Dataf

def reading_csv_Gen(Dpath):
    print(Dpath+'/Saved_data.csv')
    Dataf = pd.read_csv(Dpath+'/Saved_data.csv',parse_dates=['Fecha'],infer_datetime_format=True)
    Dataf.drop(Dataf.columns[0], axis=1, inplace=True)
    Dataf.sort_values(by=['Fecha'])
    return Dataf

def split_date (a):
    
    ''' Transforms a string including a date in format aa/month/yy to a string with a date in format aa/mm/yy'''
    
    a=a.split("/",2)

    if a[1] =='Enero':
        a[1]='01'
    if a[1] =='Febrero':
        a[1]='02'
    if a[1] =='Marzo':
        a[1]='03'
    if a[1] =='Abril':
        a[1]='04'
    if a[1] =='Mayo':
        a[1]='05'
    if a[1] =='Junio':
        a[1]='06'
    if a[1] =='Julio':
        a[1]='07'
    if a[1] =='Agosto':
        a[1]='08'
    if a[1] =='Septiembre':
        a[1]='09'
    if a[1] =='Octubre':
        a[1]='10'
    if a[1] =='Noviembre':
        a[1]='11'
    if a[1] =='Diciembre':
        a[1]='12'   
    return a

def join_list (list):
    list_str=str(list[0])
    for i in list[1:-1]:
        list_str +="/" + str(i)
    list_str+=" "+str(list[3])
    return list_str

def join_list_date (list):
    list_str=str(list[0])
    for i in list[1:]:
        list_str +="/" + str(i)
    return list_str

def convert_string_to_datetime(string):
    string=split_date (string)
    c=string[2].split(" ",1)
    newlist=[]
    for i in range(4):
        if i<2:
            newlist.append(string[i])
        else:
            newlist.append(c[i-2])
    return newlist

def convert_string_to_date(string):
    string=string.split("/",2)
    newlist=[]
    for i in range(3):
        newlist.append(string[i])
    return newlist


## Creation of dataframes

def read_n_create_files (csv_file):

    ''' Read a csv file and transform it in order to create a dataframe
    This DataFrame have the following columns:
    
    --- 'Fecha' : Date of the readings (one per file) string
    --- 'Fecha_completa': Date and hour in datetime
    --- 'Day':Day numeric
    --- 'Month': Month numeric
    --- 'Year':Year numeric
    --- 'Hour' : Hour of the reading (24 per file per 'Clave de nodo' or 'Region' 
    --- 'Zona de carga': Node code
    --- 'Cantidad total de energia asignada por Carga Directamente Modelada(MWh)' : Demand directly modelated
    --- 'Cantidad total de energia asignada por Carga Indirectamente Modelada(MWh)' : Demand indirectly modelated
    --- 'Cantidad total de energia asignada (MWh)' : Asignated demand

  
    (Intpus) : complete file address
    (output) : pandas Dataframe
    
    '''
    
    counter = 0
    features=[]
    data=[]
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            counter +=1
            if counter ==5:
                date = str(row)[9:-2]
            if counter == 8:
                features=row
            if counter>8:
                data.append(row)
         
    df= pd.DataFrame(data, columns=features[:6])
    df['Fecha']=date
    sdate = split_date(date)
    df['Día']=sdate[0]
    df['Mes']=sdate[1]
    df['Año']=sdate[2]
    df['Hora'] = df['Hora'].astype(int)-1
    df['Hora'] = df['Hora'].astype(str)
    df['Minuto']='00'

    # Adjusting hour '25', 
    df.loc[df['Hora']=='24',['Minuto']]='59'
    df.loc[df['Hora']=='24',['Hora']]='23'
    
    # Creation of feature "Fecha Completa" as datetime
    df['Fecha_completa']=df['Fecha']+' '+df['Hora']+':'+df['Minuto']
    df['Fecha_completa']=df['Fecha_completa'].apply(lambda x: datetime.strptime(join_list(convert_string_to_datetime(x)), "%d/%m/%Y %H:%M"))
    
    # Creation of feature "Fecha" as datetime
    df['Fecha']=df['Día']+'/'+df['Mes']+'/'+df['Año']
    df['Fecha']=df['Fecha'].apply(lambda x: datetime.strptime(join_list_date(convert_string_to_date(x)), "%d/%m/%Y"))
    
    # Disadjuting hour '25', 
    df.loc[df['Minuto']=='59',['Hora']]='24'
    df['Minuto']='00'
    df['Hora'] = df['Hora'].astype(int)+1
          
    # Adjustment in case "nodos" where included on the list
    
    if 'Clave del nodo' in features:
        df['Nodo o Region']=df['Clave del nodo']
        df['Precio']=df['Precio marginal local ($/MWh)']
        df.drop(['Precio marginal local ($/MWh)','Clave del nodo'], axis=1, inplace=True)
                         
        columns_order=['Fecha','Fecha_completa','Día','Mes','Año','Hora','Minuto','Nodo o Region', 'Precio','Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']
        
        df[['Día','Mes','Año','Minuto','Precio', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']]=df[['Día','Mes','Año','Minuto','Precio', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']].apply(pd.to_numeric)
        df=df[columns_order]
        
    # Adjustment in case "Zona de Carga" (PNDs) where included on the list
    elif 'Zona de Carga' in features:
        df['Nodo o Region']=df['Zona de Carga']
        df['Precio']=df['Precio Zonal  ($/MWh)']
        df['Componente de energia ($/MWh)']=df['Componente energia  ($/MWh)']
        df['Componente de perdidas ($/MWh)']=df['Componente perdidas  ($/MWh)']
        df['Componente de congestion ($/MWh)']=df['Componente Congestion  ($/MWh)']
        
        df.drop(['Precio Zonal  ($/MWh)','Zona de Carga','Componente energia  ($/MWh)','Componente perdidas  ($/MWh)','Componente Congestion  ($/MWh)'], axis=1, inplace=True)
                                                 
        columns_order=['Fecha','Fecha_completa','Día','Mes','Año','Hora','Minuto','Nodo o Region', 'Precio', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']
        df[['Día','Mes','Año','Minuto','Precio', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']]=df[['Día','Mes','Año','Minuto','Precio', 'Componente de energia ($/MWh)', 'Componente de perdidas ($/MWh)', 'Componente de congestion ($/MWh)']].apply(pd.to_numeric)
        df=df[columns_order]
        
    # Adjustment in case "zona de carga" (Demands) where included on the list of features
    elif 'Zona de carga' in features:
        df['Nodo o Region']=df['Zona de carga']
        df['Demanda D. mod (MWh)']=df['Cantidad total de energia asignada por Carga Directamente Modelada(MWh)']
        df['Demanda I. mod (MWh)']=df['Cantidad total de energia asignada por Carga Indirectamente Modelada(MWh)']
        df['Demanda total (MWh)']=df['Cantidad total de energia asignada (MWh)']

        df.drop(['Cantidad total de energia asignada por Carga Directamente Modelada(MWh)','Cantidad total de energia asignada por Carga Indirectamente Modelada(MWh)','Cantidad total de energia asignada (MWh)','Zona de carga'], axis=1, inplace=True)

        columns_order=['Fecha','Fecha_completa','Día','Mes','Año','Hora','Minuto','Nodo o Region', 'Demanda total (MWh)', 'Demanda D. mod (MWh)', 'Demanda I. mod (MWh)']
        df[['Día','Mes','Año','Minuto','Demanda total (MWh)', 'Demanda D. mod (MWh)', 'Demanda I. mod (MWh)']]=df[['Día','Mes','Año','Minuto','Demanda total (MWh)', 'Demanda D. mod (MWh)', 'Demanda I. mod (MWh)']].apply(pd.to_numeric)
        df=df[columns_order]
    return df

def merge_all (plds,demands,commodities,generation):
    mergeddf=pd.merge(left=plds,right=demands, how='outer',on=['Fecha','Fecha_completa','Nodo o Region','Día','Mes','Año','Hora','Minuto'])
    tmergeddf=pd.merge(left= mergeddf,right=commodities, how='outer', on=['Fecha','Día','Mes','Año'])
    tmergeddf.dropna(inplace=True)
    completedf=pd.merge(left=tmergeddf,right=generation, how='outer',on=['Fecha','Día','Mes','Año','Hora'])
    completedf.dropna(inplace=True)
    finaldf=completedf.sort_values(by=['Fecha_completa','Nodo o Region'])
    return finaldf


def mapping_nodes(df):
    region_dict={'ACAPULCO':1,'AGUASCALIENTES':2,'APATZINGAN':3,'CABORCA':4,'CAMARGO':5,'CAMPECHE':6,'CANCUN':7,'CARMEN':8,'CASAS GRANDES':9,'CELAYA':10,'CENTRO ORIENTE':11,'CENTRO SUR':12,'CHETUMAL':13,'CHIHUAHUA':14,'CHILPANCINGO':15,'CHONTALPA':16,'CIENEGA':17,'COATZACOALCOS':18,'COLIMA':19,'CONSTITUCION':20,'CORDOBA':21,'CUAUHTEMOC':22,'CUAUTLA':23,'CUERNAVACA':24,'CULIACAN':25,'DURANGO':26,'ENSENADA':27,'FRESNILLO':28,'GUADALAJARA':29,'GUASAVE':30,'GUAYMAS':31,'HERMOSILLO':32,'HUAJUAPAN':33,'HUASTECA':34,'HUATULCO':35,'HUEJUTLA':36,'IGUALA':37,'IRAPUATO':38,'IXMIQUILPAN':39,'IZUCAR':40,'JIQUILPAN':41,'JUAREZ':42,'LA PAZ':43,'LAGUNA':44,'LAZARO CARDENAS':45,'LEON':46,'LOS ALTOS':47,'LOS CABOS':48,'LOS MOCHIS':49,'LOS RIOS':50,'LOS TUXTLAS':51,'MANZANILLO':52,'MATAMOROS':53,'MATEHUALA':54,'MAZATLAN':55,'MERIDA':56,'MEXICALI':57,'MINAS':58,'MONCLOVA':59,'MONTEMORELOS':60,'MONTERREY':61,'MORELIA':62,'MORELOS':63,'MOTUL TIZIMIN':64,'NAVOJOA':65,'NOGALES':66,'NUEVO LAREDO':67,'OAXACA':68,'OBREGON':69,'ORIZABA':70,'PIEDRAS NEGRAS':71,'POZA RICA':72,'PUEBLA':73,'QUERETARO':74,'REYNOSA':75,'RIVIERA MAYA':76,'SABINAS':77,'SALTILLO':78,'SALVATIERRA':79,'SAN CRISTOBAL':80,'SAN JUAN DEL RIO':81,'SAN LUIS POTOSI':82,'SAN MARTIN':83,'SANLUIS':84,'TAMPICO':85,'TAPACHULA':86,'TECAMACHALCO':87,'TEHUACAN':88,'TEHUANTEPEC':89,'TEPIC VALLARTA':90,'TEZIUTLAN':91,'TICUL':92,'TIJUANA':93,'TLAXCALA':94,'TUXTLA':95,'URUAPAN':96,'VDM CENTRO':97,'VDM NORTE':98,'VDM SUR':99,'VERACRUZ':100,'VICTORIA':101,'VILLAHERMOSA':102,'XALAPA':103,'ZACAPU':104,'ZACATECAS':105,'ZAMORA':106,'ZAPOTLAN':107,'ZIHUATANEJO':108}
    df['Nodo o Region']=df['Nodo o Region'].map(region_dict)
    return df
def group_by_division(df):
    divisions_dict={'BAJA CALIFORNIA':['ENSENADA','MEXICALI','SANLUIS','TIJUANA'],
                    'BAJA CALIFORNIA SUR':['CONSTITUCION','LA PAZ','LOS CABOS'],
                    'BAJÍO':['AGUASCALIENTES','CELAYA','FRESNILLO','IRAPUATO','IXMIQUILPAN','LEON','QUERETARO','SALVATIERRA','SAN JUAN DEL RIO','ZACATECAS'],
                    'CENTRO OCCIDENTE':['APATZINGAN','COLIMA','JIQUILPAN','LAZARO CARDENAS','MANZANILLO','MORELIA','URUAPAN','ZACAPU','ZAMORA'],
                    'CENTRO ORIENTE':['CENTRO ORIENTE','IZUCAR','PUEBLA','SAN MARTIN','TECAMACHALCO','TEHUACAN','TLAXCALA'],
                    'CENTRO SUR':['ACAPULCO','CENTRO SUR','CHILPANCINGO','CUAUTLA','CUERNAVACA','IGUALA','MORELOS','ZIHUATANEJO'],'GOLFO CENTRO':['HUASTECA','HUEJUTLA','MATEHUALA','SAN LUIS POTOSI','TAMPICO','VICTORIA'],
                    'GOLFO NORTE':['MATAMOROS','MONCLOVA','MONTEMORELOS','MONTERREY','NUEVO LAREDO','PIEDRAS NEGRAS','REYNOSA','SABINAS','SALTILLO'],
                    'JALISCO':['CIENEGA','GUADALAJARA','LOS ALTOS','MINAS','TEPIC VALLARTA','ZAPOTLAN'],
                    'NOROESTE':['CABORCA','CULIACAN','GUASAVE','GUAYMAS','HERMOSILLO','LOS MOCHIS','MAZATLAN','NAVOJOA','NOGALES','OBREGON'],
                    'NORTE':['CAMARGO','CASAS GRANDES','CHIHUAHUA','CUAUHTEMOC','DURANGO','JUAREZ','LAGUNA'],
                    'ORIENTE':['COATZACOALCOS','CORDOBA','LOS TUXTLAS','ORIZABA','POZA RICA','TEZIUTLAN','VERACRUZ','XALAPA'],'PENINSULAR':['CAMPECHE','CANCUN','CARMEN','CHETUMAL','MERIDA','MOTUL TIZIMIN','RIVIERA MAYA','TICUL'],
                    'SURESTE':['CHONTALPA','HUAJUAPAN','HUATULCO','LOS RIOS','OAXACA','SAN CRISTOBAL','TAPACHULA','TEHUANTEPEC','TUXTLA','VILLAHERMOSA'],
                    'VALLE DE MEXICO CENTRO':['VDM CENTRO'],
                    'VALLE DE MEXICO NORTE':['VDM NORTE'],
                    'VALLE DE MEXICO SUR':['VDM SUR']}
    # Reverse dict
    regions_division_dict={}
    for key,value in divisions_dict.items():
        for item in value:
            regions_division_dict[item]=key
            
    dfdivision=df.copy()
    
    dfdivision['Nodo o Region']=df['Nodo o Region'].map(regions_division_dict)
    dfdivision=dfdivision.groupby(['Fecha_completa','Nodo o Region']).mean()
    
    return dfdivision

def split_train_predict(df):
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Componente de energia ($/MWh)'], df['Componente de energia ($/MWh)'], test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    results=model_comparison(df,X_train, y_train, X_val,y_val, X_test, y_test)
    return results

def train_predict(learner, sample_size, X_train, y_train, X_val,y_val, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time.time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size]) #learner.fit(X_train[:], y_train[:])
    end = time.time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end-start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time.time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train[:])
    end = time.time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
            
    # Compute mean_squared_error on the first 300 training samples which is y_train[:300]
    results['mse_train'] = mean_squared_error(y_train[:],predictions_train)
    
    # Compute mean_squared_error on the first 300 validation samples which is y_train[:300]
    results['mse_val'] = mean_squared_error(y_val[:],predictions_val)
        
    # Compute accuracy on test set using accuracy_score()
    results['mse_test'] = mean_squared_error(y_test,predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['mae_train'] = median_absolute_error(y_train[:],predictions_train)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    results['mae_val'] = median_absolute_error(y_val[:],predictions_val)        
    
    # Compute F-score on the test set which is y_test
    results['mae_test'] = median_absolute_error(y_test,predictions_test)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

def model_comparison(df,X_train, y_train, X_val,y_val, X_test, y_test):

    #base_estimatorABR=RandomForestRegressor(n_estimators=100)

    # TODO: Initialize the three models
    reg_A = DummyRegressor(strategy='mean')
    reg_B = LinearRegression()
    reg_C = RandomForestRegressor(n_estimators=100) #clf_C = SVC(kernel='poly',degree=3,C=100) corrected -- deleted all except random_state
    #reg_D = AdaBoostRegressor(base_estimator= base_estimatorABR,random_state=13) # deleted n_estimators = 100
    reg_D = AdaBoostRegressor(random_state=13) # deleted n_estimators = 100


    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    # HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    samples_100 = len(y_train)
    samples_10 = int(100/10)*samples_100
    samples_1 = int(100/10)*samples_10

    # Collect results on the learners
    results = {}
    for reg in [reg_A,reg_B,reg_C,reg_D]:
        reg_name = reg.__class__.__name__
        results[reg_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[reg_name][i] = \
            train_predict(reg, samples, X_train, y_train, X_val, y_val, X_test, y_test)
    return results
