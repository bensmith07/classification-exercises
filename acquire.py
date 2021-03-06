import os
import pandas as pd
from env import get_db_url, user, password, host

# 1. Make a function named get_titanic_data that returns the titanic 
# data from the codeup data science database as a pandas data frame. 
# Obtain your data from the Codeup Data Science Database. 

def get_titanic_data():
    
    filename = 'titanic.csv'
    
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    url = get_db_url('titanic_db')
    sql = '''
    SELECT * FROM passengers
    '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)

    print('Saving to local CSV... ')
    df.to_csv(filename, index=False)
    
    return df

# 2. Make a function named get_iris_data that returns the data from the 
# iris_db on the codeup data science database as a pandas data frame. 
# The returned data frame should include the actual name of the species 
# in addition to the species_ids. Obtain your data from the Codeup Data 
# Science Database.
def get_iris_data():
    
    filename = 'iris.csv'
    
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
        
    url = get_db_url('iris_db')
    sql = '''
    SELECT *
      FROM species
      JOIN measurements USING(species_id);
    '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)
    
    print('Saving to local CSV...')
    df.to_csv(filename, index=False)
    
    return df

# 3. Make a function named get_telco_data that returns the data from the 
# telco_churn database in SQL. In your SQL, be sure to join all 4 tables 
# together, so that the resulting dataframe contains all the contract, 
# payment, and internet service options. Obtain your data from the 
# Codeup Data Science Database.

def get_telco_data():
    
    filename = 'telco_churn.csv'
    
    if os.path.exists(filename):
        print('Reading from local CSV...')
        return pd.read_csv(filename)
    
    url = get_db_url('telco_churn')
    sql = '''
    SELECT * 
      FROM customers
        JOIN contract_types USING(contract_type_id)
        JOIN internet_service_types USING(internet_service_type_id)
        JOIN payment_types USING(payment_type_id)
    '''
    
    print('No local file exists\nReading from SQL database...')
    df = pd.read_sql(sql, url)
    
    print('Saving to local CSV...')
    df.to_csv(filename, index=False)
    
    return df

# 4. Once you've got your get_titanic_data, get_iris_data, and get_telco_data 
# functions written, now it's time to add caching to them. To do this, edit the 
# beginning of the function to check for the local filename of telco.csv, titanic.csv, 
# or iris.csv. If they exist, use the .csv file. If the file doesn't exist, then 
# produce the SQL and pandas necessary to create a dataframe, then write the 
# dataframe to a .csv file with the appropriate name.
