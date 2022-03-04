import pandas as pd

def prep_iris(df):
    df = df.drop(columns=['species_id', 'measurement_id'])
    df = df.rename(columns={'species_name': 'species'})
    dummy_df = pd.get_dummies(df['species'], drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)    
    return df

def prep_titanic(df, drop_after_encoding=True):
    # drop duplicate rows, if they exist:
    df = df.drop_duplicates()
    # drop unnecessary columns
    df = df.drop(columns=['class', 'embarked', 'deck', 'age', 'passenger_id'])
    # rename columns
    df = df.rename(columns={'parch': 'n_parents_and_children', 'sibsp': 'n_sibs_and_spouse'})
    # add family size column
    df['family_size'] = df.n_parents_and_children + df.n_sibs_and_spouse
    # encode categorical columbns with dummy variables then drop the original columns
    categorical_columns = ['sex', 'embark_town']
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        if drop_after_encoding:
            df = df.drop(columns=col)
    return df

def prep_telco(df):
    # drop duplicate rows, if present
    df = df.drop_duplicates()
    # drop columns:
    # *_type_id columns are simply foreign key columns that have corresponding string values
    # customer_id is a primary key that is not useful for our analysis
    df = df.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id', 'customer_id'])
    # encode categorical columns with dummy variables
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=df[col].name,
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        df = df.drop(columns=col)
    return df