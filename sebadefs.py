def distribution_analysis(df_column):
    """
    This funtion can detect what type of distribution has a non categorical variable.
    Should be passed has dataframe column
    # https://github.com/cokelaer/fitter
    """
    from fitter import Fitter
    from fitter import get_common_distributions
   
    # If i want to check inly the most commmon distributions
    dist_list = get_common_distributions()
    
    
    f = Fitter(data= df_column)#, distributions= dist_list)
    f.fit()
    return(f.summary()) #f.get_best()
    

def column_to_int(df,column): 
    """
    Convert dataframe column to int64, and if can´t convert NaN is placed (errors='coerce')
    Then drop rows where we have NaNs.
    """
    import pandas as pd
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df_int = df.dropna(subset=[column])
    # convert to int64 
    df_int = df_int.astype({column: int})
    
    return df_int


def clean_column_from_special_chars(df,column):
    """
    Clean special characters from a column dataframe.
    The input is converted to string (as an object for a dataframe).
    """
    df[column] = df[column].astype(str)
    df[column] = df[column].str.replace(r'\W', '',regex=True)
    return df[column]


def clean_column_from_alphabetic_chars(df,column):
    """
    Clean all non number characters from a column dataframe .
    The input is converted to string (as an object for a dataframe).
    """
    df[column] = df[column].astype(str)
    df[column] = df[column].str.replace(r'[^\d+]', '',regex=True)
    return df[column]