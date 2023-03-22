from utils import *
# Given a row (list) from the power data, and another row from the weather data, 
# this function combines the two rows into a single row based on the state

def build_row(power, weath):
    merged_row = []

    # if the state is South Australia (Adeliade), remove the sunshine and evaporation features
    # since they do not exist
    # also remove the cloud amount
    if power[0] == 'SA1':
        merged_row.append(power[1])
        merged_row.extend(power[2:])
        merged_row.extend(weath[1:4])
        merged_row.extend(weath[6:11])
        merged_row.extend(weath[12:17])
        merged_row.extend(weath[18:])

    # if the state is Queensland (Brisbane), remove the evaporation feature since it does not exist
    elif power[0] == 'QLD1':
        merged_row.append(power[1])
        merged_row.extend(power[2:])
        merged_row.extend(weath[1:4])
        merged_row.extend(weath[5:])

    # the other states generally have all the information 
    else:
        merged_row.append(power[1])
        merged_row.extend(power[2:])
        merged_row.extend(weath[1:])

    return [merged_row]

# Converts the output list to a suitable dataframe with the correct titles based on the state/city
def convert_to_dataframe(outputlist,state):
    df = pd.DataFrame(outputlist)

    if state == 'SA1':
        df.columns = adelaide_column_titles     

    elif state == 'QLD1':
        df.columns = brisbane_column_titles
    
    else:
        df.columns = melb_sydney_column_titles
    
    return df

def merge(state, powerfile, weatherfile): 
    pricedf = pd.read_csv(powerfile)
    statedf = pd.read_csv(weatherfile)
    index = 0
    outarray = []
    sum_arr = []
    #this algorithm basically records all data over a certain day, and stops when it encounters a different day. 
    #it then can be used to merge the data by day.
    for i in statedf.index:
        date = str_to_date(statedf['Date'][i])
        #assuming that the dates are necessarily increasing in the powerdata, with respect to state.
        sum_arr = []
        for j in range(index,len(pricedf.index)):
            if (pricedf['REGION'][j] != state):
                continue
            date2 = str_to_date(pricedf['SETTLEMENTDATE'][j])
            if date2.date() != date.date():
                index = j
                break
            '''
            outarray.extend(build_row(pricedf.iloc[j].values,statedf.iloc[i].values)) #this can be used to load all power data, and not cut it by day.
            '''
            sum_arr.append(pricedf.iloc[j].values)
        
        df = pd.DataFrame(sum_arr)
        if not df.empty:
            powerarray = [df[0][0],statedf['Date'][i],np.mean(df[2]),any(df[3])] #take averages over the day, and set powersure to true if one occured.
            
            outarray.extend(build_row(powerarray,statedf.iloc[i].values)) #build_row will create the output row, from two rows we want to combine
        
    # Convert the output to a dataframe
    df = convert_to_dataframe(outarray,state)
    return df

