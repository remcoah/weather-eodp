from utils import *
# given a csv file from the merged data, this function conducts the necessary 
# preprocessing steps to conduct analysis
TEST_SIZE = 0.3 #change if you want

def drop_cols(statefile): 
    """ This function removes irrelevant data attributes from the statefile dataframe"""
    statefile.drop('Direction of Maximum Wind Gust', inplace=True, axis=1)
    statefile.drop('9am Wind Direction', inplace=True, axis=1)
    statefile.drop('9am MSL Pressure (hPa)', inplace=True, axis=1)
    statefile.drop('3pm Wind Direction', inplace=True, axis=1)
    statefile.drop('3pm MSL Pressure (hPa)', inplace=True, axis=1)
    statefile.drop('Time of Maximum Wind Gust', inplace=True, axis=1)
    statefile.drop('Date', inplace=True, axis=1)
    return statefile

def clean_data(statefile): 
    """This function makes all data the same data type and converts all numerical values into floats"""
    cols = list(statefile.columns)

    statefile.loc[:,cols] = statefile.loc[:,cols].apply(pd.to_numeric, errors='coerce')
    # ONLY ISSUE WITH THIS CODE -> calm turns into null, but we can fix that in the data impuration phase
    return statefile

def impute_data(state,statefile):
    """This function imputes the data using regression imputation or imputing values to 0 when appropriate"""

    # check each column to see if any data needs to be imputed
    totalcountm = 0
    totalcountw = 0
    for column in statefile:
        if statefile[column].isnull().values.any():
            if (column in windspeed_column_titles) or (column == 'Time of Maximum Wind Gust'):
                # impute 'calm' windspeed values and their record time to 0
                statefile[column].fillna(0, inplace=True)

    # perform regression imputation.
    imp = IterativeImputer(max_iter=10,estimator = RandomForestRegressor(n_estimators=5,bootstrap=True,n_jobs=-1,random_state=0))
    targeted_cols = mean_imputation_column_titles(state)
    imputed_df = (pd.DataFrame(imp.fit_transform(statefile[targeted_cols]),columns = statefile[targeted_cols].columns))
    statefile[targeted_cols] = (pd.DataFrame(imp.fit_transform(statefile[targeted_cols]),columns = statefile[targeted_cols].columns))[targeted_cols]

    return statefile

def MutualInformation(state, statefile):
    "This function determines the mutual information between the different attributes for a given state"
    data = {}
    transf = KBinsDiscretizer(n_bins = 3, encode = 'ordinal', strategy = 'quantile')
    datak = transf.fit_transform(statefile)
    discreted = pd.DataFrame(datak,columns = statefile.columns, index=statefile.index)
    discreted['Price Surge'] = statefile['Price Surge'].values
    for c1 in discreted.columns:
        data[c1] = []
        for c2 in discreted.columns:
            data[c1].append(normalized_mutual_info_score(discreted[c1],discreted[c2]))
    dfo = pd.DataFrame(data,index=data.keys())
    dfo.to_csv(state[0:-1] + ' mutual_info.csv')
    return statefile

def change_in_power(statefile): 
    """
    This function determines the change in power demand and adds this attribute to the statefile dataframe
    """
    
    power_demand_change = []
    prev_demand = statefile['Total Demand'].values[0]
    for value in statefile['Total Demand'].values:
        power_demand_change.append(value - prev_demand)
        prev_demand = value

    statefile.insert(statefile.columns.get_loc("Total Demand") + 1, 'Change in Demand', power_demand_change)

    return statefile

def norm_data(state,statefile):
    """
    This function normalises all numerical data using min max normalisation
    """

    for column in statefile:
        if is_numeric_dtype(statefile[column]) and not pd.api.types.is_bool_dtype(statefile[column]):

            # normalise data using min max normalisation
            statefile[column] = (statefile[column]-statefile[column].min()) / (statefile[column].max() - statefile[column].min())

    return statefile


def bound_data(state,statefile): 
    """
    Removing outliers based on z scores
    Remove datapoints outside of the bound
    """
    totalcountw = 0
    totalcountm = 0
    for column in statefile:

        if is_numeric_dtype(statefile[column]) and not pd.api.types.is_bool_dtype(statefile[column]) and column != 'Rainfall (mm)':
            outliers = stats.zscore(statefile[column]).apply(lambda x: np.abs(x) < 3)
            for i in range(0,len(statefile[column])):
                
                if not outliers.iloc[0]:
                    statefile[column][i] = None
                    if column in windspeed_column_titles:
                        totalcountw += 1
                    else:
                        
                        totalcountm += 1 
    
    return statefile

def do_boxplot(statefile):
    #generates a boxplot.
    myFig = plt.figure();
    bp = statefile.boxplot()
    plt.xticks(rotation = 90)
    if(os.path.exists('boxplot.png')):
        os.remove("boxplot.png")
    myFig.savefig("boxplot.png", format="png")
    plt.close()

def prepreprocess_graphs(state,statefile):
    '''
    this function generates graphs.
    '''
    axes = []
    #below code generates the distribution of an attribute as a Histogram
    if(not os.path.exists(graph_path + state + '/')):
        os.mkdir(graph_path + state + '/')
    for col in statefile.columns:
        if col in dist_graph_columns(state):
            hist = statefile.hist(column = col,bins=20)
            for ax in hist.flatten():
                ax.set_xlabel(display_names[col])
                ax.set_ylabel('Total Observed')
                ax.set_title('Distribution of ' + display_names[col] + ' in ' + state[0:-1])
            axes.append(hist)
            if(os.path.exists(graph_path + state + '/' + display_names[col] +'_distribution.png')):
                os.remove(graph_path + state + '/' + display_names[col] +'_distribution.png')
            plt.savefig(graph_path + state + '/' + display_names[col] +'_distribution.png')
            plt.close()
    
    #the below part displays all boxplots in one figure.
    width  =math.ceil(math.sqrt(len(axes)))
    figure, axis = plt.subplots(math.ceil(len(axes)/width), width)
    i = 0
    figure.suptitle(state[0:-1])
    SMALL_SIZE = 8
    plt.rc('font', size=SMALL_SIZE)
    for col in statefile.columns:
        if col in dist_graph_columns(state):
            ax = axis[math.floor(i/width),i%width]
            ax.hist(statefile[col],bins=15)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title(display_names[col])
            i += 1
    figure.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.6)
    if(os.path.exists(graph_path + state + '/many_distributions.png')):
       os.remove(graph_path + state + '/many_distributions.png')
    figure.savefig(graph_path + state + '/many_distributions.png')
    plt.close()

def preprocess(state,statefile): # Statefile is a dataframe not the csv
    """
    Goes through every single task and returns updated dataframe object
    """
    statefile =  drop_cols(statefile)
    statefile = clean_data(statefile)
    # split data here.
    prepreprocess_graphs(state,statefile)
    split_data = train_test_split(statefile, test_size=TEST_SIZE)
    
    for i in range(0,len(split_data)):
        split_data[i] = split_data[i].reset_index(drop = True)
        split_data[i] =  impute_data(state,split_data[i])
        split_data[i] = bound_data(state,split_data[i])
        split_data[i] =  impute_data(state,split_data[i])#have to impute after bounding
        split_data[i] = change_in_power(split_data[i])
        split_data[i] = norm_data(state,split_data[i])
    MutualInformation(state, split_data[0])
    
    return split_data













