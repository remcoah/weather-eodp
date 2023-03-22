from utils import *
from merge import *
from preprocessing import *
from analysis import *

#mergine, preprocessing and analysis were split up to make testing faster.
do_merge = True
do_preprocess = True
do_analysis = True

sydney_out = None 
adelaide_out = None
brisbane_out = None
melbourne_out = None

if(not os.path.exists(merge_path)):
    os.mkdir(merge_path)

if do_merge:
    adelaide_out = merge('SA1', 'weather/price_demand_data.csv', 'weather/weather_adelaide.csv')
    brisbane_out = merge('QLD1', 'weather/price_demand_data.csv', 'weather/weather_brisbane.csv')
    melbourne_out = merge('VIC1', 'weather/price_demand_data.csv', 'weather/weather_melbourne.csv')
    sydney_out = merge('NSW1', 'weather/price_demand_data.csv', 'weather/weather_sydney.csv')
    sydney_out.to_csv(merge_path + 'sydney_merged_data.csv')
    adelaide_out.to_csv(merge_path + 'adelaide_merged_data.csv')
    brisbane_out.to_csv(merge_path + 'brisbane_merged_data.csv')
    melbourne_out.to_csv(merge_path + 'melbourne_merged_data.csv')
else:
    sydney_out = pd.read_csv(merge_path + 'sydney_merged_data.csv',index_col=[0]) 
    adelaide_out = pd.read_csv(merge_path + 'adelaide_merged_data.csv',index_col=[0]) 
    brisbane_out = pd.read_csv(merge_path + 'brisbane_merged_data.csv',index_col=[0]) 
    melbourne_out = pd.read_csv(merge_path + 'melbourne_merged_data.csv',index_col=[0])

# All the preprocessing will be done on the df
# Call preprocessing functions here
sydney_train_test = None
adelaide_train_test = None
brisbane_train_test = None
melbourne_train_test = None

if do_preprocess:
    sydney_train_test = preprocess('NSW1',sydney_out) #preprocess returns a list of length 2, with the train and test dataframes split up.
    adelaide_train_test = preprocess('SA1',adelaide_out)
    brisbane_train_test = preprocess('QLD1',brisbane_out)
    melbourne_train_test = preprocess('VIC1',melbourne_out)

city_arr = [['NSW1',sydney_train_test],['SA1',adelaide_train_test],['QLD1',brisbane_train_test],['VIC1',melbourne_train_test]] #for iteration.

if(not os.path.exists(processed_path)):
    os.mkdir(processed_path)
for v in city_arr:
    if do_preprocess:
        #save the preproccesing
        pd.concat(v[1],keys = ['train','test']).to_csv(processed_path+v[0] + '_preprocessed_data.csv')
    else:
        #load the preprocessed train/test dataset.
        pped =  pd.read_csv(processed_path+v[0] + '_preprocessed_data.csv',index_col=[1])
        pped.columns.values[0] = 'type'
        group = pped.groupby(['type'])
        v[1] = [group.get_group('train').drop('type', axis=1), group.get_group('test').drop('type', axis=1)]
        
if do_analysis:
    for v in city_arr:
        #do analysis on preprocessed data.
        graphing(v[0],pd.concat(v[1]))
        classification_model = classification(v[1][0])
        classification_eval(v[0],classification_model,v[1][1])
        regression_model = regression(v[1][0])
        regression_eval(v[0],regression_model,v[1][1])



