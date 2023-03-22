from utils import *
# given a csv file from the merged data, this function conducts the necessary 
# Analysis
"""
To do: Write a function for each of the following steps
    graphing for preprocessing/understanding, and what regressions to use.
    train Regression models, predict total demand.
    train Classification models, predict price surge.
    probably 70:30 split. standard stuff.

    create graphs to include in report. and also some confusion matrixes, precision recall etc.
    Basically we want to be able to explain what our results were.
    Have some graphs that show the characteristics of the data. Histogram, box plot, etc.
    graph change of power demand.

    do day by day analysis. to understand power draw during the day, and weather during the day.
"""
X_COLS = ['9am Temperature (Â°C)', '3pm Temperature (Â°C)']
Y_COL = 'Total Demand'

def regression(train_data): #do linear regression.
    X_train = train_data[X_COLS]
    Y_train = train_data[Y_COL]
    # Create and fit the linear model
    lm = LinearRegression()
    # Fit to the train dataset
    lm.fit(X_train, Y_train)
    return lm

def classification(train_data): #do random forest classification
    clf = RandomForestClassifier(n_estimators = 100,class_weight='balanced',bootstrap = False)
    loc = train_data.columns.get_loc("Price Surge")
    y = train_data['Price Surge']
    X = train_data.loc[:, train_data.columns!='Price Surge']

    clf.fit(X, y)
    
    return clf

def regression_eval(state, lm, test_data):
    X_test = test_data[X_COLS]
    Y_test = test_data[Y_COL]

    Y_pred = lm.predict(X_test)

    r2 = lm.score(X_test, Y_test)
    mse = mean_squared_error(Y_test, Y_pred)

    residuals = Y_test - Y_pred

    # print('R2', r2)
    # print('MSE', mse)
    # print()
    # print(Y_test)
    # print(Y_pred)
    plt.close()
    # This is the first graph 
    # The Linear Regression predicted vs actual values
    plt.scatter(Y_test, Y_pred, alpha=0.3)

    plt.title('Linear Regression (Predict Total) '+ state[0:-1])
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    
    if(os.path.exists(graph_path + state + '/' +state + ' regression.png')):
        os.remove(graph_path + state + '/' +state + ' regression.png')
    plt.savefig(graph_path + state + '/' + state + ' regression.png')
    plt.close()

    # This is the second graph
    # The Residuals plot
    plt.scatter(Y_pred, residuals, alpha=0.3)

    plt.plot([min(Y_pred), max(Y_pred)], [0,0], color='red')

    plt.title('Residual Plot '+ state[0:-1])
    plt.xlabel('Fitted')
    plt.ylabel('Residual')
    if(os.path.exists(graph_path + state + '/' +state + ' regression_residuals.png')):
        os.remove(graph_path + state + '/' +state + ' regression_residuals.png')
    plt.savefig(graph_path + state + '/' + state + ' regression_residuals.png')
    plt.close()
    return 0
    
def classification_eval(state,clf, test_data): 
    y = test_data['Price Surge']
    X = test_data.loc[:, test_data.columns!='Price Surge']
    y_pred = clf.predict_proba(X)
    
    df = pd.DataFrame(y_pred)
    #create a confusion matrix to display results.
    plot_confusion_matrix(clf, X, y)  
    plt.title('classification on Price Surge in ' + state[0:-1])
    if(os.path.exists(graph_path + state + '/' +state + ' confusion.png')):
        os.remove(graph_path + state + '/' +state + ' confusion.png')
    
    plt.savefig(graph_path + state + '/' +state + ' confusion.png')
    
    plt.close()
    return 0

def graphing(state, statefile):
    if(not os.path.exists(graph_path + state + '/')):
        os.mkdir(graph_path + state + '/')

    #make double boxplot graphs, for conditioal distribution comparisons.
    for col in statefile.columns:
        if col in box_plot_columns(state):
            myFig = plt.figure();
            group = statefile[[col,'Price Surge']].groupby(['Price Surge']) 
            data = [group.get_group(True)[col].values,group.get_group(False)[col].values]
            fig, ax = plt.subplots()
            bp = ax.boxplot(data,patch_artist=True)
            fig.suptitle('Distribution of ' + display_names[col] + ' Conditioned on Price surge in ' + state[0:-1])
            ax.set_xticklabels([True,False])
            #for patch in bp['boxes']:
                #patch.set(facecolor='blue') 
            ax.set_xlabel('Price Surge') 
            ax.set_ylabel(display_names[col] + ' (normalised)')

            if(os.path.exists(graph_path + state + '/' + 'boxplot_' +display_names[col]+ '_price_relationship.png')):
                os.remove(graph_path + state + '/' + 'boxplot_' +display_names[col]+ '_price_relationship.png')
            fig.savefig(graph_path + state + '/' + 'boxplot_' +display_names[col]+ '_price_relationship.png', format="png")
            plt.close()
    #make scatter plots between all attributes.
    for i in range(0,len(statefile.columns)):
        if statefile.columns[i] in dist_graph_columns(state):
            for j in range(i+1,len(statefile.columns)):
                if statefile.columns[j] in dist_graph_columns(state):
                    #make scatter for both
                    scat = statefile.plot.scatter(x = statefile.columns[j],y = statefile.columns[i])
                    #for ax in scat.flatten():
                    scat.set_xlabel(display_names[statefile.columns[j]] + ' (normalised)')
                    scat.set_ylabel(display_names[statefile.columns[i]]+ ' (normalised)')
                    scat.set_title('Scatter plot of ' + display_names[statefile.columns[i]] + ' VS ' + display_names[statefile.columns[j]] + ' in ' + state[0:-1])
                    if(not os.path.exists(graph_path + state + '/scatters/')):
                        os.mkdir(graph_path + state + '/scatters/')
                    if(os.path.exists(graph_path + state + '/scatters/' + display_names[statefile.columns[i]] + ' VS ' + display_names[statefile.columns[j]] +'.png')):
                        os.remove(graph_path + state + '/scatters/' + display_names[statefile.columns[i]] + ' VS ' + display_names[statefile.columns[j]] +'.png')
                    plt.savefig(graph_path + state + '/scatters/' + display_names[statefile.columns[i]] + ' VS ' + display_names[statefile.columns[j]] +'.png')
                    plt.close()
