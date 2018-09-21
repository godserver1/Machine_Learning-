import numpy as np
import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import export_graphviz
import pydot

def NodesAndLeaf(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]
    n_leaf = 0
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True
            n_leaf += 1
    return n_nodes,n_leaf

#=======================================================Data Extraction================================================================================

Data_bike = np.genfromtxt("bikes.csv",delimiter=',')
Data = Data_bike[1:,1:]
count = Data[:,Data.shape[1]-1]
Data = Data[:,:Data.shape[1]-1]

#==========================================================Finding Best Possible Parameters to avoid Overfit================================================================================

#--------------------------------------Train Test Split to find the best possible split percentage-------------------------------------------------
score = np.zeros((20,2))    
for i in range(1,20):
    part = 0.05*(i)
    X_train, X_test, y_train, y_test = train_test_split(Data, count, test_size=part)
    regressor = DecisionTreeRegressor()
    regressor = regressor.fit(X_train, y_train)
    E = regressor.predict(X_test)
    score[i][0] = part
    score[i][1] = regressor.score(X_test,y_test)
score[0][1] = score[1][1]
score[0][0] = score[1][0]
plt.plot(score[:,0], score[:,1], 'xb-')
plt.title("accuracy_Vs_Split_percentage")
plt.xlabel("Split_percentage")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy_Vs_Split_percentage')
plt.show()
use = np.argmax(score[:,1])
#print use
use_percent = use*0.05
if (use_percent == 0.0):
    use_percent = 0.5
#---------------------------------------To find the best possible depth for the gives split-----------------------------------------------------------------
depth = np.zeros((20,2))
X_train, X_test, y_train, y_test = train_test_split(Data, count, test_size=use_percent)
for i in range(0,20):
    regressor = DecisionTreeRegressor(max_depth=i+1)
    regressor = regressor.fit(X_train, y_train)
    E = regressor.predict(X_test)
    depth[i][1] = regressor.score(X_test,y_test)
    depth[i][0] = i + 1
plt.plot(depth[:,0], depth[:,1], 'xb-')
plt.title("accuracy_Vs_depth")
plt.xlabel("Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy_Vs_depth')
plt.show()
depth_best1 = np.argmax(depth[:,1])
depth_best1 = int(depth_best1)
#---------------------------------------- Impurity decrease if we cut it to some finite no of roots--------------------------------------------------------------------------------------------------------------------

imp = np.zeros((50,2))
impind = np.zeros(10)
for j in range(0,10):
    for i in range(0,50):
        reg = DecisionTreeRegressor(min_impurity_decrease=i)
        reg = reg.fit(Data,count)
        count_predicted = reg.predict(Data)
        imp[i][1] = mean_squared_error(count, count_predicted)
        (nodes,leafs) = NodesAndLeaf(reg)
        imp[i][0] = nodes
    impind[j] = np.argmax(imp[:,1])
min_impurity_decrease_val = np.mean(impind)
min_impurity_decrease_val = int(min_impurity_decrease_val)
#---------------------------K fold cross validation-------------------------------------------------------

ind = np.zeros((21,2))
inded = np.zeros(10)
for i in range(0,10):
    for j in range(0,20):
        clf = tree.DecisionTreeRegressor(max_depth=j+1)
        clf = clf.fit(Data,count)
        cv = ShuffleSplit(n_splits=10, test_size = 0.3)
        N = cross_val_score(clf, Data, count, cv = cv)
        ind[j][0] = j+1
        temp = np.mean(N)
        ind[j][1] = temp
    inded[i] = np.argmax(in0d[:,1])
#print np.mean(inded)
depth_best2 = np.mean(inded)
depth_best2 = int(depth_best2)
#--------------------------------------Buliding the tree-----------------------------------------------------------------

MSE = np.zeros(3)
MSE_roots = [0.0 for x in range(0,3)]
#------------------------------------Using first criteria------------------------------------------
regressor = DecisionTreeRegressor(max_depth=depth_best1)
regressor = regressor.fit(Data, count)
export_graphviz(regressor,out_file='tree1.dot')
(graph,) = pydot.graph_from_dot_file('tree1.dot')
graph.write_png('tree1.png')
E=regressor.predict(Data)
print "\n"
print "==================================================================================="
print "for criteria when we take depth_best1 as parameter for given data set"
print "Nodes and leafs: "
print NodesAndLeaf(regressor)
(nodes,leafs) = NodesAndLeaf(regressor)
print "depth best1 : ", depth_best1
#-------------------------------The MSE---------------------------------------------------------------
print "Mean Square Error is: ", mean_squared_error(count, E)
MSE[0] = mean_squared_error(count,E)
MSE_roots[0] = float(MSE[0]*nodes)
#--------------------------------------------------------------------------------------------------------
print "Score is :", regressor.score(Data,count)
#-------------------------------code for Important Variables--------------------------------------------
importance = np.zeros(10)
importance_index = np.zeros(10)
importance = regressor.feature_importances_
#print importance
indices = np.zeros(10)
indices = np.argsort(importance)
print "Order of importance of features (Lowest First)"
print indices                                    #reverse order
Max = regressor.max_features_
print "Max Feature is :", Max
no_of_futures = regressor.n_features_
print "No of Feature is :", no_of_futures
#------------------------------------------------------------------------------------------------------------------

#------------------------------------Using Second criteria------------------------------------------
regressor = DecisionTreeRegressor(min_impurity_decrease=min_impurity_decrease_val)
regressor = regressor.fit(Data, count)
export_graphviz(regressor,out_file='tree2.dot')
(graph,) = pydot.graph_from_dot_file('tree2.dot')
graph.write_png('tree2.png')
E=regressor.predict(Data)
print "\n"
print "==================================================================================="
print "for criteria when we take min_impurity_decrease as parameter for given data set"
print "Nodes and leafs: "
print NodesAndLeaf(regressor)
(nodes,leafs) = NodesAndLeaf(regressor)
print " min impurity decrease : ", min_impurity_decrease_val
#-------------------------------The MSE---------------------------------------------------------------
print "Mean Square Error is: ", mean_squared_error(count, E)
MSE[1] = mean_squared_error(count,E)
MSE_roots[1] = float(MSE[1]*nodes)
#--------------------------------------------------------------------------------------------------------
print "Score is :", regressor.score(Data,count)
#-------------------------------code for Important Variables--------------------------------------------
importance = np.zeros(10)
importance_index = np.zeros(10)
importance = regressor.feature_importances_
#print importance
indices = np.zeros(10)
indices = np.argsort(importance)
print "Order of importance of features (Lowest First)"
print indices                                    #reverse order
Max = regressor.max_features_
print "Max Feature is :", Max
no_of_futures = regressor.n_features_
print "No of Feature is :", no_of_futures
#------------------------------------------------------------------------------------------------------------------

#------------------------------------Using Third criteria------------------------------------------
regressor = DecisionTreeRegressor(max_depth = depth_best2)
regressor = regressor.fit(Data, count)
export_graphviz(regressor,out_file='tree3.dot')
(graph,) = pydot.graph_from_dot_file('tree3.dot')
graph.write_png('tree3.png')
E=regressor.predict(Data)
(nodes,leafs) = NodesAndLeaf(regressor)
print "\n"
print "==================================================================================="
print "for criteria when we take depth_best2 as parameter for given data set"
print "Nodes and leafs:"
print NodesAndLeaf(regressor)
print "depth best2 :", depth_best2
#-------------------------------The MSE---------------------------------------------------------------
print "Mean Square Error is: ", mean_squared_error(count, E)
MSE[2] = mean_squared_error(count,E)
MSE_roots[2] = float(MSE[2]*nodes)
#--------------------------------------------------------------------------------------------------------
print "Score is :", regressor.score(Data,count)
#-------------------------------code for Important Variables--------------------------------------------
importance = np.zeros(10)
importance_index = np.zeros(10)
importance = regressor.feature_importances_
#print importance
indices = np.zeros(10)
indices = np.argsort(importance)
print "Order of importance of features (Lowest First)"
print indices                                    #reverse order
Max = regressor.max_features_
print "Max Feature is :", Max
no_of_futures = regressor.n_features_
print "No of Feature is :", no_of_futures
#------------------------------------------------------------------------------------------------------------------
print "\n"
print "==================================================================================="
print "Errors in all 3 types are :", MSE
print "Errors multiplied by no. of nodes in all 3 types are :", MSE_roots
#------------------------------------------------------------------------------------------------------------------
#=================================================For Edited Data Set==========================================================
Data_new = Data
#print Data_new

(N,D) = Data.shape
#print N

for i in range(0,N):
    if(Data_new[i][2]==1 or Data_new[i][2]==2):
        Data_new[i][2]=1
    elif(Data_new[i][2]==5 or Data_new[i][2]==6 or Data_new[i][2]==7 or Data_new[i][2]==8 or Data_new[i][2]==9 or Data_new[i][2]==10):
        Data_new[i][2]=2
    else:
        Data_new[i][2]=3
#print Data_new[:,2]

regressor2 = DecisionTreeRegressor(max_depth=depth_best2)
regressor2 = regressor2.fit(Data_new, count)
export_graphviz(regressor,out_file='tree_case2.dot')
(graph,) = pydot.graph_from_dot_file('tree_case2.dot')
graph.write_png('tree_case2.png')
E=regressor2.predict(Data_new)
(nodes,leafs) = NodesAndLeaf(regressor2)
print "\n"
print "==================================================================================="
print "For case two i.e. edited values for the months taking depth_best2 parameter:"
print "Nodes and leafs:"
print NodesAndLeaf(regressor2)
#-------------------------------The MSE---------------------------------------------------------------
print "Mean Square Errors is: ", mean_squared_error(count, E)
#--------------------------------------------------------------------------------------------------------
print "Score is :", regressor2.score(Data_new,count)
#-------------------------------code for Important Variables--------------------------------------------
importance = np.zeros(10)
importance_index = np.zeros(10)
importance = regressor2.feature_importances_
#print importance
indices = np.zeros(10)
indices = np.argsort(importance)
print "Order of importance of features (Lowest First)"
print indices                                    #reverse order
Max = regressor2.max_features_
print "Max Feature is :", Max
no_of_futures = regressor2.n_features_
print "No of Feature is :", no_of_futures