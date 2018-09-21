import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydot


def NodesLeaf(model):
    val = 0
    no_of_nodes = model.tree_.node_count
    right_child = model.tree_.children_right
    left_child = model.tree_.children_left
    node_depth = np.zeros(shape=no_of_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=no_of_nodes, dtype=bool)
    stack = [(0, -1)]
    no_of_leaf = 0
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        if (left_child[node_id] != right_child[node_id]):
            stack.append((left_child[node_id], parent_depth + 1))
            stack.append((right_child[node_id], parent_depth + 1))
            val = val + 1
        else:
            is_leaves[node_id] = True
            no_of_leaf += 1
        #print val
    return no_of_nodes,no_of_leaf


Data_train = np.loadtxt(open("trainX.csv","r"),dtype="float",delimiter=',')
Data_train_labels = np.loadtxt(open("trainY.csv","r"),dtype="int",delimiter=',')
model = DecisionTreeClassifier()
model.fit(Data_train,Data_train_labels)


(nodes,leafs) = NodesLeaf(model)


export_graphviz(model,out_file='tree.dot')
#print(model)


(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')



Data_test = np.loadtxt(open("testX.csv","r"),dtype='float',delimiter=',')
Data_test_labels = np.loadtxt(open("testY.csv","r"),dtype='float',delimiter=',')

expected = Data_test_labels
predicted = model.predict(Data_test)


#print(metrics.classification_report(expected,predicted))
confusion_matrix = metrics.confusion_matrix(expected,predicted)
#print(confusion_matrix)
miss1 = float(confusion_matrix[0][1])*100/(confusion_matrix[0][0] + confusion_matrix[0][1])
miss2 = float(confusion_matrix[1][0])*100/(confusion_matrix[1][0] + confusion_matrix[1][1])


f = open("P1_results.txt", "w")
f.write('No of nodes : %d\n' %nodes)
f.write('No of leafs : %d\n' %leafs)
f.write('Confusion Matrix[0][0] : %d\n' %confusion_matrix[0][0])
f.write('Confusion Matrix[0][1] : %d\n' %confusion_matrix[0][1])
f.write('Confusion Matrix[1][0] : %d\n' %confusion_matrix[1][0])
f.write('Confusion Matrix[1][1] : %d\n' %confusion_matrix[1][1])
f.write('Missclassification rate of class 1: %d %% \n'%miss1)
f.write('Missclassification rate of class 2: %d %% \n'%miss2)
f.close()


N = 0
D = 0
(N,D) = Data_train.shape
#print N
#print D



N_new = 0
index = N + 1

#----------------using part of train data, via using random variable generator-------------------------------------

Data_out1 = np.zeros((10,4))
Data_out2 = np.zeros((10,4))
for l in range(0,1):
    for i in range(0,10):
        N_new = N*(i+1)/10
        Data_array = np.zeros((N_new,D))
        Data_array_labels = np.zeros(N_new)
        for j in range(0,N_new-1):
            temp = random.randint(0,N-1)
            if (index != temp):
                index = temp
                Data_array[j][:] = Data_train[index][:]
                Data_array_labels[j] = Data_train_labels[index]
        #print Data_array
        #print Data_array_labels
        model.fit(Data_array,Data_array_labels)

#-----------------------------------------------------------------On Test Data----------------------------------------

        expected = Data_test_labels
        predicted = model.predict(Data_test)
        #print(metrics.classification_report(expected,predicted))
        confusion_matrix = metrics.confusion_matrix(expected,predicted)
        #print(confusion_matrix)
        miss1 = (float(confusion_matrix[0][0])*100/(confusion_matrix[0][0] + confusion_matrix[0][1]))
        miss2 = (float(confusion_matrix[1][1])*100/(confusion_matrix[1][0] + confusion_matrix[1][1]))
        miss = (float(confusion_matrix[0][0]+confusion_matrix[1][1])*100/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]))
        #print N_new,miss1,miss2
        Data_out1[i][0] = float(N_new*100/N)
        Data_out1[i][1] = miss1
        Data_out1[i][2] = miss2
        Data_out1[i][3] = miss
#--------------------------------------------------------------On Train Data--------------------------------------------

        expected = Data_train_labels
        predicted = model.predict(Data_train)
        #print(metrics.classification_report(expected,predicted))
        confusion_matrix = metrics.confusion_matrix(expected,predicted)
        #print(confusion_matrix)
        miss1 = (float(confusion_matrix[0][0])*100/(confusion_matrix[0][0] + confusion_matrix[0][1]))
        miss2 = (float(confusion_matrix[1][1])*100/(confusion_matrix[1][0] + confusion_matrix[1][1]))
        miss = (float(confusion_matrix[0][0]+confusion_matrix[1][1])*100/(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]))
        #print N_new,miss1,miss2
        Data_out2[i][0] = float(N_new*100/N)
        Data_out2[i][1] = miss1
        Data_out2[i][2] = miss2
        Data_out2[i][3] = miss

#---------------------------------------------------------------

        #print Data_out[i][0]
        #plt.scatter(Data_out[i][0],Data_out[i][1])
        #export_graphviz(model,out_file='tree%s.dot',% i)
    #Data_out.to_csv('Data_out.csv', index=False, header=False)
#print Data_out1[:,3]
#print Data_out2[:,3]
plt.plot(Data_out1[:,0], Data_out1[:,1], 'xb-',label='Test')
plt.plot(Data_out2[:,0], Data_out2[:,1], '.r-',label='Train')
plt.title("Accuracies_class1")
plt.xlabel("% of Test Data")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracies_class1')
plt.show()

plt.plot(Data_out1[:,0], Data_out1[:,2], 'xb-',label='Test')
plt.plot(Data_out2[:,0], Data_out2[:,2], '.r-',label='Train')
plt.title("Accuracies_class2")
plt.xlabel("% of Test Data")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracies_class2')
plt.show()

plt.plot(Data_out1[:,0], Data_out1[:,3], 'xc-',label='Test')
plt.plot(Data_out2[:,0], Data_out2[:,3], '.g-',label='Train')
plt.title("Accuracies_overall")
plt.xlabel("% of Test Data")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('Accuracies_overall')
plt.show()
#print Data_out1[:,0]

