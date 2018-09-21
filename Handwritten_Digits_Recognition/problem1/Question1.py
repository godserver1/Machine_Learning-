#-----------Question1-------------------

import numpy as numpy

#-----------Data aquisition--------------

Matrix_data_train = numpy.loadtxt(open("P1_data_train.csv","r"), dtype='float',delimiter=',')
Array_labels_train = numpy.loadtxt(open("P1_labels_train.csv","r"), dtype='int',delimiter=',')


N = 0	#no of samples
D = 0	#dimension

(N,D) = Matrix_data_train.shape

#-----------parameters extraction------------

N_5 = 0					#No of 5's in training data

Mean = numpy.zeros(D)
Mean_5 = numpy.zeros(D)
Mean_6 = numpy.zeros(D)

for i in range(0,N):
	Mean = Mean + Matrix_data_train[i,:]
	if(Array_labels_train[i] == 5):
		N_5 = N_5 + 1
		Mean_5 = Mean_5 + Matrix_data_train[i,:]
	else:
		Mean_6 = Mean_6 + Matrix_data_train[i,:]

N_6 = N - N_5			#No of 6's in training data

Mean = numpy.divide(Mean,N)
Mean_5 = numpy.divide(Mean_5,N_5)				#Estimated mean of class with label as 5
Mean_6 = numpy.divide(Mean_6,N_6)				#Estimated mean of class with label as 6

Cov = numpy.zeros((D,D))
Cov_5 = numpy.zeros((D,D))
Cov_6 = numpy.zeros((D,D))

for i in range(0,N):
	MultArray = Matrix_data_train[i,:] - Mean
	Cov = Cov + numpy.outer(MultArray,MultArray)
	if(Array_labels_train[i] == 5):
		MultArray = Matrix_data_train[i,:] - Mean_5
		Cov_5 = Cov_5 + numpy.outer(MultArray,MultArray)
	else:
		MultArray = Matrix_data_train[i,:] - Mean_6
		Cov_6 = Cov_6 + numpy.outer(MultArray,MultArray)

Cov = numpy.divide(Cov,N-1)
Cov_5 = numpy.divide(Cov_5,N_5-1)				#Covariance matrix for label 5
Cov_6 = numpy.divide(Cov_6,N_6-1)				#Covariance matrix for label 6

#print(Mean)
#print(Mean_5)
#print(Mean_6)
#print(Cov)
#print(Cov_5)
#print(Cov_6)

P_5 = float(N_5)/N 								#Apriori probability estimate of 5
P_6 = float(N_6)/N 								#Apriori probability estimate of 6


#------------------Testing algorithm on Test data set-------------------------

Matrix_data_test = numpy.loadtxt(open("P1_data_test.csv","r"),dtype='float',delimiter=',')
Array_labels_test = numpy.loadtxt(open("P1_labels_test.csv","r"),dtype='int',delimiter=',')


(M,D_test) = Matrix_data_test.shape

#----------------Using Covariance matrix obtained from train data-----------------------

Cov_5_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_5))))
Cov_6_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_6))))

Cov_5_inverse = numpy.linalg.inv(Cov_5)
Cov_6_inverse = numpy.linalg.inv(Cov_6)

Prediction_matrix = [0 for x in range(0,M)]

for i in range(0,M):
	Temp_array1 = Matrix_data_test[i,:] - Mean_5
	Temp_array2 = numpy.matmul(Temp_array1,Cov_5_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_5 = numpy.log(P_5) - Cov_5_val - 0.5*Temp_val3

	Temp_array1 = Matrix_data_test[i,:] - Mean_6
	Temp_array2 = numpy.matmul(Temp_array1,Cov_6_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_6 = numpy.log(P_6) - Cov_6_val - 0.5*Temp_val3

	if (G_5>=G_6):
		Prediction_matrix[i] = 5
	else:
		Prediction_matrix[i] = 6

N_5_test = 0
N_6_test = 0

confusion_matrix = numpy.zeros((2,2))

for i in range(0,M):
	if(Array_labels_test[i] == 5):
		N_5_test = N_5_test + 1
		if(Prediction_matrix[i] == 5):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_6_test = N_6_test + 1
		if(Prediction_matrix[i] == 6):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_5_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_6_test - confusion_matrix[1][1]

misclassification_rate_5 = (confusion_matrix[0][1]*100)/N_5_test
misclassification_rate_6 = (confusion_matrix[1][0]*100)/N_6_test

print "Using Covariance matrix obtained from train data"

#print Prediction_matrix
print "The 2x2 Confusion Matrix for the case when Both the classes have Independent covariance matrices is : "
print confusion_matrix
print "misclassification rates"
print misclassification_rate_5
print misclassification_rate_6


#------------------Using Same Covariance Matrix for Both the class weighted sum----------------------------		

Cov_5 = Cov_6 = P_5*Cov_5 + P_6*Cov_6

Cov_5_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_5))))
Cov_6_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_6))))

Cov_5_inverse = numpy.linalg.inv(Cov_5)
Cov_6_inverse = numpy.linalg.inv(Cov_6)

for i in range(0,M):
	Temp_array1 = Matrix_data_test[i,:] - Mean_5
	Temp_array2 = numpy.matmul(Temp_array1,Cov_5_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_5 = numpy.log(P_5) - Cov_5_val - 0.5*Temp_val3

	Temp_array1 = Matrix_data_test[i,:] - Mean_6
	Temp_array2 = numpy.matmul(Temp_array1,Cov_6_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_6 = numpy.log(P_6) - Cov_6_val - 0.5*Temp_val3

	if (G_5>=G_6):
		Prediction_matrix[i] = 5
	else:
		Prediction_matrix[i] = 6
N_5_test=0
N_6_test=0

confusion_matrix[0][0]=0
confusion_matrix[1][1]=0

for i in range(0,M):
	if(Array_labels_test[i] == 5):
		N_5_test = N_5_test + 1
		if(Prediction_matrix[i] == 5):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_6_test = N_6_test + 1
		if(Prediction_matrix[i] == 6):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_5_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_6_test - confusion_matrix[1][1]

misclassification_rate_5 = (confusion_matrix[0][1]*100)/N_5_test
misclassification_rate_6 = (confusion_matrix[1][0]*100)/N_6_test

print "Using Same Covariance Matrix for Both the class weighted sum"

#print Prediction_matrix
print "The 2x2 Confusion Matrix for the case when Both the classes have Same covariance matrices weighted sum is : "
print confusion_matrix
print "Missclassification rates"
print misclassification_rate_5
print misclassification_rate_6


#------------------Using Same Covariance Matrix for Both the class equal to overall covariance----------------------------		

Cov_5 = Cov_6 = Cov

Cov_5_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_5))))
Cov_6_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_6))))

Cov_5_inverse = numpy.linalg.inv(Cov_5)
Cov_6_inverse = numpy.linalg.inv(Cov_6)

for i in range(0,M):
	Temp_array1 = Matrix_data_test[i,:] - Mean_5
	Temp_array2 = numpy.matmul(Temp_array1,Cov_5_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_5 = numpy.log(P_5) - Cov_5_val - 0.5*Temp_val3

	Temp_array1 = Matrix_data_test[i,:] - Mean_6
	Temp_array2 = numpy.matmul(Temp_array1,Cov_6_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_6 = numpy.log(P_6) - Cov_6_val - 0.5*Temp_val3

	if (G_5>=G_6):
		Prediction_matrix[i] = 5
	else:
		Prediction_matrix[i] = 6
N_5_test=0
N_6_test=0

confusion_matrix[0][0]=0
confusion_matrix[1][1]=0

for i in range(0,M):
	if(Array_labels_test[i] == 5):
		N_5_test = N_5_test + 1
		if(Prediction_matrix[i] == 5):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_6_test = N_6_test + 1
		if(Prediction_matrix[i] == 6):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_5_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_6_test - confusion_matrix[1][1]

misclassification_rate_5 = (confusion_matrix[0][1]*100)/N_5_test
misclassification_rate_6 = (confusion_matrix[1][0]*100)/N_6_test

print "Using Same Covariance Matrix for Both the class equal to overall covariance"
#print Prediction_matrix

print "The 2x2 Confusion Matrix for the case when Both the classes have Same covariance matrices equal to overall covariance is : "
print confusion_matrix
print "Missclassification rates"
print misclassification_rate_5
print misclassification_rate_6


#------------------Using Identity as Covariance Matrix for Both class---------------------------

Cov = numpy.eye(D)
Cov_5 = Cov_6 = Cov

Cov_5_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_5))))
Cov_6_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_6))))

Cov_5_inverse = numpy.linalg.inv(Cov_5)
Cov_6_inverse = numpy.linalg.inv(Cov_6)

for i in range(0,M):
	Temp_array1 = Matrix_data_test[i,:] - Mean_5
	Temp_array2 = numpy.matmul(Temp_array1,Cov_5_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_5 = numpy.log(P_5) - Cov_5_val - 0.5*Temp_val3

	Temp_array1 = Matrix_data_test[i,:] - Mean_6
	Temp_array2 = numpy.matmul(Temp_array1,Cov_6_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_6 = numpy.log(P_6) - Cov_6_val - 0.5*Temp_val3

	if (G_5>=G_6):
		Prediction_matrix[i] = 5
	else:
		Prediction_matrix[i] = 6
N_5_test=0
N_6_test=0

confusion_matrix[0][0]=0
confusion_matrix[1][1]=0

for i in range(0,M):
	if(Array_labels_test[i] == 5):
		N_5_test = N_5_test + 1
		if(Prediction_matrix[i] == 5):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_6_test = N_6_test + 1
		if(Prediction_matrix[i] == 6):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_5_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_6_test - confusion_matrix[1][1]

misclassification_rate_5 = (confusion_matrix[0][1]*100)/N_5_test
misclassification_rate_6 = (confusion_matrix[1][0]*100)/N_6_test

print "Using Identity as Covariance Matrix for Both class"

#print Prediction_matrix
print "The 2x2 Confusion Matrix for the case when Both the classes have Same covariance matrix equal to Identity Matrix is : "
print confusion_matrix

print "Missclassification rates"

print misclassification_rate_5
print misclassification_rate_6

