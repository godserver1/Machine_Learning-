#---------------Question 2 -----------------

import numpy as numpy
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plot
from scipy.stats import multivariate_normal

#---------Data aquisition-------------------

Matrix_train = numpy.loadtxt(open("P2_train.csv","r"),dtype = 'float',delimiter=',')

N = 0
D = 0

(N,D) = Matrix_train.shape

Array_labels = Matrix_train[:,D-1]
Matrix_train = numpy.delete(Matrix_train,D-1,axis=1)
D = D-1

#-------------Parameter Extraction------------------

Mean = numpy.zeros(D)
Mean_0 = numpy.zeros(D)
Mean_1 = numpy.zeros(D)
N_0 = 0

for i in range(0,N):
	Mean = Mean + Matrix_train[i,:]
	if(Array_labels[i] == 0):
		Mean_0 = Mean_0 + Matrix_train[i,:]
		N_0 = N_0 + 1
	else:
		Mean_1 = Mean_1 + Matrix_train[i,:]

N_1 = N - N_0

Mean = numpy.divide(Mean,N)
Mean_0 = numpy.divide(Mean_0,N_0)
Mean_1 = numpy.divide(Mean_1,N_1)

Cov = numpy.zeros((D,D))
Cov_0 = numpy.zeros((D,D))
Cov_1 = numpy.zeros((D,D))

for i in range(0,N):
	MultArray = Matrix_train[i,:] - Mean
	Cov = Cov + numpy.outer(MultArray,MultArray)
	if(Array_labels[i] == 0):
		MultArray = Matrix_train[i,:] - Mean_0
		Cov_0 = Cov_0 + numpy.outer(MultArray,MultArray)
	else:
		MultArray = Matrix_train[i,:] - Mean_1
		Cov_1 = Cov_1 + numpy.outer(MultArray,MultArray)

Cov = numpy.divide(Cov,N-1)
Cov_0 = numpy.divide(Cov_0,N_0-1)
Cov_1 = numpy.divide(Cov_1,N_1-1)

#print Cov
print Cov_0
print Cov_1

P_0 = float(N_0)/N
P_1 = float(N_1)/N

#---------------Testing Algorithm on test data------------------------

Matrix_test = numpy.loadtxt(open("P2_test.csv","r"),delimiter=',')

M = 0
D_t = 0

(M,D_t) = Matrix_test.shape

Array_labels_test = Matrix_test[:,D_t-1]
Matrix_test = numpy.delete(Matrix_test,D_t-1,axis=1)
Matrix_test_trans = Matrix_test.T
D_t = D_t - 1


#-----------Different Covariance Matrix For Both Classes--------------------

Cov_0_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_0))))
Cov_1_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_1))))

Cov_0_inverse = numpy.linalg.inv(Cov_0)
Cov_1_inverse = numpy.linalg.inv(Cov_1)

Prediction_matrix = [0 for x in range(0,M)]

G_0 = numpy.zeros(M)
G_1 = numpy.zeros(M)

for i in range(0,M):
	Temp_array1 = Matrix_test[i,:] - Mean_0
	Temp_array2 = numpy.matmul(Temp_array1,Cov_0_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_0[i] = numpy.log(P_0) - Cov_0_val - 0.5*Temp_val3

	Temp_array1 = Matrix_test[i,:] - Mean_1
	Temp_array2 = numpy.matmul(Temp_array1,Cov_1_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_1[i] = numpy.log(P_1) - Cov_1_val - 0.5*Temp_val3

	if (G_0[i]>=G_1[i]):
		Prediction_matrix[i] = 0
	else:
		Prediction_matrix[i] = 1

N_0_test = 0
N_1_test = 0

confusion_matrix = numpy.zeros((2,2))

for i in range(0,M):
	if(Array_labels_test[i] == 0):
		N_0_test = N_0_test + 1
		if(Prediction_matrix[i] == 0):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_1_test = N_1_test + 1
		if(Prediction_matrix[i] == 1):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_0_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_1_test - confusion_matrix[1][1]

misclassification_rate_0 = (confusion_matrix[0][1]*100)/N_0_test
misclassification_rate_1 = (confusion_matrix[1][0]*100)/N_1_test

print misclassification_rate_0
print misclassification_rate_1

print "The 2x2 Confusion Matrix for the case when Both the classes have Independent covariance matrices is : "
print confusion_matrix

Matrix_0 = numpy.zeros((2, N_0_test))
Matrix_1 = numpy.zeros((2, N_1_test))
p = 0
q = 0

for i in range(0, M):
	if (Array_labels_test[i] == 0):
		Matrix_0[:,p] = Matrix_test_trans[:, i]
		p = p + 1
	else:
		Matrix_1[:,q] = Matrix_test_trans[:, i]
		q = q + 1

plot.scatter(Matrix_0[0,:], Matrix_0[1, :], c='r', marker='s')
plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='b', marker='^')

x = numpy.arange(-6.75, 5.7	5, 0.025)
y = numpy.arange(-12.75, 7.75, 0.025)
X, Y = numpy.meshgrid(x, y)
pos = numpy.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
rv0 = multivariate_normal(Mean_0, Cov_0)
rv1 = multivariate_normal(Mean_1, Cov_1)
#plt.figure()
CS0 = plot.contour(X, Y, rv0.pdf(pos))
CS1 = plot.contour(X, Y, rv1.pdf(pos), colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
plot.clabel(CS0, inline=1, fontsize=10)
plot.clabel(CS1, inline=1, fontsize=10)

plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='g', marker='^')
plot.title("Scatter Plot of class 1, (label : 0)")
plot.xlabel("Attribute 0")
plot.ylabel("Attribute 1")
plot.savefig('Different_cov')
plot.show()


#-----------Equal diagonal covariance matrix of equal variance along both dimensions----------

Mat_temp = numpy.zeros((D,D))
numpy.fill_diagonal(Mat_temp,Cov[0][0])
Cov_0 = Cov_1 = Mat_temp

Cov_0_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_0))))
Cov_1_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_1))))

Cov_0_inverse = numpy.linalg.inv(Cov_0)
Cov_1_inverse = numpy.linalg.inv(Cov_1)

for i in range(0,M):
	Temp_array1 = Matrix_test[i,:] - Mean_0
	Temp_array2 = numpy.matmul(Temp_array1,Cov_0_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_0[i] = numpy.log(P_0) - Cov_0_val - 0.5*Temp_val3

	Temp_array1 = Matrix_test[i,:] - Mean_1
	Temp_array2 = numpy.matmul(Temp_array1,Cov_1_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_1[i] = numpy.log(P_1) - Cov_1_val - 0.5*Temp_val3

	if (G_0[i]>=G_1[i]):
		Prediction_matrix[i] = 0
	else:
		Prediction_matrix[i] = 1

N_0_test = 0
N_1_test = 0

confusion_matrix[0][0] = 0
confusion_matrix[1][1] = 0

for i in range(0,M):
	if(Array_labels_test[i] == 0):
		N_0_test = N_0_test + 1
		if(Prediction_matrix[i] == 0):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_1_test = N_1_test + 1
		if(Prediction_matrix[i] == 1):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_0_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_1_test - confusion_matrix[1][1]

misclassification_rate_0 = (confusion_matrix[0][1]*100)/N_0_test
misclassification_rate_1 = (confusion_matrix[1][0]*100)/N_1_test

print misclassification_rate_0
print misclassification_rate_1

print "The 2x2 Confusion Matrix for the case when Both the classes have Equal diagonal Covariance matrices of equal variation is : "
print confusion_matrix

Matrix_0 = numpy.zeros((2, N_0_test))
Matrix_1 = numpy.zeros((2, N_1_test))
p = 0
q = 0

for i in range(0, M):
	if (Array_labels_test[i] == 0):
		Matrix_0[:,p] = Matrix_test_trans[:, i]
		p = p + 1
	else:
		Matrix_1[:,q] = Matrix_test_trans[:, i]
		q = q + 1

plot.scatter(Matrix_0[0,:], Matrix_0[1, :], c='r', marker='s')
plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='b', marker='^')

x = numpy.arange(-6.75, 5.75, 0.025)
y = numpy.arange(-12.75, 7.75, 0.025)
X, Y = numpy.meshgrid(x, y)
pos = numpy.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
rv0 = multivariate_normal(Mean_0, Cov_0)
rv1 = multivariate_normal(Mean_1, Cov_1)
#plt.figure()
CS0 = plot.contour(X, Y, rv0.pdf(pos))
CS1 = plot.contour(X, Y, rv1.pdf(pos), colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
plot.clabel(CS0, inline=1, fontsize=10)
plot.clabel(CS1, inline=1, fontsize=10)

plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='g', marker='^')
plot.title("Scatter Plot of class 1, (label : 0)")
plot.xlabel("Attribute 0")
plot.ylabel("Attribute 1")
plot.savefig('equal_diagonal_and_varaince')
plot.show()

#-----------Equal diagonal Covaiance of unequal variance along different dimensions----------

Mat_temp = numpy.zeros((D,D))
numpy.fill_diagonal(Mat_temp,numpy.diag(Cov))
Cov_0 = Cov_1 = Mat_temp

Cov_0_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_0))))
Cov_1_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_1))))

Cov_0_inverse = numpy.linalg.inv(Cov_0)
Cov_1_inverse = numpy.linalg.inv(Cov_1)

for i in range(0,M):
	Temp_array1 = Matrix_test[i,:] - Mean_0
	Temp_array2 = numpy.matmul(Temp_array1,Cov_0_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_0[i] = numpy.log(P_0) - Cov_0_val - 0.5*Temp_val3

	Temp_array1 = Matrix_test[i,:] - Mean_1
	Temp_array2 = numpy.matmul(Temp_array1,Cov_1_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_1[i] = numpy.log(P_1) - Cov_1_val - 0.5*Temp_val3

	if (G_0[i]>=G_1[i]):
		Prediction_matrix[i] = 0
	else:
		Prediction_matrix[i] = 1

N_0_test = 0
N_1_test = 0

confusion_matrix[0][0] = 0
confusion_matrix[1][1] = 0

for i in range(0,M):
	if(Array_labels_test[i] == 0):
		N_0_test = N_0_test + 1
		if(Prediction_matrix[i] == 0):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_1_test = N_1_test + 1
		if(Prediction_matrix[i] == 1):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_0_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_1_test - confusion_matrix[1][1]

misclassification_rate_0 = (confusion_matrix[0][1]*100)/N_0_test
misclassification_rate_1 = (confusion_matrix[1][0]*100)/N_1_test

print misclassification_rate_0
print misclassification_rate_1

print "The 2x2 Confusion Matrix for the case when Both the classes have Equal diagonal Covariance matrices of unequal variation is : "
print confusion_matrix

Matrix_0 = numpy.zeros((2, N_0_test))
Matrix_1 = numpy.zeros((2, N_1_test))
p = 0
q = 0

for i in range(0, M):
	if (Array_labels_test[i] == 0):
		Matrix_0[:,p] = Matrix_test_trans[:, i]
		p = p + 1
	else:
		Matrix_1[:,q] = Matrix_test_trans[:, i]
		q = q + 1

plot.scatter(Matrix_0[0,:], Matrix_0[1, :], c='r', marker='s')
plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='b', marker='^')

x = numpy.arange(-6.75, 5.75, 0.025)
y = numpy.arange(-12.75, 7.75, 0.025)
X, Y = numpy.meshgrid(x, y)
pos = numpy.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
rv0 = multivariate_normal(Mean_0, Cov_0)
rv1 = multivariate_normal(Mean_1, Cov_1)
#plt.figure()
CS0 = plot.contour(X, Y, rv0.pdf(pos))
CS1 = plot.contour(X, Y, rv1.pdf(pos), colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
plot.clabel(CS0, inline=1, fontsize=10)
plot.clabel(CS1, inline=1, fontsize=10)

plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='g', marker='^')
plot.title("Scatter Plot of class 1, (label : 0)")
plot.xlabel("Attribute 0")
plot.ylabel("Attribute 1")
plot.savefig('equal_diagonal_unequal_varaince')
plot.show()


#-----------Arbitrary covariances shared by both classes----------

Cov_0 = Cov_1 = Cov

Cov_0_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_0))))
Cov_1_val = 0.5*(numpy.log(numpy.absolute(numpy.linalg.det(Cov_1))))

Cov_0_inverse = numpy.linalg.inv(Cov_0)
Cov_1_inverse = numpy.linalg.inv(Cov_1)

for i in range(0,M):
	Temp_array1 = Matrix_test[i,:] - Mean_0
	Temp_array2 = numpy.matmul(Temp_array1,Cov_0_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_0[i] = numpy.log(P_0) - Cov_0_val - 0.5*Temp_val3

	Temp_array1 = Matrix_test[i,:] - Mean_1
	Temp_array2 = numpy.matmul(Temp_array1,Cov_1_inverse)
	Temp_val3 = numpy.dot(Temp_array2,Temp_array1)
	G_1[i] = numpy.log(P_1) - Cov_1_val - 0.5*Temp_val3

	if (G_0[i]>=G_1[i]):
		Prediction_matrix[i] = 0
	else:
		Prediction_matrix[i] = 1

N_0_test = 0
N_1_test = 0

confusion_matrix[0][0] = 0
confusion_matrix[1][1] = 0

for i in range(0,M):
	if(Array_labels_test[i] == 0):
		N_0_test = N_0_test + 1
		if(Prediction_matrix[i] == 0):
			confusion_matrix[0][0] = confusion_matrix[0][0] + 1
	else:
		N_1_test = N_1_test + 1
		if(Prediction_matrix[i] == 1):
			confusion_matrix[1][1] = confusion_matrix[1][1] + 1

confusion_matrix[0][1] = N_0_test - confusion_matrix[0][0]
confusion_matrix[1][0] = N_1_test - confusion_matrix[1][1]

misclassification_rate_0 = (confusion_matrix[0][1]*100)/N_0_test
misclassification_rate_1 = (confusion_matrix[1][0]*100)/N_1_test

print misclassification_rate_0
print misclassification_rate_1

print "The 2x2 Confusion Matrix for the case when Both the classes have Arbitrary covariances shared by both classes is : "
print confusion_matrix

Matrix_0 = numpy.zeros((2, N_0_test))
Matrix_1 = numpy.zeros((2, N_1_test))
p = 0
q = 0

for i in range(0, M):
	if (Array_labels_test[i] == 0):
		Matrix_0[:,p] = Matrix_test_trans[:, i]
		p = p + 1
	else:
		Matrix_1[:,q] = Matrix_test_trans[:, i]
		q = q + 1

plot.scatter(Matrix_0[0,:], Matrix_0[1, :], c='r', marker='s')
plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='b', marker='^')

x = numpy.arange(-6.75, 5.75, 0.025)
y = numpy.arange(-12.75, 7.75, 0.025)
X, Y = numpy.meshgrid(x, y)
pos = numpy.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
rv0 = multivariate_normal(Mean_0, Cov_0)
rv1 = multivariate_normal(Mean_1, Cov_1)
#plt.figure()
CS0 = plot.contour(X, Y, rv0.pdf(pos))
CS1 = plot.contour(X, Y, rv1.pdf(pos), colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'))
plot.clabel(CS0, inline=1, fontsize=10)
plot.clabel(CS1, inline=1, fontsize=10)

plot.scatter(Matrix_1[0,:], Matrix_1[1, :], c='g', marker='^')
plot.title("Scatter Plot of class 1, (label : 0)")
plot.xlabel("Attribute 0")
plot.ylabel("Attribute 1")
plot.savefig('arbitrary_cov')
plot.show()

