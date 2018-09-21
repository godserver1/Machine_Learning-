#------------------------Question 3--------------------------------------

import numpy as numpy
import matplotlib.pyplot as plot

Matrix_wage_dataset = numpy.loadtxt(open("Wage_dataset.csv", "r"), delimiter=',')	# Data matrix of 3000 x 9 dimensions

print "Give order and attribute :"
order = input()
attribute = input()


R = numpy.zeros(order)
A = numpy.zeros((order, order))
W = numpy.zeros(order)


N = 0
d = 0

(N, d) = Matrix_wage_dataset.shape

def Polynomial_regression (power, c):
	count = 0
	threshold = 10000000000					#initially taken to be very high
	for i in range(0, power):
		val = 0
		for k in range(0, N):
			val = val + (Matrix_wage_dataset[k][d-1]**1)*(Matrix_wage_dataset[k][c]**i)
			count = count + 1
		R[i] = val
		for j in range(0, power):
			val = 0
			for k in range(0, N):
				val = val + (Matrix_wage_dataset[k][d-1]**0)*(Matrix_wage_dataset[k][c]**(i+j))
				count = count - 1
			A[i][j] = val

	for i in range(0, power):
		R_temp = R
		R_temp = numpy.resize(R_temp,i+1)

		A_temp = numpy.zeros((i+1, i+1))
		for j in range(0, i+1):
			for k in range(0, i+1):
				A_temp[j][k] = A[j][k]

		A_temp_inverse = numpy.linalg.inv(A_temp)
		W = numpy.matmul(A_temp_inverse, R_temp)

		error = 0.0
		for k in range(0, N):
			Mean = 0.0
			count = count + 1
			for j in range(0, i+1):
				Mean = Mean + (W[j]*(Matrix_wage_dataset[k][c]**j))
			error = error + ((Matrix_wage_dataset[k][d-1] - Mean)**2)
	#print count
		if (threshold > error):
			threshold = error
		else:
			return (i)
	return power

Best_order = Polynomial_regression(order, attribute)	# Polynomail regression

print ('Best Order for Polynomial Regression is : %s' %Best_order)

x = numpy.arange(2000, 2010, 1.0)
y = numpy.arange(0, 500, 10.0)
X, Y = numpy.meshgrid(x, y)

plot.scatter(Matrix_wage_dataset[0,:], Matrix_wage_dataset[11, :], c='g', marker='^')
plot.title("Scatter Plot of Wage Vs Year")
plot.xlabel("Year")
plot.ylabel("Wage")
plot.savefig('Wage_Vs_Year')
plot.show()

plot.scatter(Matrix_wage_dataset[1,:], Matrix_wage_dataset[11, :], c='g', marker='^')
plot.title("Scatter Plot of Wage Vs Age")
plot.xlabel("Age")
plot.ylabel("Wage")
plot.savefig('Wage_Vs_Age')
plot.show()

plot.scatter(Matrix_wage_dataset[5,:], Matrix_wage_dataset[11, :], c='g', marker='^')
plot.title("Scatter Plot of Wage Vs Education")
plot.xlabel("Education")
plot.ylabel("Wage")
plot.savefig('Wage_Vs_Education')
plot.show()
