import numpy as np
grad = np.genfromtxt ('/Users/varunsivashankar/Desktop/Admission_Predict_Ver1.1.csv', delimiter=",")
grad = grad[1:,1:]
print(grad)


Y = grad[:, 7]
Y = Y.reshape(500,1)

m = 500
n = 7


X = np.zeros((m,n+1))
for i in range(0,m):
    X[i,0] = 1
for j in range (1,8):
    for i in range (0,m):
        X[i,j] = grad[i, j-1]

# print(X[1:5,:])

W = np.ones((n+1,1))

Partials = np.matmul(X.transpose(), np.matmul(X,W) - Y)

alpha = 0.001
reps = 10000

for i in range(0, reps):
    W = W - alpha*Partials
    Partials = np.matmul(X.transpose(), (np.matmul(X,W) - Y))

# print (W)
# print (W.shape)

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

features = ["GRE Score", "TOEFL Score", "University Rating", "Statement of Purpose",
 				"Letter of Recommendation", "Cumulative GPA", "Research"]
# userInput = however we're getting user input
improve = []

for i in range(0:7):
	improve.append(W[i] * (x_max[i] - userInput[i]))

index_max = np.argmax(improve)

print("Area to improve on:" + features[inde_max])
