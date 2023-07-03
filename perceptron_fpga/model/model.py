import sklearn
import sklearn.neural_network
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

DECIMALS = 32


digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

clf = sklearn.neural_network.MLPRegressor(alpha=0, verbose=True)
# target = digits.target # esto para que haga regresion contra el valor
target = OneHotEncoder(sparse=False).fit_transform(
    digits.target.reshape(-1, 1)
)  # para que haga regresion contra

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.8, shuffle=False
)

clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print(f"{clf.out_activation_=}")
print(f"{clf.intercepts_=}")
print("predicho:")
print(predicted)
print("real:")
print(y_test)
print("error absoluto promedio: " + str(mean_absolute_error(y_test, predicted)))
coefs = clf.coefs_

coefs = [coef for layer in coefs for row in layer for coef in row]

output = "coefs.txt"
with open(output, "w") as f:
    f.write('\n'.join(str(x) for x in coefs))
print("Coeficientes escritos en " + output)
