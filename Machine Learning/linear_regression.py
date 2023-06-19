import numpy as np

# def LinearRegression(x, y):
#     n = len(x)
#     sum_x = sum(x)
#     sum_y = sum(y)
#     sum_xy = sum(x[i] * y[i] for i in range(n))
#     sum_xx = sum(x[i] * x[i] for i in range(n))

#     slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
#     intercept = (sum_y - slope * sum_x) / n

#     return slope, intercept


def LinearRegression(x, y):

    n = len(x)
    mean_x = sum(x)/n
    mean_y = sum(y)/n

    ssx = sum((x[i] - mean_x)**2 for i in range(n))
    ssy = sum((y[i] - mean_y)**2 for i in range(n))

    sp = sum((x[i] - mean_x)*(y[i] - mean_y) for i in range(n))

    b = sp/ssx
    a = mean_y - (b*mean_x)

    return b, a


# Example data (2x + 31)
x = [1, 2, 3, 4, 5]
y = [33, 35, 37, 39, 41]

# Fit the model
slope, intercept = LinearRegression(x, y)
print(slope, intercept)

b, a = LinearRegression(x, y)
print(b, a)

# Predict new values
x_unknown = 7
y_pred = b * x_unknown + a

print("Predicted values:", y_pred)
