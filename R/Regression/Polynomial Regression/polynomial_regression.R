# Polynomial Regression

# Read dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Spliting dataset into train and test
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)


# Feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fit Linear Regression to the dataset
linear_reg = lm(formula = Salary ~.,
                data = dataset)

# Fit Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~.,
              data = dataset)

# Visualze the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point( aes(x = dataset$Level, y = dataset$Salary),
              color = 'red')+
  geom_line( aes(x = dataset$Level, y = predict(linear_reg, newdata = dataset)),
              colour = 'blue')+
  ggtitle('Level vs Salary (Linear Regression)')+
  xlab('Levels')+
  ylab('Salary')

# Visualize the Polynomial Regression results

# smoothing plot values
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() +
  geom_point( aes(x = dataset$Level, y = dataset$Salary),
              color = 'red')+
  geom_line( aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
             colour = 'blue')+
  ggtitle('Level vs Salary (Polynomial Regression)')+
  xlab('Levels')+
  ylab('Salary')

# Prediction : Linear Regression
y_pred = predict(linear_reg, newdata = data.frame(Level = 6.5))

# Prediction : Polynomial Regression
y_pred = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))

