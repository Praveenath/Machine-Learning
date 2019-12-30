# Support Vector Regression

# Read dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fit SVR to the dataset
library(e1071)
regressor = svm(formula = Salary ~.,
                data = dataset,
                type = 'eps-regression')

# smoothing plot values
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

# Visualze the Linear Regression results
library(ggplot2)
ggplot() +
  geom_point( aes(x = dataset$Level, y = dataset$Salary),
              color = 'red')+
  geom_line( aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
             colour = 'blue')+
  ggtitle('Level vs Salary (SVR)')+
  xlab('Levels')+
  ylab('Salary')

# Prediction : SV Regression
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))


