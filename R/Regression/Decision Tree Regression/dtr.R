# Decision Tree Regression

# Read dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fit Decision Tree Regression to the dataset
library(rpart)
regressor = rpart(formula = Salary ~.,
                data = dataset,
                control = rpart.control(minsplit = 1))

# smoothing plot values
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

# Visualze the Decision Tree Regression results
library(ggplot2)
ggplot() +
  geom_point( aes(x = dataset$Level, y = dataset$Salary),
              color = 'red')+
  geom_line( aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
             colour = 'blue')+
  ggtitle('Level vs Salary (Decision Tree Regression)')+
  xlab('Levels')+
  ylab('Salary')

# Prediction : SV Regression
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))