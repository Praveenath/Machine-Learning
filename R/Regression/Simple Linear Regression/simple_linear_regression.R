# Read dataset
dataset = read.csv('Salary_Data.csv')
# dataset = dataset[,2:3]

# Spliting dataset into train and test
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

# Fit Simple Linear Regression
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)
# summary(regressor)

# Predicting test results
y_pred = predict(regressor, newdata = test_set)

# Visualising the training results
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
                colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab("Years of Experience") +
  ylab('Salary')
# Visualising the test results
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training Set)') +
  xlab("Years of Experience") +
  ylab('Salary')