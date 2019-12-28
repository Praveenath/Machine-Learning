# Import dataset
dataset = read.csv("50_Startups.csv")

# Encode categorical data
dataset$State = factor(dataset$State,
                       levels = c('New York','California', 'Florida'),
                       labels = c(1,2,3))

# Split the data into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fit Multiple Linear Regression to the training data
# regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
#               data = training_set)
regressor = lm(formula = Profit ~ ., 
               data = training_set)
summary(regressor)

# Prediction on test data
y_pred = predict(regressor, newdata = test_set)

# Bulid optimal model using Backward Elimination
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset)
summary(regressor)
