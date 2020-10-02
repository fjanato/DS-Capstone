##############################
## BEGIN SECTION 0          ##
## Environment Setup        ##
##############################
# Install Packages
install.packages('corrplot')
install.packages('psych')

# Load Libraries
library(tidyverse)
library(dbplyr)
library(lubridate)
library(data.table)
library(ggplot2)
library(repr)
library(caret)
library(chron)
library(gridExtra)
library(randomForest)
library(MLmetrics)
library(MASS)
library(ROCR)
library(pROC)
library(modeest)
library(corrplot)
library(psych)

# Experimental Scenario
# 
# Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. 
# Each time series is from a different engine ñ i.e., the data can be considered to be from a fleet of engines of the same type. 
# Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. 
# This wear and variation is considered normal, i.e., it is not considered a fault condition. 
# There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. 
# The data is contaminated with sensor noise.
# 
# The engine is operating normally at the start of each time series, and develops a fault at some point during the series. 
# In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. 
# The objective is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of 
# operational cycles after the last cycle that the engine will continue to operate. 
# Also provided a vector of true Remaining Useful Life (RUL) values for the test data.
#####################################################################################################
# Train Set has all equipment run cycles, the last cycle in this dataset is when failure happened
# Test Set has also equipment run cycles however in this set the failure cycle was not provided
# The remainder cycles to failure are provide in a separate dataset CTF (Cycles to Failure)



# Load Dataset for Study
test <- read_delim("test_FD001.txt", delim = " ", col_names = FALSE)
train <- read_delim("train_FD001.txt", delim = " ", col_names = FALSE)
CTF <- read_csv("RUL_FD001.txt", col_names = FALSE)

# Set header Names for Train set
names(train) <- c('UnitNum','TimeCycles','OperSet1','OperSet2','OperSet3','SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
                  ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
                  ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21','CyclesToFail')

# Set header Names for Test set
names(test) <- c('UnitNum','TimeCycles','OperSet1','OperSet2','OperSet3','SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
                 ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
                 ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21','CyclesToFail')

# Set header Names for RUL number applicable to Test set
names(CTF) <- c('CyclesToFail')

summary(train)



# Find the last Cycle number for each UnitNum (Jet Engine) in the Train set
TrainFailCycle <- as.vector(tapply(train$TimeCycles, train$UnitNum, max))

# Find the last Cycle number for each UnitNum (Jet Engine) in the Test set
TestFailCycle <- as.vector(tapply(test$TimeCycles, test$UnitNum, max))

# Plot Last Cycle per Unit
qplot(y= TrainFailCycle, x=1:length(TrainFailCycle), main = 'Failure Cycle by Engine', ylab = 'Cycle of Failure', xlab = 'Unit Number', geom = "line")


# Show distribution of Failure Cycle in Engines
hist(TrainFailCycle)



# Get RUL values for file 
RealCTF <- unlist(CTF)

# Calculate Cycles to Fails Value for Train Set
train$CyclesToFail <- TrainFailCycle[train$UnitNum] - train$TimeCycles

# Calculate Cycles to Fails Value for Test Set
test$CyclesToFail <-   RealCTF[test$UnitNum] + TestFailCycle[test$UnitNum]

test$CyclesToFail <-  test$CyclesToFail - test$TimeCycles


#Plot RUL per cycle number on Train set
ggplot(train, aes(TimeCycles, CyclesToFail) ) +
  geom_point() +
  stat_smooth() +
  ggtitle("Cycles Runs per Engine vs Cycles to Fail")


#Plot RUL per cycle number on Test set
ggplot(test, aes(TimeCycles, CyclesToFail) ) +
  geom_point() +
  stat_smooth() +
  



# Build the preliminary list of variables for the prediction model 
num_cols = c('OperSet1','OperSet2','SensorRead02','SensorRead03','SensorRead04','SensorRead06','SensorRead07','SensorRead08','SensorRead09',
             'SensorRead11','SensorRead12','SensorRead13','SensorRead14','SensorRead15','SensorRead17','SensorRead20','SensorRead21')


# Calculate Correlations between variables
correlations <- cor(data.frame(train[,num_cols]))

# Visualize correlations
corrplot(correlations)

# many unknows correll, check SD to validate vars w/o



train_sd <-sapply(train[,num_cols], sd)
train_sd


rev_correl <- cor(data.frame(train[,num_cols]))
#correlations

heatmap(rev_correl, symm = TRUE)

corrplot(rev_correl, type = "upper",  order = 'hclust')


# Remove Sen_Read_6 no strong correlation with any other variable but itself, same case as for OperSet 1 and 2
rev_cols = c('SensorRead02','SensorRead03','SensorRead04','SensorRead07','SensorRead08','SensorRead09',
             'SensorRead11','SensorRead12','SensorRead13','SensorRead14','SensorRead15','SensorRead17','SensorRead20','SensorRead21')


rev_correl <- cor(data.frame(train[,rev_cols]))
#correlations

heatmap(rev_correl, symm = TRUE)

corrplot(rev_correl,  order = 'hclust')


#Scale numeric features
preProcValues <- preProcess(train[,rev_cols], method = c("center", "scale"))

train[,rev_cols] = predict(preProcValues, train[,rev_cols])
test[,rev_cols] = predict(preProcValues, test[,rev_cols])
head(train[,rev_cols])



#TrainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter=FALSE)  
TrainFormula <- CyclesToFail ~ SensorRead02 + SensorRead03 + SensorRead04 + SensorRead07 + SensorRead08 + SensorRead09 +
  SensorRead11 + SensorRead12 + SensorRead13 + SensorRead14 + SensorRead15 + SensorRead17 + SensorRead20 + SensorRead21


#########################
#apply linear regression
## define and fit the linear regression model


lin_mod = lm(TrainFormula, data = train)

print_metrics = function(lin_mod, df, score, label){
  resids = df[,label] - df[,score]
  resids2 = resids**2
  N = length(score)
  r2 = as.character(round(summary(lin_mod)$r.squared, 4))
  adj_r2 = as.character(round(summary(lin_mod)$adj.r.squared, 4))
  cat(paste('Mean Square Error      = ', as.character(round(sum(resids2)/N, 4)), '\n'))
  cat(paste('Root Mean Square Error = ', as.character(round(sqrt(sum(resids2)/N), 4)), '\n'))
  cat(paste('Mean Absolute Error    = ', as.character(round(sum(abs(resids))/N, 4)), '\n'))
  cat(paste('Median Absolute Error  = ', as.character(round(median(abs(unlist(resids))), 4)), '\n'))
  cat(paste('R^2                    = ', r2, '\n'))
  cat(paste('Adjusted R^2           = ', adj_r2, '\n'))
}

#score = predict(lin_mod, newdata = test)
test$LR_score <- predict(lin_mod, newdata = test)


# Model performance
predictions <- lin_mod %>% predict(test)
# data.frame("Linear Regression Model",
#   RMSE = RMSE(predictions, test$CyclesToFail),
#   R2 = R2(predictions, test$CyclesToFail)
# )
print_metrics(lin_mod, test, 'LR_score', 'CyclesToFail')  

# Visualize Linear Regression Model

ggplot(test, aes(CyclesToFail, LR_score) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ x)



#############################
## Apply Polynomial Regression


# lm(CyclesToFail ~ polym(SensorRead02, SensorRead03, SensorRead04, SensorRead07, SensorRead08, SensorRead09,
#                         SensorRead11, SensorRead12, SensorRead13, SensorRead14,
#                         SensorRead15, SensorRead17, SensorRead20, SensorRead21, 
#                          degree = 2, raw = TRUE), data = train) %>% summary()


# Build the model
poly_model <- lm(CyclesToFail ~ polym(SensorRead02, SensorRead03, SensorRead04, SensorRead07, SensorRead08, SensorRead09,
                                      SensorRead11, SensorRead12, SensorRead13, SensorRead14,
                                      SensorRead15, SensorRead17, SensorRead20, SensorRead21, 
                                      degree = 2, raw = TRUE), data = train)


# Make predictions
test$poly_score <- predict(poly_model, newdata = test)
predictions <- poly_model %>% predict(test)
# Model performance
print_metrics(poly_model, test, 'poly_score', 'CyclesToFail')  

# Plot Model Accuracy
ggplot(test, aes(CyclesToFail, poly_score) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x, 5, raw = TRUE))


######################################
#apply Random Forest
rf_model = randomForest(TrainFormula, data = train, ntree = 5)

test$RFscore = predict(rf_model, newdata = test)

# Model performance
print_metrics(rf_model, test, 'RFscore', 'CyclesToFail')  

predictions <- rf_model %>% predict(test)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test$CyclesToFail),
  R2 = 1-R2(predictions, test$CyclesToFail)
)

# Plot Model Accuracy
ggplot(test, aes(CyclesToFail, RFscore) ) +
  geom_point() +
  stat_smooth(method = lm, formula = y ~ poly(x, 5, raw = TRUE))


###############3





#Print confussion Matrix to review performance
print_metrics = function(df, label){
  ## Compute and print the confusion matrix
  cm = as.matrix(table(Actual = test$CyclesToFail, Predicted = test$RFscore))
  print(cm)
  
  ## Compute and print accuracy
  accuracy = round(sum(sapply(1:nrow(cm), function(i) cm[i,i]))/sum(cm), 3)
  cat("\n")
  cat(paste("Accuracy = ", as.character(accuracy)), "\n \n")

  # Compute and print precision, recall and F1
  precision = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[i,]))
  recall = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[,i]))
  F1 = sapply(1:nrow(cm), function(i) 2*(recall[i] * precision[i])/(recall[i] + precision[i]))

  metrics = sapply(c(precision, recall, F1), round, 3)
  metrics = t(matrix(metrics, nrow = nrow(cm), ncol = ncol(cm)))
  metrics = t(matrix(metrics, nrow = nrow(cm), ncol = 3))
  dimnames(metrics) = list(c("Precision", "Recall", "F1"), unique(test$CyclesToFail))
  print(metrics)
}  
print_metrics(test, "CyclesToFail")






options(repr.plot.width=4, repr.plot.height=3)
imp = varImp(rf_model)
imp = varImp(lin_mod)
imp = varImp(poly_model)



imp[,'Feature'] = row.names(imp)
ggplot(imp, aes(x = Feature, y = Overall)) + geom_point(size = 4) +
  ggtitle('Variable importance for features') +
  theme(axis.text.x = element_text(angle = 45))

options(repr.plot.width=8, repr.plot.height=6)
print(imp)



hist_resids = function(df, score, label, bins = 10){
  options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
  df$resids = df[,label] - score
  bw = (max(df$resids) - min(df$resids))/(bins + 1)
  ggplot(df, aes(resids)) + 
    geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
    geom_density(aes(y=..density..), color = 'blue') +
    xlab('Residual value') + ggtitle('Histogram of residuals')
}

hist_resids(test, 'LR_score', 'CyclesToFail') 
######################


