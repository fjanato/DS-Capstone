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

summary(train)
# Data preparation
# Create a new dataset to get the Failure Cycle from the Train Data
# Train_fail_cylce <- train %>% group_by(Unit_Num) %>% summarize(failcycle = max(TimeCycles))
# Train_fail_cylce$status <- 'fail'
# plot(Train_fail_cylce)
# hist(Train_fail_cylce$failcycle)

names(train) <- c('UnitNum','TimeCycles','OperSet1','OperSet2','OperSet3','SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
                  ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
                  ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21','CyclesToFail')

names(test) <- c('UnitNum','TimeCycles','OperSet1','OperSet2','OperSet3','SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
                  ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
                  ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21','CyclesToFail')

names(CTF) <- c('CyclesToFail')

TrainFailCycle <- tapply(train$TimeCycles, train$UnitNum, max)
TestFailCycle <- tapply(test$TimeCycles, test$UnitNum, max)
qplot(y= TrainFailCycle, x=1:length(TrainFailCycle), main = 'Failure Cycle by Engine', ylab = 'Cycle of Failure', xlab = 'Unit Number')


# #create a binomial classification on the training set, fail for the last reading and run for the previous one
# Train_RF <- merge(train, Train_fail_cylce, by=1:2, all.x = TRUE)
# Train_RF$status <- ifelse(is.na(Train_RF$status), 'run', 'fail')
# Train_RF$status <- factor(Train_RF$status, levels = c('fail','run'))

RealCTF <- unlist(CTF)

train$CyclesToFail <- TrainFailCycle[train$UnitNum] - train$TimeCycles

#test$CyclesToFail <- CTF[test$UnitNum] + TestFailCycle[test$UnitNum] - test$TimeCycles

test$CyclesToFail <-   RealCTF[test$UnitNum] + TestFailCycle[test$UnitNum] - test$TimeCycles

# Get a list of all Measurement related columns
sensor_measure_cols = c('SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
                        ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
                        ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21')


operation_setting_cols = c('OperSet1','OperSet2','OperSet3')

num_cols = c('OperSet1','OperSet2','OperSet3','SensorRead01','SensorRead02','SensorRead03','SensorRead04','SensorRead05'
             ,'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead10','SensorRead11','SensorRead12','SensorRead13','SensorRead14'
             ,'SensorRead15','SensorRead16','SensorRead17','SensorRead18','SensorRead19','SensorRead20','SensorRead21')





# Visualize Correlations between variables
correlations <- cor(data.frame(train[,num_cols]))
#correlations

heatmap(correlations, symm = TRUE)

corrplot(correlations)

# many unknows correll, check SD to validate vars w/o



train_sd <-sapply(train[,3:26], sd)
train_sd

# remove variables that have no Standard deviations, no changes in time, mean not affecting the result
rev_cols = c('OperSet1','OperSet2','SensorRead02','SensorRead03','SensorRead04',
             'SensorRead06','SensorRead07','SensorRead08','SensorRead09','SensorRead11','SensorRead12','SensorRead13','SensorRead14',
             'SensorRead15','SensorRead17','SensorRead20','SensorRead21')

rev_correl <- cor(data.frame(train[,rev_cols]))
#correlations

heatmap(rev_correl, symm = TRUE)

corrplot(rev_correl,  order = 'hclust')


# Remove Sen_Read_6 no strong correlation with any other variable but itself, same case as for OperSet 1 and 2
# #rev_cols1 = c('SensorRead02','SensorRead03','SensorRead04',
#              'SensorRead07','SensorRead08','SensorRead09','SensorRead11','SensorRead12','SensorRead13','SensorRead14',
#              'SensorRead15','SensorRead17','SensorRead20','SensorRead21')
rev_cols1 = c('SensorRead02','SensorRead03','SensorRead04',
              'SensorRead07','SensorRead11','SensorRead12',
              'SensorRead15','SensorRead17','SensorRead20','SensorRead21')

rev_correl <- cor(data.frame(train[,rev_cols1]))
#correlations

heatmap(rev_correl, symm = TRUE)

corrplot(rev_correl,  order = 'hclust')


#Scale numeric features
preProcValues <- preProcess(train[,rev_cols1], method = c("center", "scale"))

train[,rev_cols1] = predict(preProcValues, train[,rev_cols1])
test[,rev_cols1] = predict(preProcValues, test[,rev_cols1])
head(train[,rev_cols1])



#TrainControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, verboseIter=FALSE)  
TrainFormula <- CyclesToFail ~ SensorRead02 + SensorRead03 + SensorRead04 + SensorRead07 +
  SensorRead11 + SensorRead12 + SensorRead15 + SensorRead17 + SensorRead20 + SensorRead21

TrainFormula <- CyclesToFail ~ SensorRead09 + SensorRead14 

#########################
#apply linear regression
## define and fit the linear regression model
lin_mod = lm(TrainFormula, data = train)

summary(lin_mod)$coefficients


print_metrics = function(lin_mod, df, score, label){
  resids = df[,label] - score
  resids2 = resids**2
  N = length(score)
  r2 = as.character(round(summary(lin_mod)$r.squared, 4))
  adj_r2 = as.character(round(summary(lin_mod)$adj.r.squared, 4))
  cat(paste('Mean Square Error      = ', as.character(round(sum(resids2)/N, 4)), '\n'))
  cat(paste('Root Mean Square Error = ', as.character(round(sqrt(sum(resids2)/N), 4)), '\n'))
  cat(paste('Mean Absolute Error    = ', as.character(round(sum(abs(resids))/N, 4)), '\n'))
  #cat(paste('Median Absolute Error  = ', as.character(round(median(abs(resids)), 4)), '\n'))
  cat(paste('R^2                    = ', r2, '\n'))
  cat(paste('Adjusted R^2           = ', adj_r2, '\n'))
}

#score = predict(lin_mod, newdata = test)
test$score <- predict(lin_mod, newdata = test)
print_metrics(lin_mod, test, score, label = 'CyclesToFail')  



hist_resids = function(df, score, label, bins = 10){
  options(repr.plot.width=4, repr.plot.height=3) # Set the initial plot area dimensions
  df$resids = df[,label] - score
  bw = (max(df$resids) - min(df$resids))/(bins + 1)
  ggplot(df, aes(resids)) + 
    geom_histogram(binwidth = bw, aes(y=..density..), alpha = 0.5) +
    geom_density(aes(y=..density..), color = 'blue') +
    xlab('Residual value') + ggtitle('Histogram of residuals')
}

hist_resids(test, score, label = 'CyclesToFail') 
######################



#########
#apply Random Forest
rf_model = randomForest(TrainFormula, data = train, ntree = 5)

test$scores = predict(rf_model, newdata = test)
head(test[,c(2,28)])
###############3









#Print confussion Matrix to review performance
print_metrics = function(df, label){
  ## Compute and print the confusion matrix
  cm = as.matrix(table(Actual = test$CyclesToFail, Predicted = test$scores))
  print(cm)
  
  ## Compute and print accuracy 
  #accuracy = round(sum(sapply(1:nrow(cm), function(i) cm[i,i]))/sum(cm), 3)
  #cat("\n")
  #cat(paste("Accuracy = ", as.character(accuracy)), "\n \n")                           
  
  ## Compute and print precision, recall and F1
  #precision = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[i,]))
 # recall = sapply(1:nrow(cm), function(i) cm[i,i]/sum(cm[,i]))    
  #F1 = sapply(1:nrow(cm), function(i) 2*(recall[i] * precision[i])/(recall[i] + precision[i]))    
  
  #metrics = sapply(c(precision, recall, F1), round, 3)        
  #metrics = t(matrix(metrics, nrow = nrow(cm), ncol = ncol(cm)))      
  #metrics = t(matrix(metrics, nrow = nrow(cm), ncol = 3))       
  #dimnames(metrics) = list(c("Precision", "Recall", "F1"), unique(test$CyclesToFail))      
  #print(metrics)
}  
print_metrics(test, "CyclesToFail")







options(repr.plot.width=4, repr.plot.height=3)
#imp = varImp(rf_model)
imp = varImp(lin_mod)



imp[,'Feature'] = row.names(imp)
ggplot(imp, aes(x = Feature, y = Overall)) + geom_point(size = 4) +
  ggtitle('Variable importance for features') +
  theme(axis.text.x = element_text(angle = 45))

options(repr.plot.width=8, repr.plot.height=6)
print(imp)

# Remove less important Variables
TrainFormulaRev <- TimeCycles ~ Operation_Setting_01 +  
  Operation_Setting_02 + 
  #Operation_Setting_03 + 
  #Sensor_Measure_01 + 
  Sensor_Measure_02 + 
  Sensor_Measure_03 + 
  Sensor_Measure_04 + 
  #Sensor_Measure_05 + 
  #Sensor_Measure_06 + 
  Sensor_Measure_07 + 
  Sensor_Measure_08 + 
  Sensor_Measure_09 + 
  #Sensor_Measure_10 + 
  Sensor_Measure_11 + 
  Sensor_Measure_12 + 
  Sensor_Measure_13 + 
  Sensor_Measure_14 + 
  Sensor_Measure_15 + 
  #Sensor_Measure_16 + 
  Sensor_Measure_17 + 
  #Sensor_Measure_18 + 
  #Sensor_Measure_19 + 
  Sensor_Measure_20 + 
  Sensor_Measure_21


