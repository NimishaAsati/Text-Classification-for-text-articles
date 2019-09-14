# ----------------------------- Multiclass Support Vector Machine

# Changing the working directory to the location where data files recide
setwd("C:/Personal/Fall 2018/Statistical Models and Methods for Business Analysis/SMMBA_Assignments/Homework 2/Dataset")

#Loading training data
dataFile <- file("articles.train", "r")
dataLines <- readLines(dataFile)
m <- length(dataLines)
close(dataFile)

# Split every string element by tokenizing space and colon.
dataTokens = strsplit(dataLines, "[: ]")

# Extract every first token from each line as a vector of numbers, which is the class label.
training_Y = sapply(dataTokens, function(example) {as.numeric(example[1])})

# Extract the rest of tokens from each line as a list of matrices (one matrix for each line)
# where each row consists of two columns: (feature number, its occurrences)
X_list = lapply(dataTokens, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})


# Add one column that indicates the example number at the left
X_list = mapply(cbind, x=1:length(X_list), y=X_list)

# Merge a list of different examples vertically into a matrix
X_data_train = do.call('rbind', X_list)

#adding maximum data point of test data to make the train and test data similar
datapoint <- matrix(c(x=1,60636,0), nrow = 1, ncol = 3)

X_list2 <- rbind(datapoint,X_data_train)

# Get a sparse data matrix X (rows: training exmaples, columns: # of occurrences for each of features)
X_training = sparseMatrix(x=X_list2[,3], i=X_list2[,1], j=X_list2[,2])

#Loading test data
dataFile <- file("articles.test","r")
dataLines <- readLines(dataFile)
m <- length(dataLines)
close(dataFile)

# Split every string element by tokenizing space and colon.
dataTokens = strsplit(dataLines, "[: ]")

# Extract every first token from each line as a vector of numbers, which is the class label.
test_Y = sapply(dataTokens, function(example) {as.numeric(example[1])})

# Extract the rest of tokens from each line as a list of matrices (one matrix for each line)
# where each row consists of two columns: (feature number, its occurrences)
X_list = lapply(dataTokens, function(example) {n = length(example) - 1; matrix(as.numeric(example[2:(n+1)]), ncol=2, byrow=T)})

# Add one column that indicates the example number at the left
X_list = mapply(cbind, x=1:length(X_list), y=X_list)

# Merge a list of different examples vertically into a matrix
X_data_test = do.call('rbind', X_list)

# Get a sparse data matrix X (rows: training exmaples, columns: # of occurrences for each of features)
X_test = sparseMatrix(x=X_data_test[,3], i=X_data_test[,1], j=X_data_test[,2])


test_unique <- unique(X_data_test[,2])
summary(test_unique)
#last column of test data 60636


# ----------------------------- Question 5 (b)-----------------------------

library(e1071)
set.seed(567)

# Changing target class number to 1 and others to -1 in training and test dataset for 1st variable
training_Y1 <-lapply(training_Y, function(x) if (x == 1) {1} else {-1})
training_Y1 <- as.numeric(training_Y1)
training_Y1 <- as.factor(training_Y1)

test_Y1 <-lapply(test_Y, function(x) if (x == 1) {1} else {-1})
test_Y1 <- as.numeric(test_Y1)
test_Y1 <- as.factor(test_Y1)

# Changing target class number to 1 and others to -1 in training and test dataset for 2nd variable
training_Y2 <-lapply(training_Y, function(x) if (x == 2) {1} else {-1})
training_Y2 <- as.numeric(training_Y2)
training_Y2 <- as.factor(training_Y2)

test_Y2 <-lapply(test_Y, function(x) if (x == 2) {1} else {-1})
test_Y2 <- as.numeric(test_Y2)
test_Y2 <- as.factor(test_Y2)

# Changing target class number to 1 and others to -1 in training and test dataset for 3rd variable
training_Y3 <-lapply(training_Y, function(x) if (x == 3) {1} else {-1})
training_Y3 <- as.numeric(training_Y3)
training_Y3 <- as.factor(training_Y3)

test_Y3 <-lapply(test_Y, function(x) if (x == 3) {1} else {-1})
test_Y3 <- as.numeric(test_Y3)
test_Y2 <- as.factor(test_Y2)

# Changing target class number to 1 and others to -1 in training and test dataset for 4th variable
training_Y4 <-lapply(training_Y, function(x) if (x == 4) {1} else {-1})
training_Y4 <- as.numeric(training_Y4)
training_Y4 <- as.factor(training_Y4)

test_Y4 <-lapply(test_Y, function(x) if (x == 4) {1} else {-1})
test_Y4 <- as.numeric(test_Y4)
test_Y4 <- as.factor(test_Y4)


# Hard-Margin Linear Classifiers with cost = 1000
hard1 <- svm(X,training_Y1,kernel = "linear", decision.values=TRUE, cost = 1000)
hard2 <- svm(X,training_Y2,kernel = "linear", decision.values=TRUE, cost = 1000)
hard3 <- svm(X,training_Y3,kernel = "linear", decision.values=TRUE, cost = 1000)
hard4 <- svm(X,training_Y4,kernel = "linear", decision.values=TRUE, cost = 1000)


# Getting prediction values from all models for both training and test data
pred_train1 = predict(hard1, X_training)
pred_test1 <- predict(hard1, X_test)

pred_train2 = predict(hard2, X_training)
pred_test2 <- predict(hard2, X_test)

pred_train3 = predict(hard3, X_training)
pred_test3 <- predict(hard3, X_test)

pred_train4 = predict(hard4, X_training)
pred_test4 <- predict(hard4, X_test)


# Creating a consolidated table with all training results
training_results <- data.frame(pred_train1,pred_train2,pred_train3, pred_train4)
colnames(training_results) <- c(1,2,3,4)
training_final = colnames(training_results)[apply(training_results,1,which.max)]

# Creating a consolidated table with all test results
test_results <- data.frame(pred_test1,pred_test2,pred_test3,pred_test4)
colnames(test_results) <- c(1,2,3,4)
test_final = colnames(test_results)[apply(test_results,1,which.max)]

# Accuracy on training data
training_Y <- as.character(training_Y)
mean(training_final==training_Y)*100

# Accuracy on test data
training_Y <- as.character(training_Y)
mean(test_final==test_Y)*100


# ----------------------------- Question 5 (c)-----------------------------

library(dplyr)

# Changing the working directory to the location where data files recide
setwd("C:/Personal/Fall 2018/Statistical Models and Methods for Business Analysis/SMMBA_Assignments/Homework 2/Dataset")

#Loading training data
dataFile <- file("articles.train", "r")
dataLines <- readLines(dataFile)

#Splitting the above entire training data into 75% training and 25% validation
train<-sample_frac(dataLines, 0.75)
sid<-as.numeric(rownames(train))

# Changing target class number to 1 and others to -1 in new training dataset for 1st variable
train_Y1 <-lapply(train, function(x) if (x == 1) {1} else {-1})
train_Y1 <- as.numeric(train_Y1)
train_Y1 <- as.factor(train_Y1)

# Changing target class number to 1 and others to -1 in new training dataset for 2nd variable
train_Y2 <-lapply(train, function(x) if (x == 2) {1} else {-1})
train_Y2 <- as.numeric(train_Y2)
train_Y2 <- as.factor(train_Y2)

# Changing target class number to 1 and others to -1 in new training dataset for 3rd variable
train_Y3 <-lapply(train, function(x) if (x == 3) {1} else {-1})
train_Y3 <- as.numeric(train_Y3)
train_Y3 <- as.factor(train_Y3)

# Changing target class number to 1 and others to -1 in new training dataset for 4th variable
train_Y4 <-lapply(train, function(x) if (x == 4) {1} else {-1})
train_Y4 <- as.numeric(train_Y4)
train_Y4 <- as.factor(train_Y4)


# Running Soft SVM models

# For C=0.125
svm_model_c1 <- svm(X,train_Y1, data=train, cost=0.125, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=0.125, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=0.125, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=0.125, method="C-classification", kernel="linear")

# For C=0.25
svm_model_c1 <- svm(X,train_Y1, data=train, cost=0.25, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=0.25, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=0.25, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=0.25, method="C-classification", kernel="linear")

# For C=0.5
svm_model_c1 <- svm(X,train_Y1, data=train, cost=0.5, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=0.5, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=0.5, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=0.5, method="C-classification", kernel="linear")

# For C=1
svm_model_c1 <- svm(X,train_Y1, data=train, cost=1, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=1, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=1, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=1, method="C-classification", kernel="linear")

# For C=2
svm_model_c1 <- svm(X,train_Y1, data=train, cost=2, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=2, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=2, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=2, method="C-classification", kernel="linear")

# For C=20
svm_model_c1 <- svm(X,train_Y1, data=train, cost=20, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=20, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=20, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=20, method="C-classification", kernel="linear")

# For C=256
svm_model_c1 <- svm(X,train_Y1, data=train, cost=256, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=256, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=256, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=256, method="C-classification", kernel="linear")

# For C=512
svm_model_c1 <- svm(X,train_Y1, data=train, cost=512, method="C-classification", kernel="linear")
svm_model_c2 <- svm(X,train_Y2, data=train, cost=512, method="C-classification", kernel="linear")
svm_model_c3 <- svm(X,train_Y3, data=train, cost=512, method="C-classification", kernel="linear")
svm_model_c4 <- svm(X,train_Y4, data=train, cost=512, method="C-classification", kernel="linear")

