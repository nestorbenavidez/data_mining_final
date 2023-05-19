#setwd("C:/Users/gonca/OneDrive/Ambiente de Trabalho/Project Data Mining")
predictive_maintenance= read.csv("predictive_maintenance.csv")

#Exploratory data analysis#

str(predictive_maintenance)
predictive_maintenance[,c(1:3,9,10)] <- lapply(predictive_maintenance[,c(1,2,3,9,10)], as.factor)
predictive_maintenance[,-c(1:3,9,10)] <- lapply(predictive_maintenance[,-c(1,2,3,9,10)], as.numeric)

#Missing values:
summary(predictive_maintenance)
sum(is.na(predictive_maintenance)) #predictive_maintenance doesn't have missing values

#Understand if all target=0 corresponds to 0 failures
predictive_maintenance$Target[predictive_maintenance$Target == 0]
predictive_maintenance$Failure.Type[predictive_maintenance$Failure.Type == "No Failure"]

predictive_maintenance$Target[predictive_maintenance$Failure.Type == "Random Failures"] <-"1"
predictive_maintenance$Failure.Type[predictive_maintenance$Target == 0] <- "No Failure"
predictive_maintenance$Target[predictive_maintenance$Failure.Type == "No Failure"] <- "0"

summary(predictive_maintenance$Failure.Type)

#Outliers
boxplot(predictive_maintenance$Air.temperature..K.,main="BoxPlot Air Temperature")
boxplot(predictive_maintenance$Process.temperature..K., main="BoxPlot Process Temperature")
boxplot(predictive_maintenance$Rotational.speed..rpm., main="BoxPlot Rotational Speed")
boxplot(predictive_maintenance$Torque..Nm., main="BoxPlot Torque")
boxplot(predictive_maintenance$Tool.wear..min., main="BoxPlot Tool Wear")

outliers<-boxplot.stats(predictive_maintenance$Rotational.speed..rpm.)$out
outliers1<-boxplot.stats(predictive_maintenance$Torque..Nm.)$out

#Correlation#
library(corrplot)
predictive_maintenance_numeric <- predictive_maintenance[, c(4:8)]
corr_matrix <- cor(predictive_maintenance_numeric)

# Display the correlation matrix as a plot
corrplot(corr_matrix,method = "number",tl.cex = 0.6)

#Select the meaningful data for the prediction
df= predictive_maintenance[,c(3,4,5,6,7,8,9)]
str(df)


#######################################################################

#Predictive analysis#

#Split the data into training and testing data

install.packages("splitTools")
library(splitTools)


set.seed(123)
split_predictive_maintenance <- partition(predictive_maintenance$Target, p = c(train = 0.7, test = 0.3))
str(split_predictive_maintenance)

trainset <- df[split_predictive_maintenance$train,]
testset <- df[split_predictive_maintenance$test,]

#Balance trainset
install.packages("caret")
library(caret)
balanced_train_T <- downSample(trainset,trainset$Target)
balanced_train_FT <- downSample(trainset,trainset$Failure.Type)

balanced_train_T= balanced_train_T[-c(8)]


#KNN#
library("fastDummies")
train_dummies <- dummy_cols(balanced_train_T, select_columns=c("Type"),
                            remove_first_dummy = TRUE,
                            remove_selected_columns = TRUE)
test_dummies <- dummy_cols(testset, select_columns=c("Type"),
                           remove_first_dummy = TRUE,
                           remove_selected_columns = TRUE)
str(train_dummies)
str(test_dummies)
train_dummies[,c(7,8)] <- lapply(train_dummies[,c(7,8)], as.numeric)
test_dummies[,c(7,8)] <- lapply(test_dummies[,c(7,8)], as.numeric)

#install.packages("FNN")
library(FNN)
prediction <- knn(train_dummies[,-c(6)], test_dummies[,-c(6)],
                  train_dummies$Target,k=2)

# Create a confusion matrix
confusion_knn <- confusionMatrix(prediction, testset$Target)
confusion_knn
confusion_knn$overall

#Training the Decision Tree Classifier
install.packages("rpart")
library(rpart)
library(rpart.plot)
install.packages("rattle")
library(rattle)


tree= rpart(Target~ ., ,data = balanced_train_T, method = "class",parms = list(split= "information"), minsplit=4, minbucket=3)
#Visualization of the tree

install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(tree)
prp(tree)

#Predictions

tree.target.predicted = predict(tree, testset, type = "class")

#Confusion Matrix for evaluating the model

library(caret)
confMatrix_tree= confusionMatrix(tree.target.predicted, testset$Target)
confMatrix_tree
confMatrix_tree$overall


#With the Tree pruned

tree_pruned= prune(tree, cp=0.03)
prp(tree_pruned)
prediction_tree_pruned= predict(tree_pruned, testset, type = "class")

#Confusion Matrix
confuMatrix_tree_pruned= confusionMatrix(prediction_tree_pruned, testset$Target)
confuMatrix_tree_pruned


#Gini Index

tree_gini= rpart(Target~., data = balanced_train_T, method="class", parms = list(split="gini"), minsplit=4, minbucket=3)
prp(tree_gini)
prediction_tree_gini= predict(tree_gini, testset, type= "class")

#Confusion Matrix

confuMatrix_tree_gini= confusionMatrix(prediction_tree_gini, testset$Target)
confuMatrix_tree_gini

#With the tree pruned
tree_pruned_gini = prune(tree_gini, cp=0.03)
prp(tree_pruned_gini)
prediction_tree_pruned_gini= predict(tree_pruned_gini, testset, type="class")

#Confusion Matrix
confuMatrix_tree_pruned_gini= confusionMatrix(prediction_tree_pruned_gini, testset$Target)
confuMatrix_tree_pruned_gini

# Load the required library
library(e1071)

# Read the dataset
predictive_maintenance <- read.csv("predictive_maintenance.csv")

# Remove unnecessary columns from the dataset
predictive_maintenance <- predictive_maintenance[, !(names(predictive_maintenance) %in% c("UID", "productID", "Failure.Type"))]

# Set a seed for reproducibility
set.seed(123)

# Generate random indices for train/test split
train_indices <- sample(nrow(predictive_maintenance), 0.8 * nrow(predictive_maintenance))  # 80% for training, adjust as needed
trainset <- predictive_maintenance[train_indices, ]
testset <- predictive_maintenance[-train_indices, ]

# Check the column names and structure of the dataset
print(names(trainset))
str(trainset)

# Identify the correct column name for the target variable
target_column <- "Target"  # The target column name is "Target"

# Train the Naive Bayes model
model_NB <- naiveBayes(as.formula(paste(target_column, "~ .")), data = trainset)

# Make predictions on the test set
predictions_NB <- predict(model_NB, newdata = testset)

# Create a confusion matrix
ConfuMatrix_NB <- table(predictions_NB, testset[, target_column])

# Print the confusion matrix
print(ConfuMatrix_NB)

# Calculate the accuracy
accuracy <- sum(diag(ConfuMatrix_NB)) / sum(ConfuMatrix_NB)

# Print the accuracy
cat("Accuracy:", accuracy, "\n")


