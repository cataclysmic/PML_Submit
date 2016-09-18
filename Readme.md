Loading the necessary standard libraries

library(caret)
library(ggplot2)
library(randomForest)


mainData <- read.csv("pml-training.csv")

Collecting information on the dataset.

#dim(mainData)
#names(mainData)
#table(mainData$classe)
#str(mainData)

Some variables appear to contain errors in the data (DIV/0!) or numerical values marked as strings.

#table(mainData$user_name)
#table(mainData$new_window)
#table(mainData$max_picth_arm)

Reloading the data, correcting for (DIV/0!).

mainData <- read.csv("pml-training.csv",na.strings="#DIV/0!")
#str(mainData)


Dropping useless variables before string to numeric conversion to avoid false conversion.

drops <- c("X","user_name","new_window","cvtd_timestamp","raw_timestamp_part_1","raw_timestamp_part_2","num_window")
    
mvars <- names(mainData) %in% drops
mainData<- mainData[!mvars]

Converting all variables to numeric with exception of classe.

for(i in seq(1,length(colnames(mainData))-1)) {
    mainData[,i] <- as.numeric(mainData[,i])
}
#str(mainData)

Several variables had no computable Std. Deviation and other variables had a zero variance. These are removed in the next step.

 drops2 <- c("kurtosis_yaw_belt", "skewness_yaw_belt", "kurtosis_yaw_dumbbell", "skewness_yaw_dumbbell", "kurtosis_yaw_forearm", "skewness_yaw_forearm","amplitude_yaw_belt", "amplitude_yaw_dumbbell", "amplitude_yaw_forearm")

mvars <- names(mainData) %in% drops2
mainData <- mainData[!mvars]

Further there is still a considerable number of variables that contain only NAs. We now filter those.

varNames <- colnames(mainData[colSums(is.na(mainData)) == 0])[-(1:7)]
mainData <- mainData[varNames]
#str(mainData)


pml-testing.csv dataset misses variables necessary for prediction using the potentially 'larger' model built focussing on available data in the training set alone. We compare available data in both sets and only keep variables available in both datasets.

predData <- read.csv("pml-testing.csv")

namePred <- colnames(predData[colSums(is.na(predData)) == 0])[-(1:7)]
namePred <- sort(c(namePred,"classe"))
nameTrain <- sort(names(mainData))
nameBoth <- intersect(nameTrain,namePred)

mainData <- mainData[nameBoth]
#names(mainData)


*Data cleaning process finished.*

========================================================================

set.seed(4356)

Creating data training und test set.

partitionIndex <- createDataPartition(y=mainData$classe,p=0.7,list=FALSE)
training <- mainData[partitionIndex,]
testing <- mainData[-partitionIndex,]

Doing 100 trees in a random forest approach. (Im on a slow machine and the process takes a while.)

modelRF <- randomForest(training[-13], training$classe, ntree=100)

Checking performance on the training set ...

confusionMatrix(training$classe,predict(modelRF,newdata=training))

... on the test set.

confusionMatrix(testing$classe,predict(modelRF,newdata=testing))

=========================================================================

Trying less computationally intensive ordered logit model.

library(MASS)

modelOlog <- polr(as.factor(classe)~.,data=training)

pred <- predict(modelOlog,training)

Checking performance on the training set ...

confusionMatrix(as.factor(training$classe),pred)

... on the test set.

confusionMatrix(testing$classe,predict(modelOlog,newdata=testing))

=========================================================================

Accuracy for the ordered logit model is much worse than for the random forest approach. Despite being a more computationally intensive approach, I would use the random forest approach, as the accuracy of the ordered logit model is below 50 percent.

As seen above the accuracy for the random forest approach is > 99% on the training set and nearly 99% for the test set. The prediction accuracy for the pml-testing.csv should be very high. In fact the 20 case prediction test yields 100% correct predictions.
