---
title: "Pratical Machine Learning"
author: "Felix Albrecht"
date: "September 18, 2016"
output: html_document
---

Loading the necessary standard libraries

```{r}
library(caret)
library(ggplot2)
library(randomForest)

origData <- read.csv("pml-training.csv")
```

Collecting information on the dataset.

```{r}
#dim(origData)
#names(origData)
#table(origData$classe)
#str(origData)
```

Some variables appear to contain errors in the data (DIV/0!) or numerical values marked as strings.

```{r}
#table(origData$user_name)
#table(origData$new_window)
#table(origData$max_picth_arm)
```

Reloading the data, correcting for (DIV/0!).

```{r}
origData <- read.csv("pml-training.csv",na.strings="#DIV/0!")
#str(origData)
```


Dropping useless variables before string to numeric conversion to avoid false conversion.

```{r}
drops <- c("X","user_name","new_window","cvtd_timestamp","raw_timestamp_part_1","raw_timestamp_part_2","num_window")
    
mvars <- names(origData) %in% drops
origData<- origData[!mvars]
```

Converting all variables to numeric with exception of classe.

```{r}
for(i in seq(1,length(colnames(origData))-1)) {
    origData[,i] <- as.numeric(origData[,i])
}
#str(origData)
```

Several variables had no computable Std. Deviation and other variables had a zero variance. These are removed in the next step.

```{r}
 drops2 <- c("kurtosis_yaw_belt", "skewness_yaw_belt", "kurtosis_yaw_dumbbell", "skewness_yaw_dumbbell", "kurtosis_yaw_forearm", "skewness_yaw_forearm","amplitude_yaw_belt", "amplitude_yaw_dumbbell", "amplitude_yaw_forearm")

mvars <- names(origData) %in% drops2
origData <- origData[!mvars]
```

Further there is still a considerable number of variables that contain only NAs. We now filter those.

```{r}
varNames <- colnames(origData[colSums(is.na(origData)) == 0])[-(1:7)]
origData <- origData[varNames]
#str(origData)
```


pml-testing.csv dataset misses variables necessary for prediction using the potentially 'larger' model built focussing on available data in the training set alone. We compare available data in both sets and only keep variables available in both datasets.

```{r}
predData <- read.csv("pml-testing.csv")

namePred <- colnames(predData[colSums(is.na(predData)) == 0])[-(1:7)]
namePred <- sort(c(namePred,"classe"))
nameTrain <- sort(names(origData))
nameBoth <- intersect(nameTrain,namePred)

origData <- origData[nameBoth]
#names(origData)
```


*Data cleaning process finished.*

========================================================================

```{r}
set.seed(4356)
```

Creating data training und test set.

```{r}
partitionIndex <- createDataPartition(y=origData$classe,p=0.7,list=FALSE)
training <- origData[partitionIndex,]
testing <- origData[-partitionIndex,]
```

Doing 100 trees in a random forest approach. (Im on a slow machine and the process takes a while.)

```{r}
modelRF <- randomForest(training[-13], training$classe, ntree=100)
```

Checking performance on the training set ...

```{r}
confusionMatrix(training$classe,predict(modelRF,newdata=training))
```

... on the test set.

```{r}
confusionMatrix(testing$classe,predict(modelRF,newdata=testing))
```

=========================================================================

Trying less computationally intensive ordered logit model.

```{r}
library(MASS)

modelOlog <- polr(as.factor(classe)~.,data=training)

pred <- predict(modelOlog,training)
```

Checking performance on the training set ...

```{r}
confusionMatrix(as.factor(training$classe),pred)
```

... on the test set.

```{r}
confusionMatrix(testing$classe,predict(modelOlog,newdata=testing))
```
=========================================================================

Accuracy for the ordered logit model is much worse than for the random forest approach. Despite being a more computationally intensive approach, I would use the random forest approach, as the accuracy of the ordered logit model is below 50 percent.

As seen above the accuracy for the random forest approach is > 99% on the training set and nearly 99% for the test set. The prediction accuracy for the pml-testing.csv should be very high. In fact the 20 case prediction test yields 100% correct predictions.

=========================================================================
