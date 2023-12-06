library(readr)
library(CGPfunctions)
library(dplyr)

FatalData <- read_csv("VicRoadFatalData.csv")
head(FatalData)
str(FatalData)

# surface conditions linked to higher fatality rates.
surf_cond <- xtabs(~ fatal + SURFACE_COND, data=FatalData)
PlotXTabs(FatalData, fatal, SURFACE_COND)
surf_cond_df <- as.data.frame.matrix(surf_cond)

# light conditions linked to higher fatality rates.
light_cond <- xtabs(~ fatal + LIGHT_CONDITION, data=FatalData)
light_cond

light_cond_df <- as.data.frame.matrix(light_cond)
#atmosphere condition
atmos_cond <- xtabs(~ FatalData$fatal + FatalData$ATMOSPH_COND)

atmos_cond_df <- as.data.frame.matrix(atmos_cond)


# Profile of Drivers
# sex linked to higher fatalitis
sex_profile <- xtabs(~ FatalData$fatal + FatalData$SEX)
sex_profile

# age group linked to hhigher fatalities
age_profile <- xtabs(~ FatalData$fatal + FatalData$`Age Group`)
age_profile

# helmet or seatbelt
helm_profile <- xtabs(~ FatalData$fatal + FatalData$HELMET_BELT_WORN)
helm_profile


# characteristics of the vehicle
bodystyle <- xtabs(~ FatalData$fatal + FatalData$VEHICLE_BODY_STYLE)
bodystyle 

make <- xtabs(~ FatalData$fatal + FatalData$VEHICLE_MAKE)
make

type <- xtabs(~ FatalData$fatal + FatalData$VEHICLE_TYPE)
type

#split the data into a training set and a test set (train 80%/ test 20%)
#we need to sample for imbalanced data (Fatal) 
# Method of downsampling
set.seed(3142)
Yes <- which(FatalData$fatal == "TRUE")
No <- which(FatalData$fatal == "FALSE")
length(Yes)
length(No)
No.downsample <- sample(No, length(Yes))
FatalData.down <- FatalData[c(No.downsample,Yes),]

downIndexSet <- sample(2, nrow(FatalData.down), replace = T, prob = c(0.8, 0.2))
train.down <- FatalData.down[downIndexSet==1,]
train.down$fatal <- ifelse(train.down$fatal == "TRUE", 1, 0)
test.down <- FatalData.down[downIndexSet==2,]
test.down$fatal <- ifelse(test.down$fatal == "TRUE", 1, 0)
interModelA.down <- glm(fatal ~ SEX + AGE + HELMET_BELT_WORN + 
                     VEHICLE_YEAR_MANUF + VEHICLE_BODY_STYLE + 
                     VEHICLE_MAKE + VEHICLE_TYPE +
                     FUEL_TYPE + TOTAL_NO_OCCUPANTS + 
                     ACCIDENTDATE + ACCIDENTTIME +
                     DAY_OF_WEEK + ACCIDENT_TYPE + 
                     LIGHT_CONDITION + ROAD_GEOMETRY +
                     SPEED_ZONE + SURFACE_COND +
                     ATMOSPH_COND + ROAD_SURFACE_TYPE, data = train.down, family = 'binomial')
summary(interModelA.down)

interModelB.down <- glm(fatal ~ SEX + AGE + HELMET_BELT_WORN + VEHICLE_TYPE +
                          TOTAL_NO_OCCUPANTS + ACCIDENTTIME + ACCIDENT_TYPE + 
                          LIGHT_CONDITION + ROAD_GEOMETRY + SPEED_ZONE +
                          SURFACE_COND + ATMOSPH_COND, data = train.down, family = 'binomial')
summary(interModelB.down)

library('caret')
expected_valueTest <- test.down$fatal
predicted_valueTest <- predict(interModelB.down, test.down)
predicted_valueTest <- ifelse(predicted_valueTest > 0, 1, 0)
confusionMatrix(data = as.factor(predicted_valueTest), reference = as.factor(expected_valueTest))

# AUC of ROC of logistic model against test data
library(pROC)
roc_object <- roc( expected_valueTest, predicted_valueTest )
auc(roc_object)

# output summary
sink("logisticModelOutput.txt")
print(summary(interModelB.down))
sink()

# Scale down test and training set
scaled.train.down <- train.down %>%
  select(!fatal) %>%
  mutate_if(is.numeric, scale) %>%
  mutate(fatal=train.down$fatal)

scaled.test.down <- test.down %>%
  select(!fatal) %>%
  mutate_if(is.numeric, scale) %>%
  mutate(fatal=test.down$fatal)



# Predictive Models
# stratified K fold cross validation on elastic net logistic regression
set.seed(3142)
folds <- 10
cvIndex <- createFolds(factor(scaled.train.down$fatal), folds, returnTrain = T)
tc <- trainControl(index = cvIndex,
                   method = 'cv', 
                   number = folds)

lr_fit <- train(factor(fatal) ~ SEX + 
                  AGE + LICENCE_STATE 
                + HELMET_BELT_WORN +
                  VEHICLE_YEAR_MANUF
                + VEHICLE_BODY_STYLE
                + VEHICLE_MAKE
                + VEHICLE_TYPE
                + FUEL_TYPE
                + VEHICLE_COLOUR
                + TOTAL_NO_OCCUPANTS, 
                data = scaled.train.down,
               method = "glmnet",
               family = "binomial",
               trControl = tc,
               verbose = FALSE)

lr_fit
coef(lr_fit$finalModel, lr_fit$bestTune$lambda)
print(varImp(lr_fit))

# according to stratified 10 fold cross validation, optimal model for
# logistic regression is an elastic net regression with 
# alpha = 0.55
# lambda = 0.0001795657
# training accuracy = 0.6117673
# error rate = 0.3882327

lr_train_pred <- predict(lr_fit, scaled.train.down)
confusionMatrix(data = factor(lr_train_pred), reference = factor(scaled.train.down$fatal))

# confusion matrix training accuracy = 0.6263
# confusion matrix error rate = 0.3737
# false positive rate = 0.3130
# true positive rate = 0.5659


# stratified K fold cross validation on random forest
folds <- 5
rfcvIndex <- createFolds(factor(train.down$fatal), folds, returnTrain = T)
rftc <- trainControl(index = rfcvIndex,
                     method = 'cv', 
                     number = folds)

rf_fit <- train(factor(fatal) ~ SEX + 
                  AGE + LICENCE_STATE 
                + HELMET_BELT_WORN +
                  VEHICLE_YEAR_MANUF
                + VEHICLE_BODY_STYLE
                + VEHICLE_MAKE
                + VEHICLE_TYPE
                + FUEL_TYPE
                + VEHICLE_COLOUR
                + TOTAL_NO_OCCUPANTS, 
                data = train.down,
                method = "rf",
                trControl = rftc,
                tuneLength = 6)

print(rf_fit) # should give the mtry that has best accuracy
# mtry --> the number of variables to randomly sample as candidates at each split
# Accuracy = 0.6022969 with mtry = 2

rf_train_pred <- predict(rf_fit, train.down)
confusionMatrix(data = factor(rf_train_pred), reference = factor(train.down$fatal))

# confusion matrix training accuracy = 0.6466
# confusion matrix error rate = 0.3534
# false positive rate = 0.2036
# true positive rate = 0.4971

print(varImp(rf_fit))
plot(varImp(rf_fit))


# stratified K fold cross validation on knn
knnfolds <- 10
knncvIndex <- createFolds(factor(scaled.train.down$fatal), knnfolds, returnTrain = T)
knntc <- trainControl(index = knncvIndex,
                     method = 'cv', 
                     number = knnfolds)


knn_fit <- train(factor(fatal) ~ SEX + 
                  AGE + LICENCE_STATE 
                + HELMET_BELT_WORN +
                  VEHICLE_YEAR_MANUF
                + VEHICLE_BODY_STYLE
                + VEHICLE_MAKE
                + VEHICLE_TYPE
                + FUEL_TYPE
                + VEHICLE_COLOUR
                + TOTAL_NO_OCCUPANTS, 
                data = scaled.train.down,
                method = "knn",
                trControl = knntc,
                preProcess = c("center", "scale"),
                tuneLength = 10)

knn_fit # best optimed tuning param is k = 9
# Accuracy = 0.5886350 
# Error rate = 0.411365 


knn_train_pred <- predict(knn_fit, scaled.train.down)
confusionMatrix(data = factor(knn_train_pred), reference = factor(scaled.train.down$fatal))
# confusion matrix training accuracy = 0.6817
# confusion matrix error rate = 0.3183
# false positive rate = 0.2758
# true positive rate = 0.6394

# KNN and LR both performed well, lets evaluate both to see which one is better
# using confusion matrix

knn_test_pred <- predict(knn_fit, scaled.test.down)
confusionMatrix(data = factor(knn_test_pred), reference = factor(scaled.test.down$fatal))
# Confusion matrix test accuracy = 0.5572
# Confusion matrix error rate = 0.4428
# false positive rate = 0.4164
# true positive rate = 0.5305
knn_roc_object <- roc( scaled.test.down$fatal, as.numeric(knn_test_pred) )
auc(knn_roc_object) # 0.5571 

lr_test_pred <- predict(lr_fit, scaled.test.down)
confusionMatrix(data = factor(lr_test_pred), reference = factor(scaled.test.down$fatal))
# Confusion matrix test accuracy = 0.6062
# Confusion matrix error rate = 0.3938
# false positive rate = 0.3406
# true positive rate = 0.5524
lr_roc_object <- roc( scaled.test.down$fatal, as.numeric(lr_test_pred) )
auc(lr_roc_object) # 0.6059

rf_test_pred <- predict(rf_fit, test.down)
confusionMatrix(data = factor(rf_test_pred), reference = factor(test.down$fatal))
# Confusion matrix test accuracy = 0.6132
# Confusion matrix error rate = 0.3868
# false positive rate = 0.2415
# true positive rate = 0.4664
rf_roc_object <- roc( test.down$fatal, as.numeric(rf_test_pred) )
auc(rf_roc_object) # 0.6124 


# To conclude: since we are interested in being correct in identifying which
# accidents result in fatality, we want a model with good true positive rate.
# Therefore our selected model is the elastic net LR model.
# The RF model does a better prediction but it is best at minimising false
# positives. Although this is a good feature in a model, in the context of
# our problem, this feature is not prioritised.

# insert driver eval data

driver_eval_data <- read_csv("Drivers_Eval.csv")

# compute their probabilities using elastic net lr
scaled.driver_eval <- driver_eval_data %>%
  mutate_if(is.numeric, scale) 

lr_driver_eval_pred <- predict(lr_fit, scaled.driver_eval, type = "prob")

# attach probabilities to driver eval

dfEval <- cbind(driver_eval_data, lr_driver_eval_pred)

# order driver eval descending on probabilities
dfEval <- dfEval[order(dfEval$'1',decreasing=TRUE),]

# add top 2500 to new csv
selected_drivers <- as.data.frame(dfEval$DRIVER_ID)
colnames(selected_drivers[1]) = "DRIVER_ID"

sdtest <- head(selected_drivers, 2500)

# output csv
write.csv(sdtest, "selected_drivers.csv")