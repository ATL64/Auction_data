library(randomForest)
library(caret)
library(pROC)
library(dplyr)



##############################################
######## Introduction ########################
##############################################

# In this section we will do a very basic preprocessing of the dataset (no feature engineering at all),
# and train two basic models for classifying purchases vs no purchases.
# The reason to do a short inspection here is to
# a) Benchmark the model performance
# b) Check the output of variable importance
# 
# The training of models will be simple, using the caret package, and we will not optimize grid searches, specific metrics nor use all the rows available.
# Given the little optimization done, the results need to be taken with a grain of salt.


##############################################
######### Read and pre process the data ######
##############################################

#Read the csv
auction_dataset<-read.csv("/Users/Documents/Auction dataset/dataset.csv", sep=',', header = TRUE)



## Number of unique sellers, segments, categories
length(unique(auction_dataset$us_id)) # 242106
length(unique(auction_dataset$buyer_group)) # 6
length(unique(auction_dataset$product_group)) # 3213
nrow(auction_dataset[auction_dataset$total_bids==1,]) #229428


#Binary output column
auction_dataset$output<-as.numeric(auction_dataset$after_8_day_purchases>1)
sum(auction_dataset$output==1) #45492, which leaves purchases at around 6.5% minority class. I will not deal with unbalanced classes in this exercise due to time constraint.a


#Remove non predictor variables
auction_dataset$us_id<-NULL
auction_dataset$product_group<-NULL
#Note: Removing category variables due to their high number of values. We could try other approaches as gathering them into groups.
auction_dataset$after_8_day_value<-NULL
auction_dataset$after_8_day_purchases<-NULL


#Create data for training. Given my current computer limitations, I will take the data size down (of course we could easily spin up a powerful VM on the cloud and run it on the whole dataset).
set.seed(2018) #For reproducibility
auction_dataset<-auction_dataset[sample(nrow(auction_dataset), 4000), ]
auction_dataset$output<-as.factor(auction_dataset$output)
auction_dataset$buyer_group<-as.factor(gsub(" ", "_",gsub("/", "_",substr(as.character(auction_dataset$buyer_group),4,100))))
levels(auction_dataset$output)<-c('no_purchase','purchase')




##############################################
######### Train some models ##################
##############################################




train <- auction_dataset[1:2000,]
test  <-  auction_dataset[2001:4000,]
prop.table(table(train$output))

# no_purchase    purchase 
# 0.9415      0.0585 

ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

set.seed(2222)

orig_fit <- train(output ~ .,
                  data = train,
                  method = "gbm",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)


test_roc <- function(model, data) {
  roc(data$output,
      predict(model, data, type = "prob")[, "purchase"])
}

orig_fit %>%
  test_roc(data = test) %>%
  auc()

#Area under the curve: 0.849

reference<-test$output
predictions<-predict(orig_fit, test)  

confusionMatrix(predictions,reference)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    no_purchase purchase
# no_purchase        1833       98
# purchase             27       42
# 
# Accuracy : 0.9375         
# 95% CI : (0.926, 0.9477)
# No Information Rate : 0.93           
# P-Value [Acc > NIR] : 0.1004         
# 
# Kappa : 0.3729         
# Mcnemar's Test P-Value : 3.825e-10      
#                                          
#             Sensitivity : 0.9855         
#             Specificity : 0.3000         
#          Pos Pred Value : 0.9492         
#          Neg Pred Value : 0.6087         
#              Prevalence : 0.9300         
#          Detection Rate : 0.9165         
#    Detection Prevalence : 0.9655         
#       Balanced Accuracy : 0.6427         
#                                          
#        'Positive' Class : no_purchase 



# Let's try a random forest and check variable importance

set.seed(2222)

rf_fit <- train(output ~ .,
                  data = train,
                  method = "rf",
                  verbose = FALSE,
                  metric = "ROC",
                  trControl = ctrl)



rf_fit %>%
  test_roc(data = test) %>%
  auc()

#Area under the curve: 0.8324

fm <- rf_fit$finalModel
varImp(fm)


#                          Overall
# buyer_groupegory_2      2.054966
# buyer_groupegory_3      4.051649
# buyer_groupegory_4      2.029386
# buyer_groupegory_5      1.147509
# buyer_groupegory_6      2.070385
# duration_of_auction     5.372963
# price_start            12.052725
# bids                    9.122314
# first_3_day_bids        7.595729
# last_3_day_bids         8.906558
# price_final            12.556377
# final_price_percentile 15.065981
# last_8_day_searches    14.722177
# last_8_day_item_views  26.806301
# last_8_day_purchases   23.495291
# last_3_day_searches    10.032758
# last_3_day_item_views  19.814833
# last_3_day_purchases    8.040225


