install.packages("PerformanceAnalytics")
install.packages("caret", dependencies = c("Depends", "Suggests"))    
require(caret)
library(car)
library(PerformanceAnalytics)
library(MASS)
library(rpart)
library(rpart.plot)
options(scipen = 999, digits = 3)
#options(warn=-1)

dataset <- read.csv("features_rfm.csv",
                    sep=",", dec = ".",
                    stringsAsFactors=TRUE, na.strings = "NA", row.names = 1)


#####################################
# Exploratory Data Analysis and Preprocessing

head(dataset)
table(dataset$target)

str(dataset)

#separating user_id from dataset
client_id = dataset[1]
dataset$user_id <- NULL

#changing target from 0/1 to C0/C1 in order to have it categoriacal
dataset$target<-paste("c",dataset$target,sep="")

#setting taget as a categorial binary feature
dataset$target<-factor(dataset$target)

#setting shipping_cap as a categorial muticategry feature
dataset$shipping_cap<-factor(dataset$shipping_cap)

#setting referral_token  as a categorial binary feature
dataset$referral_token<-factor(dataset$referral_token)

str(dataset)

#separating categorical and numerical features in two dataset

isfactor <- sapply(dataset, function(x) is.factor(x))
isfactor
factordata <- dataset[, isfactor]
str(factordata)

numeric <- sapply(dataset, function(x) is.numeric(x))
numeric <-dataset[, numeric]
str(numeric)


# 1. missing data

sapply(dataset, function(x)(sum(is.na(x))))
#there are no missing data so we do not need to delete observations with missing or impute them.


#2. collinearity

R=cor(numeric)
R

correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors
summary(R[upper.tri(R)])

# visualize the correlation plot
chart.Correlation(numeric, method = "pearson", histogram=TRUE, pch=22, height=1500, width=1500)

# Save the plot
jpeg("rplot.jpg", width = 1500, height = 1500)
chart.Correlation(numeric, method = "pearson", histogram=TRUE, pch=22, height=1500, width=1500)
dev.off()

# removing the feature with collinearity -> registration
numeric_sel = numeric [,-1]
str(numeric_sel)

# create an unique dataset with numeric and factor variabiles
all=cbind(numeric_sel,factordata)
head(all,n=10)


# 3. Non Zero-Variance

nzv = nearZeroVar(all, saveMetrics = TRUE)
nzv

# removing the feature with no variance -> service_rent_count
dataset_sel = all [,-9]
str(dataset_sel)


# 4. transformation and scaling

dataset_scaled <- preProcess(dataset_sel, method = c("scale", "BoxCox"))
dataset_scaled
# the method Box-Cox transformation has transformed two features: registration and last_sign_in_at
# the method scale has scaled 123 features and ignored the two categorical variables

ls(dataset_scaled)
dataset_scaled$bc

# adding the data to the scale object  
dataset_sel_scaled=predict(dataset_scaled, newdata = all)
head(dataset_sel_scaled)

#comparing the distribution in the transformed feature
#setting e two way plotting areas in the plots window
par(mfrow=c(2,2))

hist(dataset_sel$last_sign_in_at)
hist(dataset_sel_scaled$last_sign_in_at)

#taking back the plots window to standard
par(mfrow=c(1,1))


# 5. Visualizing data distribution compared to the target

# referral_token
counts <- table(dataset_sel_scaled$target, dataset_sel_scaled$referral_token)
barplot(counts,
        xlab="", col=c("darkblue","red"),
        legend = rownames(counts),main="Referral Token",
        args.legend=list(
          x=ncol(counts) +0.5,
          y=max(colSums(counts))
        ))

# coupon_count
counts <- table(dataset_sel_scaled$target, dataset_sel_scaled$coupon_count)
barplot(counts,
        xlab="", col=c("darkblue","red"),
        legend = rownames(counts),main="Coupon Count",
        args.legend=list(
          x=ncol(counts) +0.5,
          y=max(colSums(counts))
        ))

# used_bonus_points
counts <- table(dataset_sel_scaled$target, dataset_sel_scaled$used_bonus_points)
barplot(counts,
        xlab="", col=c("darkblue","red"),
        legend = rownames(counts),main="Used Bonus Points",
        args.legend=list(
          x=ncol(counts) +0.5,
          y=max(colSums(counts))
        ))


# 6. Features Selection through regression model with step AIC procedure

fit <- glm(target~. , data=dataset_sel_scaled, family="binomial")
summary(fit)
step <- stepAIC(fit, direction="both")

# training of a glm model with the features selected by the sep AIC process
set.seed(1234)
Control=trainControl(method= "cv", number=5,  classProbs = TRUE)

glm_aicPP = train(target ~ last_sign_in_at + used_bonus_points + service_wash_count + 
                    service_rent_count , data=dataset_sel_scaled ,  method = "glm", trControl = Control)

confusionMatrix(glm_aicPP)
summary(glm_aicPP)


# 7. Features Selection through Decision Tree

set.seed(1)
Max_tree <- rpart(target ~ ., data = dataset_sel_scaled, method = "class", cp = 0, minsplit = 1)
rpart.plot(Max_tree, type = 4, extra = 1)  

# number of times each variable has been used in a split
ls(Max_tree)
ls(Max_tree$frame)
a=data.frame(Max_tree$frame$var)
table(a)

# complexity of the tree
ls(Max_tree)
Max_tree$cptable
plotcp(Max_tree)

# variable importance
vi=data.frame(Max_tree$variable.importance)
vi

set.seed(1)  
default.ct <- train(target~ ., data = dataset_sel_scaled, method = "rpart1SE")

Vimportance <- varImp(default.ct)
plot(Vimportance)