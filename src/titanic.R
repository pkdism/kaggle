library(dplyr)
library(corrplot)
library(ggfortify)
library(tabplot)
library(stringr)
require(randomForest)
library(Boruta)
#Load data
setwd("~/Desktop/programs/kaggle/src/")
train <- read.csv("../data/titanic/train.csv")
test <- read.csv("../data/titanic/test.csv")
train[is.na(train)] = 0
test[is.na(test)] = 0
summary(train)
numeric_columns <- sapply(train, is.numeric)
train.numeric <- train[,c(numeric_columns)]

#Correlations
train.corr <- cor(train.numeric)
corrplot(train.corr, method = "circle", type = "lower", tl.cex = 0.5, title = "Train")
plot(density(train.numeric$Age))
lines(density(train.numeric$Survived))
densityplot(~Age + Survived, data = train.numeric)
ggplot(train, aes(x = Age, group = Survived, fill = Survived)) + geom_histogram(position="dodge",binwidth=0.25)+theme_bw()
ggplot(train, aes(x = Age, group = Survived, color = Survived)) + geom_density()
ggplot(train, aes(x = Fare, group = Survived, fill = Survived)) + geom_histogram(position="dodge",binwidth=0.25)+theme_bw()
ggplot(train, aes(x = Fare, group = Survived, color = Survived)) + geom_density()

tableplot(train[,c("Embarked", "Survived")], sortCol = 2)
tableplot(train[,c("Survived", "Sex", "Pclass", "SibSp", "Parch")], sortCol = 1)

#Data preparation and cleaning
train$Sex = ifelse(train$Sex=="male",1,0)
train$Pclass1 = ifelse(train$Pclass==1,1,0)
train$Pclass2 = ifelse(train$Pclass==2,1,0)
train$Pclass3 = ifelse(train$Pclass==3,1,0)
train$EmbarkedS = ifelse(train$Embarked=="S",1,0)
train$EmbarkedC = ifelse(train$Embarked=="C",1,0)
train$EmbarkedQ = ifelse(train$Embarked=="Q",1,0)
titles <- "(Col.| Major.| Don.| Sir.| Dr.| Rev.| Mme.| Mlle.| Ms.| Mr.| Mrs.| Miss.| Master.| Lady.| Dona| Coll| Colb)"
train$title <- str_extract(string = train$Name, pattern = titles)
train$cabin_class <- substr(train$Cabin,1,1)
train$ticket_class <- sub('.',"",train$Ticket)
train$ticket_class <- sub("/","",train$ticket_class)
train$family_size <- train$SibSp + train$Parch + 1
train$single <- ifelse(train$family_size==1,1,0)
train$small_family <- ifelse(train$family_size >= 2 & train$family_size <= 4,1,0)
train$large_family <- ifelse(train$family_size >= 5,1,0)

test$Sex = ifelse(test$Sex=="male",1,0)
test$Pclass1 = ifelse(test$Pclass==1,1,0)
test$Pclass2 = ifelse(test$Pclass==2,1,0)
test$Pclass3 = ifelse(test$Pclass==3,1,0)
test$EmbarkedS = ifelse(test$Embarked=="S",1,0)
test$EmbarkedC = ifelse(test$Embarked=="C",1,0)
test$EmbarkedQ = ifelse(test$Embarked=="Q",1,0)
test$title <- str_extract(string = test$Name, pattern = titles)
test$cabin_class <- substr(test$Cabin,1,1)
test$ticket_class <- sub('.',"",test$Ticket)
test$ticket_class <- sub("/","",test$ticket_class)
test$family_size <- test$SibSp + test$Parch + 1
test$single <- ifelse(test$family_size==1,1,0)
test$small_family <- ifelse(test$family_size >= 2 & test$family_size <= 4,1,0)
test$large_family <- ifelse(test$family_size >= 5,1,0)

#Split training set into train and test to check accuracy
rows <- nrow(train)
train.test <- train[as.numeric(rows*0.8):rows,]
train.train <- train[1:as.numeric(rows*0.8)-1,]

imp_features <- Boruta(train.train, train.train$Survived)
final_decision <- as.data.frame(imp_features$finalDecision)

train.glm <- glm(Survived ~ Pclass1+Pclass2+Pclass3+EmbarkedQ+EmbarkedC+EmbarkedS+Sex+Age+cabin_class+title+family_size+single+small_family+large_family+Fare+Parch+SibSp, data = train.train)

result <- as.data.frame(predict(train.glm, train.test, type = "response"))
result <- ifelse(result>0.50,1,0)
x = table(result,train.test$Survived)
x
cat(sprintf("accuracy = %f", sum(diag(x))/sum(x)))

#Final model
train.glm <- glm(Survived ~ Pclass1+Pclass2+Pclass3+EmbarkedQ+EmbarkedC+EmbarkedS+Sex+Age+cabin_class+title+family_size+single+small_family+large_family+Fare+Parch+SibSp, data = train)

test$title <- gsub("Dona",NA,test$title)
test$title <- ifelse(test$title=="Dona",NA,test$title)
test$title <- ifelse(test$title=="Coll",NA,test$title)
test$title <- ifelse(test$title=="Colb",NA,test$title)

result <- as.data.frame(predict(train.glm, test, type = "response"))
result <- as.data.frame(ifelse(result>0.50,1,0))
colnames(result) <- c("Survived")
result$PassengerId <- test$PassengerId
# result$Survived = ifelse(result$Survived >= 0.6, 1, ifelse(result$Survived <= 0.4,0,0.5))
result <- result[,c(2,1)]
# write.csv(result, file = "../output/titanic2.csv", row.names = FALSE)
