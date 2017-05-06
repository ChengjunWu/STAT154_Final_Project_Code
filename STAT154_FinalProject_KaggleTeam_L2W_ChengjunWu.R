 # https://inclass.kaggle.com/c/predictbusinessreviews

################################
# Chengjun Wu                  #
# STAT 154 Final Project RCode #
# Team L2W                     #
################################




library(tm)
library(nnet)
library(e1071)

 # load in data
review_train = read.csv('yelp_academic_dataset_review_train.csv')
review_test = read.csv('yelp_academic_dataset_review_test.csv')
reviews = as.vector(review_train$text)
stopWords = c(stopwords("en"), "") 




#############################A BASIC FIT##########################
library(SnowballC)
cleanCorpus = function(corpus){
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus = tm_map(review_corpus, stemDocument)
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}
cleanCorpus2 = function(corpus){
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

 # training set
set.seed(154)
tst = sample(c(1:116474), size = 23295)
reviews.train0 = review_train[-tst,]
reviews.train = as.vector(reviews.train0$text)

review_corpus.train = cleanCorpus(Corpus(VectorSource(reviews.train)))

review_dtm_train = DocumentTermMatrix(review_corpus.train)
review_dtm_train = removeSparseTerms(review_dtm_train, 0.993)
review_dtm_train = as.matrix(review_dtm_train)

 # testing set
reviews.test0 = review_train[tst,]
reviews.test = as.vector(reviews.test0$text)

review_corpus.test = cleanCorpus(Corpus(VectorSource(reviews.test)))

review_dtm_test = DocumentTermMatrix(review_corpus.test)
review_dtm_test = removeSparseTerms(review_dtm_test, 0.994)
review_dtm_test = as.matrix(review_dtm_test)

cnames = colnames(review_dtm_train)

test.review = subset(review_dtm_test, 
                     select = colnames(review_dtm_test)[colnames(review_dtm_test) %in% cnames])

zero.m = matrix(0, nrow= 23295, ncol = 1)

colnames(zero.m) = cnames[!(cnames %in% colnames(test.review))]

test.review = cbind(test.review, zero.m)

 # combine more variables
review_dtm_train = cbind(review_dtm_train, review.funny = reviews.train0$funny, review.useful = reviews.train0$useful,
                         review.cool = reviews.train0$cool)
test.review = cbind(test.review, review.funny = reviews.test0$funny, review.useful = reviews.test0$useful,
                    review.cool = reviews.test0$cool)
 # fit model
trn.data = as.data.frame(cbind(review_dtm_train, y_train = reviews.train0$stars))
basic_lin_mod = lm(y_train ~ ., data = trn.data)
 # evaluate performance
predicts = predict(basic_lin_mod, as.data.frame(test.review))

submit.lm = as.data.frame(cbind(business_id = as.character(reviews.test0$business_id),
                             predicts = as.numeric(predicts)))
submit.lm = aggregate(as.numeric(as.character(predicts)) ~ business_id, submit.lm, mean)
l5 = which(submit.lm$`as.numeric(as.character(predicts))`>5)
colnames(submit.lm)[2] = "predicts"
submit.lm[l5,"predicts"] = 5
s1 = which(submit.lm$predicts<1)
submit.lm[s1,"predicts"] = 1

true_value = aggregate(stars ~ business_id, reviews.test0, mean)
       # MSE
mean((submit.lm$predicts - true_value$stars)^2)

  # fit the model on the whole dataset
review_corpus = cleanCorpus(Corpus(VectorSource(reviews)))

review_dtm = DocumentTermMatrix(review_corpus)

review_dtm = removeSparseTerms(review_dtm, 0.993)

X_train = as.matrix(review_dtm)

X_train = cbind(X_train, review.funny = review_train$funny, review.useful = review_train$useful,
                         review.cool = review_train$cool)
y_train = review_train$stars
trn.data = as.data.frame(cbind(X_train, y_train))

basic_lin_mod = lm(y_train ~ ., data = trn.data)




 # out of sample fit
review_test = read.csv('yelp_academic_dataset_review_test.csv')
reviews.tt = as.vector(review_test$text)
review_corpus = cleanCorpus(Corpus(VectorSource(reviews.tt)))
review_dtm.tt = DocumentTermMatrix(review_corpus)
review_dtm.tt = removeSparseTerms(review_dtm.tt, 0.999)
review_dtm.tt = as.matrix(review_dtm.tt)

test.matrix = cbind(review_dtm.tt, review.funny = review_test$funny, review.useful = review_test$useful,
                    review.cool = review_test$cool)

cnames = colnames(X_train)

test.matrix = subset(test.matrix, select = colnames(test.matrix)[colnames(test.matrix) %in% cnames])
dim(test.matrix)

preds = as.data.frame(test.matrix)
predicts = predict(basic_lin_mod, preds)

 # make a submission
submit = as.data.frame(cbind(business_id = as.character(review_test$business_id),
                             predicts = as.numeric(predicts)))
submit = aggregate(as.numeric(as.character(predicts)) ~ business_id, submit, mean)

colnames(submit) = c("business_id","stars")

l5 = which(submit$stars>5)
submit[l5,"stars"] = 5
s1 = which(submit$stars<1)
submit[s1,"stars"] = 1

write.csv(submit,"submission.csv")









############OLS########################

review_corpus = cleanCorpus2(Corpus(VectorSource(reviews)))
review_dtm_full = DocumentTermMatrix(review_corpus)
review_dtm_big = removeSparseTerms(review_dtm_full, 0.999)
big.matrix = as.matrix(review_dtm_big)
library(tidytext)
adj = colnames(big.matrix)[colnames(big.matrix) %in% sentiments$word]

 # fit ols
sentiment.matrix = subset(big.matrix, select = adj)


#####wordCloud#######################################################
library(wordcloud)                                                 ##
wordcloud(review_corpus, max.words = 200,                          ##
          random.order = FALSE, colors = brewer.pal(4, "Dark2"))   ##
                                                                   ##
sum.vec = apply(sentiment.matrix, 2, sum)                          ##
g.sum = sum(sum.vec)                                               ##
freq.vec = sum.vec/g.sum                                           ##
word = cbind(colnames(sentiment.matrix),sum.vec)                   ##
wordcloud(words = rownames(word) , freq = freq.vec,                ##
          max.words=250, random.order=FALSE,                       ##
          colors=brewer.pal(8, "Dark2"))                           ##
###################word cloud end####################################


y_train = review_train$stars
trn.data.sentiments = as.data.frame(cbind(sentiment.matrix, y_train))

lin.model = lm(y_train ~ ., data = trn.data.sentiments)

reviews.tt = as.vector(review_test$text)
review_corpus.tt = cleanCorpus2(Corpus(VectorSource(reviews.tt)))
review_dtm.tt = DocumentTermMatrix(review_corpus.tt)

cnames = colnames(sentiment.matrix)

review_dtm.tt = removeSparseTerms(review_dtm.tt, 0.9999)
preds = as.matrix(review_dtm.tt)
preds = subset(preds, select = colnames(preds)[colnames(preds) %in% cnames])
dim(preds)

preds = as.data.frame(preds)
preds = cbind(preds,cosmopolitan = rep(0,21919))
predicts = predict(lin.model, preds)
 # make a submission
submit = as.data.frame(cbind(business_id = as.character(review_test$business_id),
                             predicts = as.numeric(predicts)))
submit = aggregate(as.numeric(as.character(predicts)) ~ business_id, submit, mean)

colnames(submit) = c("business_id","stars")
write.csv(submit,"submission0504.csv")


###############PCA REG#################################
review_dtm_small = removeSparseTerms(review_dtm_full, 0.99)
small.matrix = as.matrix(review_dtm_small)
adj2 = colnames(small.matrix)[colnames(small.matrix) %in% sentiments$word]
sentiment.matrix.small = subset(small.matrix, select = adj2)


library(pls)
pcar = pcr(review_train$stars ~ sentiment.matrix.small, ncomp = 25, validation = "CV")
validationplot(pcar, val.type="MSEP")
summary(pcar)
 # make predictions

cnames = colnames(sentiment.matrix.small)

review_dtm.tt = removeSparseTerms(review_dtm.tt, 0.999)

preds = as.matrix(review_dtm.tt)
preds = subset(preds, select = colnames(preds)[colnames(preds) %in% cnames])
dim(preds)

predicts = predict(pcar, preds)

 # make a submission
submit.pca = as.data.frame(cbind(business_id = as.character(review_test$business_id),
                             predicts = as.numeric(predicts)))
submit.pca = aggregate(as.numeric(as.character(predicts)) ~ business_id, submit.pca, mean)

colnames(submit.pca) = c("business_id","stars")
write.csv(submit.pca,"submission0505pca.csv")


###########RANDOM FOREST##############################
library(randomForest)

ntrees = c(950, 1000, 1050, 1100, 1150)

set.seed(154)
tst = sample(c(1:116474), size = 23295)
reduct = tst
tst = reduct[1:5824]
 # training set
reviews.train0 = review_train[-tst,]
reviews.train = as.vector(reviews.train0$text)

review_corpus.train = cleanCorpus2(Corpus(VectorSource(reviews.train)))

review_dtm_train = DocumentTermMatrix(review_corpus.train)
review_dtm_train = removeSparseTerms(review_dtm_train, 0.99)
review_dtm_train = as.matrix(review_dtm_train)
dim(review_dtm_train)

adj1 = colnames(review_dtm_train)[colnames(review_dtm_train) %in% sentiments$word]

sentiment.matrix.train = subset(review_dtm_train, select = adj1)
dim(sentiment.matrix.train)
 
 # model
library("foreach")
library("doSNOW")
registerDoSNOW(makeCluster(4, type="SOCK"))

rf1 = randomForest(sentiment.matrix.train,as.factor(reviews.train0$stars),ntrees=100)
rf2 = randomForest(sentiment.matrix.train,as.numeric(reviews.train0$stars),ntrees=100)
 # evaluate performance
reviews.test0 = review_train[tst,]
reviews.test = as.vector(reviews.test0$text)

review_corpus.test = cleanCorpus2(Corpus(VectorSource(reviews.test)))

review_dtm_test = DocumentTermMatrix(review_corpus.test)
review_dtm_test = removeSparseTerms(review_dtm_test, 0.99)
review_dtm_test = as.matrix(review_dtm_test)
dim(review_dtm_test)

cnames = colnames(sentiment.matrix.train)

test.review = subset(review_dtm_test, 
                     select = colnames(review_dtm_test)[colnames(review_dtm_test) %in% cnames])

zero.m = matrix(0, nrow= 11648, ncol = 7)

colnames(zero.m) = cnames[!(cnames %in% colnames(test.review))]

test.review = cbind(test.review, zero.m)

preds = predict(rf1, newdata = test.review)
preds2 = predict(rf2, newdata = test.review)

 # MSE for classification
mean((preds - reviews.test0$stars)^2)

 # MSE for regression
mean((preds2 - reviews.test0$stars)^2)


#######################BOOSTING#############################

library(gbm)

# model

tst = sample(c(1:116474), size = 23295)
 # training set
reviews.train0 = review_train[-tst,]
reviews.train = as.vector(reviews.train0$text)

review_corpus.train = cleanCorpus2(Corpus(VectorSource(reviews.train)))

review_dtm_train = DocumentTermMatrix(review_corpus.train)
review_dtm_train = removeSparseTerms(review_dtm_train, 0.999)
review_dtm_train = as.matrix(review_dtm_train)
dim(review_dtm_train)

adj1 = colnames(review_dtm_train)[colnames(review_dtm_train) %in% sentiments$word]

sentiment.matrix.train = subset(review_dtm_train, select = adj1)
dim(sentiment.matrix.train )
 # testing set
reviews.test0 = review_train[tst,]
reviews.test = as.vector(reviews.test0$text)

review_corpus.test = cleanCorpus2(Corpus(VectorSource(reviews.test)))

review_dtm_test = DocumentTermMatrix(review_corpus.test)
review_dtm_test = removeSparseTerms(review_dtm_test, 0.99)
review_dtm_test = as.matrix(review_dtm_test)

cnames = colnames(sentiment.matrix.train)

test.review = subset(review_dtm_test, 
                     select = colnames(review_dtm_test)[colnames(review_dtm_test) %in% cnames])

zero.m = matrix(0, nrow= 23295, ncol = 8)

colnames(zero.m) = cnames[!(cnames %in% colnames(test.review))]

test.review = cbind(test.review, zero.m)

trn.data = as.data.frame(cbind(sentiment.matrix.train, resp = reviews.train0$stars))

trn.data = na.omit(trn.data)


boosting = gbm(resp ~., data=trn.data, n.trees = 100,
               interaction.depth =2, shrinkage = 0.01, verbose = FALSE)

boosting.pred = predict(boosting, as.data.frame(test.review), type = "response", n.tree = 100)


# MSE for regression
submit.b = as.data.frame(cbind(business_id = as.character(reviews.test0$business_id),
                                 predicts = as.numeric(boosting.pred)))
submit.b = aggregate(as.numeric(as.character(boosting.pred)) ~ business_id, submit.b, mean)
colnames(submit.b)[2] = "predicts"
true_value = aggregate(stars ~ business_id, reviews.test0, mean)

MSE.b = mean((submit.b$predicts - true_value$stars)^2)
MSE.b #0.8431929







