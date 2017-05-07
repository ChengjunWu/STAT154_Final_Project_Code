set.seed(154)
library(tm)
library(nnet)
library(e1071)

############
review_train = read.csv('yelp_academic_dataset_review_train.csv')
review_test = read.csv('yelp_academic_dataset_review_test.csv')
#############
reviews = as.vector(review_train$text)
reviews.tt = as.vector(review_test$text)
stopWords = c(stopwords("en"), "") 


cleanCorpus = function(corpus){
  # You can also use this function instead of the first. 
  # Here you clean all the reviews at once using the 
  # 'tm' package. Again, a lot more you can add to this function...
  
  review_corpus = tm_map(corpus, content_transformer(tolower))
  review_corpus = tm_map(review_corpus, removeNumbers)
  review_corpus = tm_map(review_corpus, removePunctuation)
  review_corpus = tm_map(review_corpus, removeWords, c("the", "and", stopwords("english")))
  review_corpus =  tm_map(review_corpus, stripWhitespace)
}

library(SnowballC)
review_corpus = cleanCorpus(Corpus(VectorSource(reviews)))
review_corpus.tt = cleanCorpus(Corpus(VectorSource(reviews.tt)))

review_dtm = DocumentTermMatrix(review_corpus)
review_dtm.tt = DocumentTermMatrix(review_corpus.tt)

review_dtm = removeSparseTerms(review_dtm, 0.999)


# install.packages("tidytext")
library(tidytext)
big.matrix = as.matrix(review_dtm)

adj = colnames(big.matrix)[colnames(big.matrix) %in% sentiments$word]

X_train = big.matrix[,colnames(big.matrix) %in% adj]

y_train = review_train$stars

trn.data = data.frame(y_train, X_train)


pca = prcomp(X_train)
# head(summary(pca))
screeplot(pca, type = "l", main = "Scree Plot")
# summary(pca)

firstPC = pca$x[,1]
secondPC<-pca$x[,2]
plot(firstPC,secondPC,pch=20,main="PCA view of my simple Data", col = grey)
# c("red","blue","green", "yellow", "black")[y_train]

text(firstPC,secondPC,labels = y_train, cex = 0.7)






library("factoextra")
eig.val <- get_eigenvalue(pca)
head(eig.val,15)
head(eig.val,500)[200:250,]


library(kernlab)
kpca<-kpca(X_train,kernel = "rbfdot", kpar = list(sigma = 0.1),
                        features = 2, th = 1e-4)
