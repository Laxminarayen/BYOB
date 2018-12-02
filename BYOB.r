#creating new folder only with the pdf's
setwd("E://pdf")
set.seed("2496")
library(tesseract)
library(pdftools)
library(caret)
library(magrittr)
library(stats)
library(magick)
#install.packages("text2vec")
library(text2vec)
library(stringi)
library(stringr)
library(purrrlyr)
library(Rcpp)
#install.packages("XML")
library(XML)
#install.packages("htmltidy")
library(htmltidy)
library(tidyverse)
library(stringi)
#install.packages("xgboost")
library(xgboost)
library(magrittr)
library(rvest)
library(httr)
#install.packages("onehot")
library(onehot)
#importing already loaded pdfs from the current working directory
files <- list.files(pattern = "*.pdf")
View(files)
L = length(files)
converted_Png <- list()
convert <- list() 
#Loop to convert the pdf into png 
for(i in 1:L){
  converted_Png <- pdf_convert(files[i],dpi = 600)
  convert[[i]] <- converted_Png
  converted_Png <- NULL
}
View(convert)
text_converted <- list()
tconvert <- list()
file2 <- list.files(pattern = "*.png")
View(file2)
#Loop to convert the images to texts using tesseract ocr
for(i in 1:length(file2)){
  tconvert <- ocr(file2[i],engine = tesseract("eng"))
  text_converted[[i]] <- tconvert
  tconvert <- NULL
}
result1 <- list()
result24 <- data.frame()
#Loop to split the converted text Line by line
for(i in 1:length(file2)){
  result1[i] <- strsplit(text_converted[[i]],split = "\n")
}
library(data.table)
#using library data.table to append the result list into a dataframe 
result24 = rbindlist(Map(as.data.frame, result1))
write.csv(result24,"Line_Items.csv")
#reading the csv in order to replace the empty spaces by NA
A <- read.csv("Line_Items.csv", na.strings = c("","NA"))
#function to omit all the blank spaces in the csv file
A <- na.omit(A)
View(A)
A <- (A[,-1]) 
A <- data.frame(A)
colnames(A) <- c("Line_Items")
write.csv(A,"Line_Label.csv")
#keras in R interface
install.packages("keras")
library(keras)
install_keras()
#With given time constraints and limited resources we could manually encode labels in train
#for only 3412 rows  
A <- read.csv("E:\\File.csv")
#creating a new column as duplicate so that if necessary cash figures could be referrenced from this col
A["Line_Items_Dup"] = A$Line_Items
A["Labels_dup" ] <- A$Labels
A$Labels_dup <- as.numeric(A$Labels_dup)
A["Labels_dup" ] <- A$Labels
train = A[1:3411,]
test <- A[3412:17097,]
#converting symbols to characters
A$Line_Items_Dup <- gsub("\\£", "Pound",A$Line_Items_Dup) 
A$Line_Items_Dup <- gsub(",", "Comma",A$Line_Items_Dup)  
#converting numbers to string "Numbers"
for(i in 0:9){
  A$Line_Items_Dup <- gsub( i, " Numbers ",A$Line_Items_Dup)
}
#tokenizing the string
prep_fun <- tolower
tok_fun <- text_tokenizer(num_words = 40,
                          filters = "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n", lower = TRUE,
                          split = " ", char_level = TRUE, oov_token = NULL)
tok_fun<- fit_text_tokenizer(tok_fun,A$Line_Items_Dup)
it_complete <- itoken(A$Line_Items_Dup, 
                      preprocessor = prep_fun, 
                      tokenizer = tok_fun,
                      progressbar = TRUE)
sequencecomplete <- texts_to_sequences(tok_fun, A$Line_Items_Dup)
completeddata <- pad_sequences(
  sequencecomplete
)
#performing one hot encoding on the labels
one_hot_model <- onehot(train[,c(2,4)],max_levels = 50)
predictdataframe <- predict(one_hot_model,train)
predictdataframe <- predictdataframe[,-c(39)]
sequence_Target <- as.matrix(predictdataframe)
#Splitting actual train and test from tokenized text
x_train <- completeddata[tri,]
x_test <- completeddata[-tri,]
#Looking at the dimensions of x for train and test
cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')
#defining the model
model <- keras_model_sequential() 
model %>%
  # Creates dense embedding layer; outputs 3D tensor
  # with shape (batch_size, sequence_length, output_dim)
  layer_embedding(input_dim = ncol(x_train), 
                  output_dim = 208, 
                  input_length = 416) %>%
  layer_dropout(rate = 0.5) %>%
  layer_lstm(units = 208) %>%
  layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = 38, activation = 'sigmoid')
model %>%  compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)
model %>%keras::fit(
  x_train,sequence_Target ,epochs = 10, batch_size = 256,
  callbacks = callback_early_stopping(patience = 3, monitor = 'acc'),
  validation_split = 0.3
)
#predicting the probablity for the labels from the model
test$Labels <- predict_proba(model,x_test, batch_size = 32, verbose = 1)
write.csv(test,"Predicted.csv")
