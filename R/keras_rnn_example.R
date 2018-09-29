library(keras)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32)
summary(model)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE)
summary(model)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32)  # This last layer only returns the last outputs.
summary(model)

max_features <- 10000
maxlen <- 500  # Cuts off texts after this many words (among the max_features most common words)
batch_size <- 32
cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb 
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")

cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen = maxlen)
input_test <- pad_sequences(input_test, maxlen = maxlen)
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")

compile(model, optimizer = "rmsprop", loss = "binary_crossentropy", metrics = c("acc"))

history <- fit(model, input_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")

compile(model, optimizer = "rmsprop", loss = "binary_crossentropy", metrics = c("acc"))

history <- fit(model, input_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)