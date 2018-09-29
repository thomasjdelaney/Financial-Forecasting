if (!require(devtools) ) install.packages("devtools")
if (!require(M4comp2018) )  devtools::install_github("carlanetto/M4comp2018")
if (!require(M4metalearning)) {
  devtools::install_github("robjhyndman/M4metalearning")
  devtools::install_github("pmontman/tsfeatures")
  devtools::install_github("pmontman/customxgboost")
}
if (!require(rbenchmark)) install.packages("rbenchmark")

library(M4metalearning)
library(M4comp2018)
library(rbenchmark)

# Isolating financial series only ####
isfin <- sapply(M4, function(time_series) time_series$type == "Finance")
fin_inds <- which(isfin)
num_fin_time_series <- length(fin_inds)
fin_series <- M4[fin_inds]

# creating the indices and the chunk ####
set.seed(1202)
indices <- sample(num_fin_time_series)
train_chunk_1_inds <- indices[1:3333]
train_chunk_2_inds <- indices[3334:6666]
train_chunk_3_inds <- indices[6667:10000]

# selecting a chunk for this script ####
train_chunk_inds <- train_chunk_3_inds

# creating the training data chunk ####
subdivision_length <- 50
divided_chunk_inds <- split(train_chunk_inds, ceiling(seq_along(train_chunk_inds)/subdivision_length))
num_subdivisions <- length(divided_chunk_inds)
message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " subdivisions to process: ", num_subdivisions)

trained_chunk = list()
for (i in 1:num_subdivisions){
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " processing subdivision ", i, "...")
  subdiv_inds <- divided_chunk_inds[[i]]
  subdiv_series <- fin_series[subdiv_inds]
  subdiv_series <- temp_holdout(subdiv_series)
  subdiv_series <- calc_forecasts(subdiv_series, forec_methods(), n.cores = 6) # TODO: more cores.
  subdiv_series <- calc_errors(subdiv_series)
  subdiv_series <- THA_features(subdiv_series)
  trained_chunk <- c(trained_chunk, subdiv_series)
}
message(format(Sys.time(), "%Y-%m-%d %H:%M:%OS3"), " Done. Saving...")

# saving the chunk for this script ####
saveRDS(trained_chunk, "./rds/train_chunk_3.rds")
