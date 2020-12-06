library(tidyverse)
library(tidymodels)
library(xgboost)


baked_train <- read.csv("data/baked_train.csv") %>% 
  select(score, everything(), -X)

train_x = data.matrix(baked_train[, -1])
train_y = data.matrix(baked_train[, 1])
# test_x = data.matrix(baked_test[, -73])
# test_y = data.matrix(baked_test[, 73])

## set xgb matrices
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
# xgb_test = xgb.DMatrix(data = test_x, label = test_y)

## first xgb model

def_mod <- xgb.cv(
  data = train_x,
  label = train_y,
  nrounds = 700,
  objective = "reg:squarederror",
  early_stopping_rounds = 50,
  nfold = 10,
  verbose = 1,
  eval_metric = "rmse",
  params = list(
    eta = .05,
    gamma = 10,
    subsample = 0.9850567,
    colsample_bytree = 0.6432432,
    
    nthread = 24
  ))
# 0.9850567
saveRDS(def_mod, "knn_mod.Rds")
# # importance_matrix <- xgb.importance(model = bst)
# print(importance_matrix)
# xgb.plot.importance(importance_matrix = importance_matrix)
# 
bst <- xgboost(data = train_x,
                 label = train_y,
                 nrounds = 477,
                 verbose = 1,
                   params = list(
                     eta = .05,
                     gamma = 10,
                     subsample = 0.9850567,
                     colsample_bytree = 0.6432432,
                     nthread = 24
                   ))

saveRDS(bst, "knn_model_submit.Rds")
# xgb_
# test_x = data.matrix(baked_test[, -69])
# test_y = data.matrix(baked_test[, 69])
# 
# pred <- predict(model_submit, as.matrix(baked_test))
# 
# 
# actual <- baked_test$score
# id <- test$id
# 
# names(baked_train)
# predictions <- tibble("Id" = id, "Predicted" = pred)
# write_csv(predictions, "initial_predictions.csv")
# ?as_tibble
# tib
# actual <- test_y
#
# Metrics::rmse(actual, pred)
# 
# # def_mod$evaluation_log[def_mod$best_iteration, ]
# # # 
# saveRDS(def_mod, "def_mod_v3.Rds")
# 
# pull_eval <- function(m) {
#   m[["evaluation_log"]] %>%
#     pivot_longer(-iter,
#                  names_to = c("set", NA, "stat"),
#                  names_sep = "_",
#                  values_to = "val") %>%
#     pivot_wider(names_from = "stat",
#                 values_from = "val")
# }

# def_mod %>%
#   pull_eval() %>%
#   filter(iter > 7) %>%
#   ggplot(aes(iter, mean, color = set)) +
#   geom_line() +
#   geom_point()

# ######### Train learning rate 
# lr <- seq(0.0001, 0.3, length.out = 30)
# 
# lr_mods <- map(lr, function(learn_rate) {
#   xgb.cv(
#     data = train_x,
#     label = train_y,
#     nrounds = 5000,
#     objective = "reg:squarederror",
#     early_stopping_rounds = 50, 
#     nfold = 10,
#     verbose = 0,
#     params = list( 
#       eta = learn_rate,
#       nthread = 20
#     ) 
#   )  
# }) 
# 
# saveRDS(lr_mods, "lr_mods.Rds")
# 
# names(lr_mods) <- lr
# evals_lr <- map_df(lr_mods, pull_eval, .id = "learning_rate")
# 
# rmse_lr <- evals_lr %>% 
#   group_by(learning_rate, set) %>% 
#   summarize(min = min(mean)) %>% 
#   pivot_wider(names_from = set, values_from = min) %>% 
#   arrange(test)
# 
# rmse_lr %>% 
#   ungroup() %>% 
#   mutate(learning_rate = as.numeric(learning_rate)) %>% 
#   filter(test < 120) %>% 
#   ggplot(aes(learning_rate, test)) +
#   geom_point()
# 
# # Check learning curves
# lr_mods[[rmse_lr$learning_rate[1]]] %>% 
#   pull_eval() %>% 
#   filter(mean < 120) %>% 
#   ggplot(aes(iter, mean, color = set)) +
#   geom_line() +
#   geom_point()
# 

pred <- predict(bst, as.matrix(test_x))
id <- test$id

names(baked_train)
predictions <- tibble("Id" = id, "Predicted" = pred)
write_csv(predictions, "experimental predictions.csv")
?as_tibble
tib
actual <- test_y
#
Metrics::rmse(actual, pred)
