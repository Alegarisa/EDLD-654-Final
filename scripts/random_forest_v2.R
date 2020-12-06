### loading libraries ###
library(rio)
#library(here)

library(tidyverse)
library(tidymodels)

#library(vip)
#library(rpart.plot)


options(scipen=999)
###############################################################################

### loading data ###

# sample a fraction, but use HPC to run model with all data.
set.seed(3000)

full_train <- import("data/train.csv", setclass = "tbl_df") %>%
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch)) %>%
  sample_frac(0.5)

bonus <- import("data/bonus_data.csv") %>%
  mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
  mutate(ncessch = as.double(ncessch))

## joining data
data <- left_join(full_train, bonus) # check to use data instead of d
############################################################################### 

### Split & resample ###
set.seed(3000)
data_split <- initial_split(data)

set.seed(3000)
data_train <- training(data_split)
data_test <- testing(data_split)

# set.seed(3000)
# cv <- vfold_cv(data_train)
###############################################################################

### Preprocess ###
rec <- recipe(score ~ ., data_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
              pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
  update_role(contains("id"), ncessch, ncesag, sch_name, new_role = "id_vars") %>%
  step_zv(all_predictors(), -starts_with("lang_cd")) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_novel(all_nominal()) %>%
  step_unknown(all_nominal()) %>% 
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors(), -starts_with("lang_cd"))

prepped_rec <- prep(rec)
prepped_rec
 
# not used:
# step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>% why removing the id vars?
# step_knnimpute(all_numeric()) %>% too computationally intensive 

###############################################################################

### Building the Random Forest model - default ###

# mtry = floor(p/3) 77/3 = 26
# trees = 500 (num.trees)
# min_n = 5 (min.node.size)
# what about this: respect.unordered.factors = "order"?

# sum(rec$var_info$role == "predictor") # 77 predictors

#(cores <- parallel:: detectCores())

rf_def <- rand_forest() %>%
  set_engine("ranger",
             num.threads = 8,
             importance = "permutation",
             verbose = TRUE) %>%
  set_mode("regression")

#translate(rf_def) 

### workflow for default model
rf_def_wkflw <- workflow() %>% 
  add_model(rf_def) %>% 
  add_recipe(rec)

### Fitting the default model ###

# tictoc::tic()
# set.seed(3000)
# rf_def_fit <- fit(
#   rf_def_wkflw, 
#   data_train)
# tictoc::toc()
# this took 147.967 sec ---> less than 3 minutes

### Best estimate of Random Forest default model ###

# assumes you feed it the workflow fit like rf_def_fit above
extract_rmse <- function(wf_fit) {
  sqrt(wf_fit$fit$fit$fit$prediction.error)
}

# extract_rmse(rf_def_fit) 

# with mtry = 26, min_n = 5, trees = 500:
# rmse = 95.08407 (5% of data) 
# rmse = 88.80172 (50% of data) 

##################### Hyperparameter Tuning #################################
# "Start with five evenly spaced values of mtry across the range 2– p centered at the recommended default"
# 77/3 # aprox. 26

# "When adjusting node size start with three values between 1–10 and adjust depending on impact to accuracy and run time."

# "A good rule of thumb is to start with 10 times the number of features" 
# 77*10 # 770. We'll go for 1000 trees. 

# mtry_search <- seq(2, 50, 12) # using the suggestion in book
# min_n_search <- seq(1, 10, 3) # using the suggestion in book
# 
# grd <- expand.grid(mtry_search, min_n_search)
# 
# hyp_rf_search <- function(mtry_val, min_n_val, wf) {
#   mod <- rand_forest() %>% 
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>% 
#     set_mode("regression") %>% 
#     set_args(mtry = {{mtry_val}},
#              min_n = {{min_n_val}},
#              trees = 1000)
#   
#   wf <- wf %>% # shouldn't I be using rf_def_wkflw here??
#     update_model(mod)
#   
#   rmse <- fit(wf, data_train) %>% 
#     extract_rmse()
#   
#   tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
# }

# tictoc::tic()
# mtry_results_1 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
# tictoc::toc()
# this took 6392.721 sec ----> around an hour and 40 minutes
# 
# saveRDS(mtry_results_1, "mtry_results_1.Rds")

# mtry_results_1 <- readRDS("scripts/mtry_results_1.Rds")
# 
# mtry_results_1 %>%
#   arrange(rmse) 

# rmse = 93.6 (5% of data) with mtry = 14 or 26 & min_n = 10 
# rmse = 88.5 (50% of data) with mtry = 14, min_n = 10, trees = 1000

### plot 1###
# mtry_results_1 %>%
#   ggplot(aes(mtry, rmse)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~min_n)

# #################
# #################
# 
# ##### TUNING 2 #####
# # "However, Segal (2004) showed that if your data has many noisy predictors and higher mtry values are performing best, then performance may improve by increasing node size (i.e., decreasing tree depth and complexity)."
# 
# mtry_search <- seq(14, 26, 3)
# min_n_search <- seq(7, 15, 2)
# 
# grd <- expand.grid(mtry_search, min_n_search)
# 
# hyp_rf_search <- function(mtry_val, min_n_val, wf) {
#   mod <- rand_forest() %>%
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>%
#     set_mode("regression") %>%
#     set_args(mtry = {{mtry_val}},
#              min_n = {{min_n_val}},
#              trees = 1000)
# 
#   wf <- wf %>%
#     update_model(mod)
# 
#   rmse <- fit(wf, data_train) %>%
#     extract_rmse()
# 
#   tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
# }

# tictoc::tic()
# mtry_results_2 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
# tictoc::toc() 
# this took 6862.319 sec ---> almost 2 hours to run

# saveRDS(mtry_results_2, "mtry_results_2.Rds")

# mtry_results_2 <- readRDS("scripts/mtry_results_2.Rds")
# 
# mtry_results_2 %>%
#   arrange(rmse)

# rmse = 92.4 (5% of data) with mtry = 20 - 26 & min_n = 19 for an rmse of
# rmse = 88.2 (50% of data) with mtry = 14, min_n = 15, trees = 1000


### plot 2 ###
# mtry_results_2 %>%
#   ggplot(aes(mtry, rmse)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~min_n)

#################
#################

# ##### TUNING 3 #####
# 
# mtry_search <- seq(20, 26, 2)
# min_n_search <- seq(19, 30, 3)
# 
# grd <- expand.grid(mtry_search, min_n_search)
# 
# hyp_rf_search <- function(mtry_val, min_n_val, wf) {
#   mod <- rand_forest() %>% 
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>% 
#     set_mode("regression") %>% 
#     set_args(mtry = {{mtry_val}},
#              min_n = {{min_n_val}},
#              trees = 1000)
#   
#   wf <- wf %>% 
#     update_model(mod)
#   
#   rmse <- fit(wf, data_train) %>% 
#     extract_rmse()
#   
#   tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
# }
# 
# tictoc::tic()
# mtry_results_3 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
# tictoc::toc()
# 
# mtry_results_3 %>% 
#   arrange(rmse) # Best mtry = .. & min_n = 28 for an rmse of ...
# 
# ### plot 3 ###
# mtry_results_3 %>% 
#   ggplot(aes(mtry, rmse)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~min_n)
# 
# #################
# #################
# 
# ##### TUNING 4 #####
# 
# mtry_search <- seq(22, 24, 1)
# min_n_search <- seq(30, 50, 2)
# 
# grd <- expand.grid(mtry_search, min_n_search)
# 
# hyp_rf_search <- function(mtry_val, min_n_val, wf) {
#   mod <- rand_forest() %>% 
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>% 
#     set_mode("regression") %>% 
#     set_args(mtry = {{mtry_val}},
#              min_n = {{min_n_val}},
#              trees = 1000)
#   
#   wf <- wf %>% 
#     update_model(mod)
#   
#   rmse <- fit(wf, data_train) %>% 
#     extract_rmse()
#   
#   tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
# }
# 
# tictoc::tic()
# mtry_results_4 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
# tictoc::toc()
# 
# mtry_results_4 %>% 
#   arrange(rmse) # Best mtry = .. & min_n = ... for an rmse of ...
# 
# ### plot 4 ###
# mtry_results_4 %>% 
#   ggplot(aes(mtry, rmse)) +
#   geom_line() +
#   geom_point() +
#   facet_wrap(~min_n)
# 
# #################
# #################
# 
# ##### TUNING 5 #####
# 
# mtry_search <- 24
# min_n_search <- seq(38, 50, 2)
# 
# grd <- expand.grid(mtry_search, min_n_search)
# 
# hyp_rf_search <- function(mtry_val, min_n_val, wf) {
#   mod <- rand_forest() %>% 
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>% 
#     set_mode("regression") %>% 
#     set_args(mtry = {{mtry_val}},
#              min_n = {{min_n_val}},
#              trees = 1000)
#   
#   wf <- wf %>% 
#     update_model(mod)
#   
#   rmse <- fit(wf, data_train) %>% 
#     extract_rmse()
#   
#   tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
# }
# 
# tictoc::tic()
# mtry_results_5 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
# tictoc::toc()
# 
# mtry_results_5 %>% 
#   arrange(rmse) # Best mtry = .. & min_n = ... for an rmse of ...
# 
# ### plot 5 ###
# mtry_results_5 %>% 
#   ggplot(aes(min_n, rmse)) +
#   geom_line() +
#   geom_point() 
# 
# # the best hyperparameters are mtry = 24, min_n = 50. I think could keep incresing min_n, but not sure if it's "allowed" or worth it
# 
# prepped_rec <- rec %>%
#   prep %>%
#   bake(data_train)
# 
# tune_min_n <- function(n) {
#   mod <- rand_forest() %>% 
#     set_mode("regression") %>% 
#     set_engine("ranger",
#                num.threads = 8,
#                importance = "permutation",
#                verbose = TRUE) %>% 
#     set_args(mtry = 24,
#              min_n = n,
#              trees = 1000)
#   # fit model to full training dataset
#   m <- fit(mod, score ~ ., prepped_rec)
#   # Extract RMSE, store as a tibble
#   tibble(rmse = sqrt(m$fit$prediction.error))
# }
# 
# tictoc::tic()
# optimal_n <- map_df(seq(2, 150, 2), tune_min_n) 
# tictoc::toc()
# 
# optimal_n %>% 
#   mutate(n = seq(2, 150, 2)) %>% 
#   ggplot(aes(n, rmse)) +
#   geom_line() +
#   geom_point()
# 
# # --> not worth it: It took over 45 minutes to run. the plot shows that around 50 it's 90.5 and around 100 is 90. After a hundred it levels out. 
# ##############################################
# 


### summary of models ###

# rmse = 88.8 (50% of data) with mtry = 26, min_n = 5, trees = 500

# rmse = 88.5 (50% of data) with mtry = 14, min_n = 10, trees = 1000

# rmse = 88.2 (50% of data) with mtry = 14, min_n = 15, trees = 1000

# because the gain is not much, I will hard code mtry = 14 and min_n = 15 in final model.

##### final fit #####

final_mod <- rand_forest() %>%
  set_engine("ranger",
             num.threads = 8,
             importance = "permutation",
             verbose = TRUE) %>%
  set_mode("regression") %>%
  set_args(mtry = 14,
           min_n = 15,
           trees = 1000)

final_wkfl <-  workflow() %>%
  add_model(final_mod) %>%
  add_recipe(rec)

tictoc::tic()
check_fit <- fit(final_wkfl,
                 data = data_train)
tictoc::toc()

extract_rmse(check_fit) 
# rmse = 90.86 (with data_train, 5% of data)


tictoc::tic()
final_fit <- last_fit(final_wkfl,
                 split = data_split)
tictoc::toc()

final_fit$.metrics[[1]] 
# rmse = 93.0 (with data_split, 5% of data)


##### final fit #####

### to submit to Kaggle ####
full_test <- import("data/test.csv", setclass = "tbl_df") %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch))

## joining data
all_test <- left_join(full_test, bonus)

# baking the recipe
processed_test <- rec %>%
  prep() %>% 
  bake(all_test)

# make predictions
preds <- predict(check_fit, new_data = processed_test)

# a tibble
pred_frame <- tibble(Id = all_test$id, Predicted = preds$.pred)

# create file for upload 
write_csv(pred_frame, "prelim_fit_2.csv")