### loading libraries ###
library(rio)
library(here)

library(tidyverse)
library(tidymodels)

library(vip)
library(rpart.plot)


options(scipen=999)
###############################################################################

### loading data ###

# sample a fraction, but use HPC to run model with all data.
set.seed(3000)

full_train <- import(here("data", "train.csv"), setclass = "tbl_df") %>%
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch)) %>%
  sample_frac(0.05)

bonus <- import(here("data", "bonus_data.csv")) %>%
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

set.seed(3000)
cv <- vfold_cv(data_train)
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

# mtry = floor(p/3)
# trees = 500 (num.trees)
# min_n = 5 (min.node.size)
# what about this: respect.unordered.factors = "order"?

sum(rec$var_info$role == "predictor") # 77 predictors

(cores <- parallel:: detectCores())

rf_def <- rand_forest() %>% 
  set_engine("ranger",
             num.threads = cores,
             importance = "permutation",
             verbose = TRUE) %>% 
  set_mode("regression")  

translate(rf_def) 

### workflow for default model
rf_def_wkflw <- workflow() %>% 
  add_model(rf_def) %>% 
  add_recipe(rec)

### Fitting the default model ###

# using resamples

# metrics_eval <- metric_set(rmse, 
#                            rsq, 
#                            huber_loss)
# tictoc::tic()
# set.seed(3000)
# rf_def_fit <- fit_resamples(
#   rf_def_wkflw, 
#   cv_splits, 
#   metrics = metrics_eval,
#   control = control_resamples(verbose = TRUE,
#                               save_pred = TRUE,
#                               extract = function(x) x))
# tictoc::toc() 

tictoc::tic()
set.seed(3000)
rf_def_fit <- fit(
  rf_def_wkflw, 
  data_train)
tictoc::toc()

### Best estimate of Random Forest default model ###

# assumes you feed it the workflow fit like rf_def_fit above
extract_rmse <- function(wf_fit) {
  sqrt(wf_fit$fit$fit$fit$prediction.error)
}

extract_rmse(rf_def_fit) # rmse = 95.08407

##################### Hyperparameter Tuning #################################
# "Start with five evenly spaced values of mtry across the range 2– p centered at the recommended default"
77/3 # aprox. 26

# "When adjusting node size start with three values between 1–10 and adjust depending on impact to accuracy and run time."

# "A good rule of thumb is to start with 10 times the number of features" 
77*10 # 770. We'll go for 1000 trees. 

mtry_search <- seq(2, 50, 12) # using the suggestion in book
min_n_search <- seq(1, 10, 3) # using the suggestion in book

grd <- expand.grid(mtry_search, min_n_search)

hyp_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}},
             trees = 1000)
  
  wf <- wf %>% # shouldn't I be using rf_def_wkflw here??
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

tictoc::tic()
mtry_results_1 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
tictoc::toc()

mtry_results_1 %>% 
  arrange(rmse) # Best mtry = 10 & min_n = 14 or 26 for an rmse of 93.6

### plot 1###
mtry_results_1 %>% 
  ggplot(aes(mtry, rmse)) +
  geom_line() +
  geom_point() +
  facet_wrap(~min_n)

#################
#################

##### TUNING 2 #####
# "However, Segal (2004) showed that if your data has many noisy predictors and higher mtry values are performing best, then performance may improve by increasing node size (i.e., decreasing tree depth and complexity)."

mtry_search <- seq(14, 26, 3)
min_n_search <- seq(10, 20, 3)

grd <- expand.grid(mtry_search, min_n_search)

hyp_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}},
             trees = 1000)
  
  wf <- wf %>% 
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

tictoc::tic()
mtry_results_2 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
tictoc::toc()

mtry_results_2 %>% 
  arrange(rmse) # Best mtry = 19 & min_n = 20 - 26 for an rmse of 92.4. 

# this tells me that mtry is getting stable, but rmse appears to get better as min_n  gets higher. 

### plot 2 ###
mtry_results_2 %>% 
  ggplot(aes(mtry, rmse)) +
  geom_line() +
  geom_point() +
  facet_wrap(~min_n)

#################
#################

##### TUNING 3 #####

mtry_search <- seq(20, 26, 2)
min_n_search <- seq(19, 30, 3)

grd <- expand.grid(mtry_search, min_n_search)

hyp_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}},
             trees = 1000)
  
  wf <- wf %>% 
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

tictoc::tic()
mtry_results_3 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
tictoc::toc()

mtry_results_3 %>% 
  arrange(rmse) # Best mtry = .. & min_n = 28 for an rmse of ...

### plot 3 ###
mtry_results_3 %>% 
  ggplot(aes(mtry, rmse)) +
  geom_line() +
  geom_point() +
  facet_wrap(~min_n)

#################
#################

##### TUNING 4 #####

mtry_search <- seq(22, 24, 1)
min_n_search <- seq(30, 50, 2)

grd <- expand.grid(mtry_search, min_n_search)

hyp_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}},
             trees = 1000)
  
  wf <- wf %>% 
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

tictoc::tic()
mtry_results_4 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
tictoc::toc()

mtry_results_4 %>% 
  arrange(rmse) # Best mtry = .. & min_n = ... for an rmse of ...

### plot 4 ###
mtry_results_4 %>% 
  ggplot(aes(mtry, rmse)) +
  geom_line() +
  geom_point() +
  facet_wrap(~min_n)

#################
#################

##### TUNING 5 #####

mtry_search <- 24
min_n_search <- seq(38, 50, 2)

grd <- expand.grid(mtry_search, min_n_search)

hyp_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}},
             trees = 1000)
  
  wf <- wf %>% 
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

tictoc::tic()
mtry_results_5 <- map2_df(grd$Var1, grd$Var2, ~hyp_rf_search(.x, .y, rf_def_wkflw))
tictoc::toc()

mtry_results_5 %>% 
  arrange(rmse) # Best mtry = .. & min_n = ... for an rmse of ...

### plot 5 ###
mtry_results_5 %>% 
  ggplot(aes(min_n, rmse)) +
  geom_line() +
  geom_point() 

# the best hyperparameters are mtry = 24, min_n = 50. I think could keep incresing min_n, but not sure if it's "allowed" or worth it

prepped_rec <- rec %>%
  prep %>%
  bake(data_train)

tune_min_n <- function(n) {
  mod <- rand_forest() %>% 
    set_mode("regression") %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_args(mtry = 24,
             min_n = n,
             trees = 1000)
  # fit model to full training dataset
  m <- fit(mod, score ~ ., prepped_rec)
  # Extract RMSE, store as a tibble
  tibble(rmse = sqrt(m$fit$prediction.error))
}

tictoc::tic()
optimal_n <- map_df(seq(2, 150, 2), tune_min_n) 
tictoc::toc()

optimal_n %>% 
  mutate(n = seq(2, 150, 2)) %>% 
  ggplot(aes(n, rmse)) +
  geom_line() +
  geom_point()

# --> not worth it: It took over 45 minutes to run. the plot shows that around 50 it's 90.5 and around 100 is 90. After a hundred it levels out. 
##############################################


##### final fit #####

final_mod <- rand_forest() %>% 
  set_engine("ranger",
             num.threads = cores,
             importance = "permutation",
             verbose = TRUE) %>% 
  set_mode("regression") %>% 
  set_args(mtry = 24,
           min_n = 50,
           trees = 1000)

final_wkfl <-  workflow() %>% # do I need to use finalize_workflow ?
  add_model(final_mod) %>% 
  add_recipe(rec)
  
  
final_fit <- fit(final_wkfl,
                 data = data_train)

extract_rmse(final_fit) # rmse = 90.86 (with data_train)

tictoc::tic()
last_fit <- last_fit(final_wkfl, 
                 split = data_split)
tictoc::toc()

final_fit$.metrics[[1]] # rmse = 93.0 (with data_split)

##### final fit #####

### if submitting to kaggle ###
# read in data (test.csv), process, join, recipe, bake it, make predictions 
# 
# preds <- predict(final_fit$fit, new_data = processed_test)
# 
# where final_fit is what you have on lines 408/409 and
# processed_test is the test.csv that has been joined with bonus and
# had the recipe applied/baked
# 
# write out a file for kaggle that is something like
# tibble(Id = processed_test$ssid, Prediction = preds) %>%
#   write_csv("kaggle_final_rf.csv")