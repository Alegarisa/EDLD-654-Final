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
set.seed(89)

full_train <- import(here("data", "train.csv"), setclass = "tbl_df") %>% 
  sample_frac(0.01) %>% 
  select(-classification)

d <- left_join(full_train, ethnicities) # add other data 
############################################################################### 

### Split & resample ###
set.seed(89)
d_split <- initial_split(d, strata = "score") # do we want to stratify?

set.seed(89)
d_train <- training(d_split)
d_test <- testing(d_split)

set.seed(89)
cv_splits <- vfold_cv(d_train, strata = "score")
###############################################################################

### Preprocess ###
rec <- recipe(score ~ ., d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(contains("id"), ncessch, sch_name, new_role = "id_vars") %>%
  step_unknown(all_nominal()) %>% # not really sure about the difference with step_novel
  step_zv(all_predictors(), -starts_with("lang_cd")) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors(), -starts_with("lang_cd"))
###############################################################################

### Building the Random Forest model - default ###

# mtry = floor(p/3)
# trees = 500 (num.trees)
# min_n = 1 (min.node.size)
# what about this: respect.unordered.factors = "order"?

sum(rec$var_info$role == "predictor") # number of predictors

rand_for_def <- rand_forest() %>% 
  set_engine("ranger",
             importance = "permutation",
             verbose = TRUE) %>% 
  set_mode("regression")  

translate(rand_for_def)

### workflow for default model
rand_for_def_wkflw <- workflow() %>% 
  add_model(rand_for_def) %>% 
  add_recipe(rec)

### Fitting the default model ###
metrics_eval <- metric_set(rmse, 
                           rsq, 
                           huber_loss)

tictoc::tic()
set.seed(89)
rand_for_def_fit <- fit_resamples(
  rand_for_def_wkflw, 
  cv_splits, 
  metrics = metrics_eval,
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE,
                              extract = function(x) x))
tictoc::toc() 

### Best estimate of Random Forest default model ###
show_best(rand_for_def_fit, "rmse", n = 1)

#################
#################

### Building the Random Forest model - tuned ###
# mtry = tune()
# trees = 1000
# min_n = tune()
# what about this: respect.unordered.factors = "order"?

sum(rec$var_info$role == "predictor") # number of predictors

rand_for_tun <- rand_forest() %>% # change this to a tunning model
  set_engine("ranger",
             importance = "permutation",
             verbose = TRUE) %>% 
  set_mode("regression")  

translate(rand_for_tun)

### workflow for tuned model
rand_for_tun_wkflw <- workflow() %>% 
  add_model(rand_for_tun) %>% 
  add_recipe(rec)

### Fitting the tuned model ###
metrics_eval <- metric_set(rmse, 
                           rsq, 
                           huber_loss)

tictoc::tic() 
set.seed(89)
rand_for_tun_fit <- fit_resamples(
  rand_for_def_wkflw, 
  cv_splits, 
  metrics = metrics_eval,
  control = control_resamples(verbose = TRUE,
                              save_pred = TRUE,
                              extract = function(x) x))
tictoc::toc() 

### Best estimate of Random Forest tuned model ###
show_best(rand_for_tun_fit, "rmse", n = 1)

### finalize model with best hyperparameters