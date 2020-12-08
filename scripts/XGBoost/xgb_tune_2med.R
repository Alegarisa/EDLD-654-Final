library(tidyverse)
library(tidymodels)
library(xgboost)

<<<<<<< Updated upstream
## local
git <- "~/Documents/GitHub/EDLD-654-Final"

data <- import(path(git, "data/train.csv")) %>%
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch))

bonus <- import(path(git, "data/bonus_data.csv")) %>%
  mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
  mutate(ncessch = as.double(ncessch))
=======
# ## local
# git <- "~/Documents/GitHub/EDLD-654-Final"
# 
# data <- import(path(git, "data/train.csv")) %>%
#   select(-classification) %>%
#   mutate_if(is.character, factor) %>%
#   mutate(ncessch = as.double(ncessch)) %>%
#   sample_frac(.10)
# 
# bonus <- import(path(git, "data/bonus_data.csv")) %>%
#   mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
#   mutate(ncessch = as.double(ncessch))
>>>>>>> Stashed changes

## talapas
data <- read_csv("data/train.csv") %>% 
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch))

bonus <- read_csv("data/bonus_data.csv") %>% 
  mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
  mutate(ncessch = as.double(ncessch))

## join data
data <- data %>% 
  left_join(bonus)

set.seed(3000)

data_split <- initial_split(data)

train <- training(data_split)

test <- testing(data_split)

## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
              pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
<<<<<<< Updated upstream
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name, zip) %>%
  step_string2factor(all_nominal()) %>% 
  step_zv(all_predictors()) %>%
  step_novel(all_nominal()) %>% 
  step_unknown(all_nominal()) %>% 
  step_medianimpute(all_numeric()) %>% 
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)
=======
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_zv(all_predictors()) %>%
  step_medianimpute(all_numeric()) %>% 
  step_unknown(all_nominal()) %>% 
  step_dummy(all_nominal())
>>>>>>> Stashed changes


# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

baked_test <- prep(rec) %>% 
  bake(test)

## organize in to matrices
<<<<<<< Updated upstream
train_x = data.matrix(baked_train[, -45])
train_y = data.matrix(baked_train[, 45])
test_x = data.matrix(baked_test[, -45])
test_y = data.matrix(baked_test[, 45])
=======
train_x = data.matrix(baked_train[, -46])
train_y = data.matrix(baked_train[, 46])
test_x = data.matrix(baked_test[, -46])
test_y = data.matrix(baked_test[, 46])
>>>>>>> Stashed changes

## set xgb matrices
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

pull_eval <- function(m) {
  m[["evaluation_log"]] %>% 
    pivot_longer(-iter,
                 names_to = c("set", NA, "stat"),
                 names_sep = "_",
                 values_to = "val") %>% 
    pivot_wider(names_from = "stat", 
                values_from = "val") 
}

# Set learning rate, tune tree specific parameters
grid <- grid_max_entropy(min_n(c(0, 50)), # min_child_weight
                         tree_depth(), # max_depth
                         size = 30)

tree_mods <- map2(grid$min_n, grid$tree_depth, ~{
  xgb.cv(
    data = train_x,
    label = train_y,
    nrounds = 5000,
    objective = "reg:squarederror",
<<<<<<< Updated upstream
    early_stopping_rounds = 0, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = 0.1,
      max_depth = .x,
      min_child_weight = .y,
      nthread = 24
=======
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = 0.0414655172413793,
      max_depth = .x,
      min_child_weight = .y,
      nthread = 16
>>>>>>> Stashed changes
    ) 
  )  
}) 

saveRDS(tree_mods, "tree_mods_med.rds")

