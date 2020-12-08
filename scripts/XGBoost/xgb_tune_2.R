library(tidyverse)
library(tidymodels)
library(xgboost)
<<<<<<< Updated upstream
library(rio)
library(fs)
## local
git <- "~/Documents/GitHub/EDLD-654-Final"
# data <- import(path(git, "data/train.csv")) %>%
#   select(-classification) %>%
#   mutate_if(is.character, factor) %>%
#   mutate(ncessch = as.double(ncessch))
# 
bonus <- import(path(git, "data/bonus_data.csv")) %>%
  mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
  mutate(ncessch = as.double(ncessch)) %>% 
  mutate(hpi = as.numeric(hpi)) %>% 
  select(-moms, -dads, -families, -households) %>% 
  left_join(fin)

write_csv(bonus, "bonus_data_v2.csv")

# ## join data
# data <- data %>%
#   left_join(bonus)
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

<<<<<<< Updated upstream

bonus <- read_csv("data/bonus_data.csv") %>% 
  mutate(ncessch = as.double(ncessch)) %>% 
  mutate(hpi = as.numeric(hpi))
=======
bonus <- read_csv("data/bonus_data.csv") %>% 
  mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
  mutate(ncessch = as.double(ncessch))
>>>>>>> Stashed changes

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
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name, total_n, fr_lnch_n, red_lnch_n,) %>%
  step_mutate(hpi = as.numeric(hpi),
              lat = round(lat, 2),
              lon = round(lon, 2),
              median_income = log(median_income),
              median_rent = log(median_rent),
              frl_prop = fr_lnch_prop + red_lnch_prop) %>% 
  step_rm(fr_lnch_prop, red_lnch_prop) %>% 
  step_interact(terms = ~ lat:lon) %>% 
  step_string2factor(all_nominal()) %>% 
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>% 
  step_medianimpute(all_numeric()) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)
=======
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>% 
  step_dummy(all_nominal())

>>>>>>> Stashed changes

# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

<<<<<<< Updated upstream
# baked_test <- prep(rec) %>% 
#   bake(test)

## organize in to matrices
train_x = data.matrix(baked_train[, -67])
train_y = data.matrix(baked_train[, 67])
# test_x = data.matrix(baked_test[, -67])
# test_y = data.matrix(baked_test[, 67])

## set xgb matrices
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
# xgb_test = xgb.DMatrix(data = test_x, label = test_y)
=======
baked_test <- prep(rec) %>% 
  bake(test)

## organize in to matrices
train_x = data.matrix(baked_train[, -46])
train_y = data.matrix(baked_train[, 46])
test_x = data.matrix(baked_test[, -46])
test_y = data.matrix(baked_test[, 46])

## set xgb matrices
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
tree_mods <- map2(grid$min_n, grid$tree_depth, ~{
  xgb.cv(
    data = train_x,
    label = train_y,
    nrounds = 5000,
    objective = "reg:squarederror",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
<<<<<<< Updated upstream
      eta = 0.1,
      gamma = 
      max_depth = .x,
      min_child_weight = .y,
      nthread = 24
=======
      eta = 0.0414655172413793,
      max_depth = .x,
      min_child_weight = .y,
      nthread = 16
>>>>>>> Stashed changes
    ) 
  )  
}) 

saveRDS(tree_mods, "tree_mods.rds")

