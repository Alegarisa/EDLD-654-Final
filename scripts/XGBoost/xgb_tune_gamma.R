library(tidyverse)
library(tidymodels)
library(xgboost)
#
# ## local
# git <- "~/Documents/GitHub/EDLD-654-Final"
# data <- import(path(git, "data/train.csv")) %>%
#   select(-classification) %>%
#   mutate_if(is.character, factor) %>%
#   mutate(ncessch = as.double(ncessch))
# 
# bonus <- import(path(git, "data/bonus_data.csv")) %>%
#   mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
#   mutate(ncessch = as.double(ncessch))

# ## join data
# data <- data %>%
#   left_join(bonus)

## talapas
data <- read_csv("data/train.csv") %>% 
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch))


bonus <- read_csv("data/bonus_data_v2.csv") %>% 
  mutate(ncessch = as.double(ncessch)) %>% 
  mutate(hpi = as.numeric(hpi))

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
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name, total_n, fr_lnch_n, red_lnch_n) %>%
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

# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

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

# pull_eval <- function(m) {
#   m[["evaluation_log"]] %>% 
#     pivot_longer(-iter,
#                  names_to = c("set", NA, "stat"),
#                  names_sep = "_",
#                  values_to = "val") %>% 
#     pivot_wider(names_from = "stat", 
#                 values_from = "val") 
# }
# Take start time to measure time of random search algorithm
start.time <- Sys.time()

grid <- expand.grid(loss_reduction = seq(0, 80, 5))


gamma_mods <- map(grid$loss_reduction, ~{
  xgb.cv(
    data = train_x,
    label = train_y,
    nrounds = 10000,
    objective = "reg:squarederror",
    early_stopping_rounds = 50, 
    nfold = 7,
    verbose = 0,
    params = list( 
      eta = 0.1,
      gamma = .x,
      nthread = 24
    ) 
  )  
}) 
saveRDS(gamma_mods, "gamma_mods.rds")
# 
# sample_mods[[1]]$evaluation_log[sample_mods[[1]]$best_iteration, ]
# 
# sample_mods
# gamma_res <- map_df(gamma_mods,
#                ~.x$evaluation_log[.x$best_iteration, ])
# gamma_params <- map_df(gamma_mods,
#                    ~.x$params)
# 
# gamma_res <- cbind(gamma_res, gamma_params)
# 
# sample_mods[[4]]$evaluation_log[sample_mods[[4]]$best_iteration, ]
# sample_mods[[4]]$params
# 
# sample_mods$evaluation_log[sample_mods$best_iteration, ]

