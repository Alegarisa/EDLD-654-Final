library(tidyverse)
library(tidymodels)
library(xgboost)
<<<<<<< Updated upstream
# 
## local
=======

# ## local
>>>>>>> Stashed changes
# git <- "~/Documents/GitHub/EDLD-654-Final"
# 
# data <- import(path(git, "data/train.csv")) %>%
#   select(-classification) %>%
#   mutate_if(is.character, factor) %>%
<<<<<<< Updated upstream
#   mutate(ncessch = as.double(ncessch))
# 
# bonus <- import(path(git, "data/bonus_data_v2.csv")) %>%
#   mutate(ncessch = as.double(ncessch)) %>% 
#   mutate(locale = gsub("^.{0,3}", "", locale)) %>%
#   separate(col = locale, into = c("locale", "sublocale"), sep = ": ")

# 
# disc <- read.csv("data/bonus source files/disc_drop.csv")
## talapas
data <- read.csv("data/train.csv") %>%
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.numeric(ncessch))


bonus <- read.csv("data/bonus_data_v2.csv") %>%
  mutate(ncessch = as.numeric(ncessch)) %>%
  mutate(locale = gsub("^.{0,3}", "", locale)) %>%
  separate(col = locale, into = c("locale", "sublocale"), sep = ": ")

disc <- read_csv("data/disc_drop.csv") %>%
  mutate(attnd_dist_inst_id = as.double(attnd_dist_inst_id))

## join data
data <- data %>% 
  left_join(bonus) %>% 
  left_join(disc)

set.seed(3000)

data_split <- initial_split(data) 

train <- training(data_split)
test <- testing(data_split)

=======
#   mutate(ncessch = as.double(ncessch)) %>%
#   sample_frac(.10)
# 
# bonus <- import(path(git, "data/bonus_data.csv")) %>%
#   mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
#   mutate(ncessch = as.double(ncessch))

## talapas
data <- read_csv("data/train.csv") %>% 
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch)) %>% 
  sample_frac(.10)

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
>>>>>>> Stashed changes

## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
<<<<<<< Updated upstream
              pupil_tch_ratio = as.numeric(pupil_tch_ratio),
              pupil_tch_ratio = as.numeric(pupil_tch_ratio),
              pupil_tch_rate = case_when(pupil_tch_ratio < 18 ~ 1,
                                         pupil_tch_ratio < 25 ~ 2,
                                         pupil_tch_ratio < 30 ~ 3, 
                                         TRUE ~ 4),
              pupil_tch_rate = as.factor(pupil_tch_rate)) %>% 
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_mutate(hpi = as.numeric(hpi),
              lat = round(lat, 2),
              lon = round(lon, 2),
              median_income = log(median_income),
              median_rent = log(median_rent),
              frl_prop = fr_lnch_prop + red_lnch_prop,
              schl_perf = case_when(sch_percent_level_1 + sch_percent_level_2 > sch_percent_level_3 + sch_percent_level_4 ~ 1,
                                    TRUE ~ 0),
              over_100 = under_200 + over_200) %>% 
  step_interact(terms = ~ lat:lon) %>% 
  step_rm(fr_lnch_prop, red_lnch_prop) %>% 
  step_string2factor(all_nominal()) %>% 
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>% 
  step_medianimpute(all_numeric()) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_interact(~ exp_current_supp_serve_total.x:sp_ed_fg_Y) %>% 
  step_interact(~ lang_cd_S:p_hispanic_latino) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)
=======
              pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>% 
  step_medianimpute(all_numeric()) %>% 
  step_dummy(all_nominal())
>>>>>>> Stashed changes


# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)
<<<<<<< Updated upstream
# 
# baked_test <- prep(rec) %>%
#   bake(test)


train_x = data.matrix(baked_train[, -73])
train_y = data.matrix(baked_train[, 73])
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
<<<<<<< Updated upstream
# Take start time to measure time of random search algorithm
start.time <- Sys.time()

grid <- grid_max_entropy(mtry(as.integer(c(.3*185, .9*185))), # min_child_weight
                         sample_size(as.integer(c(.5*nrow(train), nrow(train)))), # max_depth
                         size = 30)

grid <- grid %>% 
  mutate(mtry = mtry/185,
         sample_size = sample_size/nrow(train))

print(grid)

sample_mods <- map2(grid$mtry, grid$sample_size, ~{
=======

# Set learning rate, tune tree specific parameters
grid <- grid_max_entropy(min_n(c(0, 50)), # min_child_weight
                         tree_depth(), # max_depth
                         size = 30)

param_test4 = {
  'subsample':[i/10.0 for i in range(6,10)],
  'colsample_bytree':[i/10.0 for i in range(6,10)]
}
tree_mods <- map2(grid$min_n, grid$tree_depth, ~{
>>>>>>> Stashed changes
  xgb.cv(
    data = train_x,
    label = train_y,
    nrounds = 5000,
<<<<<<< Updated upstream
    objective = "reg:squarederror",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 1,
    params = list( 
      eta = 0.1,
      gamma = 10,
      max_depth = 6,
      min_child_weight = 10,
      colsample_bytree = .x,
      subsample = .y,
      nthread = 24
    ) 
  )  
}) 
saveRDS(sample_mods, "sample_mods.rds")

# sample_mods[[1]]$evaluation_log[sample_mods[[1]]$best_iteration, ]
# 
# sample_mods
# sample_res <- map_df(sample_mods,
#                ~.x$evaluation_log[.x$best_iteration, ])
# sample_params <- map_df(sample_mods,
#                    ~.x$params)
# 
# res <- cbind(sample_res, sample_params)
# 
# sample_mods[[4]]$evaluation_log[sample_mods[[4]]$best_iteration, ]
# sample_mods[[4]]$params
# 
# sample_mods$evaluation_log[sample_mods$best_iteration, ]

=======
    objective = "reg:linear",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = 0.0414655172413793,
      max_depth = .x,
      min_child_weight = .y,
      nthread = 20
    ) 
  )  
}) 

saveRDS(tree_mods, "tree_mods.rds")
>>>>>>> Stashed changes
