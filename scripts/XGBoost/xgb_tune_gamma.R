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
data <- read.csv("data/train.csv") %>% 
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.numeric(ncessch)) %>% 
  select(-id)

# bonus <- import(path(git, "data/bonus_data.csv")) %>%
#   mutate(ncessch = as.double(ncessch),
#          pupil_tch_ratio = as.numeric(pupil_tch_ratio),
#          pupil_tch_ratio = as.factor(case_when(pupil_tch_ratio < 18 ~ 1,
#                                                pupil_tch_ratio < 25 ~ 2,
#                                                pupil_tch_ratio < 30 ~ 3,
#                                                TRUE ~ 4))) %>%
  # mutate(locale = gsub("^.{0,3}", "", locale)) %>%
  # separate(col = locale, into = c("locale", "sublocale"), sep = ": ") %>%
#   select(-moms, -dads, -families, -households, -no_internet) %>%
#   left_join(fin) %>% 
#   mutate(hpi = as.numeric(hpi))
# 
# write_csv(bonus, "bonus_data_v2.csv")

bonus <- read.csv("data/bonus_data_v2.csv") %>% 
  mutate(ncessch = as.numeric(ncessch)) %>% 
  mutate(locale = gsub("^.{0,3}", "", locale)) %>%
  separate(col = locale, into = c("locale", "sublocale"), sep = ": ")

# disc <- read.csv("data/bonus source files/disc_drop.csv")

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

## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
              pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
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

# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

# baked_test <- prep(rec) %>% 
#   bake(test)

## organize in to matrices
train_x = data.matrix(baked_train[, -73])
train_y = data.matrix(baked_train[, 73])
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

# grid <- as.data.frame(grid)
# ## grid
# gamma_mods <- map(grid$loss_reduction, ~{
#   xgb.cv(
#     data = train_x,
#     label = train_y,
#     nrounds = 10000,
#     objective = "reg:squarederror",
#     early_stopping_rounds = 50, 
#     nfold = 10,
#     verbose = 1,
#     params = list( 
#       eta = 0.1,
#       gamma = .x,
#       nthread = 24
#     ) 
#   )  
# }) 

## Standard CV
play_mods <- xgb.cv(
  data = train_x,
  label = train_y,
  nrounds = 10000,
  objective = "reg:squarederror",
  early_stopping_rounds = 50, 
  nfold = 10,
  verbose = 1,
  params = list( 
    eta = 0.1,
    gamma = 10,
    nthread = 24
  ))

saveRDS(play_mods, "play_mods.rds")
end_time <- Sys.time()

print(end_time - start_time)
# 
# Play <- play_mods$evaluation_log
# play_mods[[1]]$evaluation_log[sample_mods[[1]]$best_iteration, ]

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
model <- sample_mods
model <- map_df(play_mods,
                    ~.x$evaluation_log[.x$best_iteration, ])
model_params <- map_df(sample_mods,
                       ~.x$params)

mod_res <- cbind(model, model_params)
