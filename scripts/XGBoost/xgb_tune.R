library(tidyverse)
library(tidymodels)
library(xgboost)
needs(rio, fs)
# 
## local
git <- "~/Documents/GitHub/EDLD-654-Final"

data <- import(path(git, "data/train.csv")) %>%
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.double(ncessch))
# 
# bonus <- import(path(git, "data/bonus_data_v2.csv")) %>%
#   mutate(pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>%
#   mutate(ncessch = as.double(ncessch)) %>% 
#   mutate(hpi = as.numeric(hpi))

# bonus <- import(path(git, "data/bonus_data_v2.csv")) %>%
#   mutate(ncessch = as.double(ncessch))

## talapas
data <- read.csv("data/train.csv") %>% 
  select(-classification) %>%
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.numeric(ncessch))

data <- read.csv("data/test.csv") %>% 
  mutate_if(is.character, factor) %>%
  mutate(ncessch = as.numeric(ncessch))

# bonus <- import(path(git, "data/bonus_data.csv")) %>%
#   mutate(ncessch = as.double(ncessch),
#          pupil_tch_ratio = as.numeric(pupil_tch_ratio),
#          pupil_tch_ratio = as.factor(case_when(pupil_tch_ratio < 18 ~ 1,
#                                                pupil_tch_ratio < 25 ~ 2,
#                                                pupil_tch_ratio < 30 ~ 3,
#                                                TRUE ~ 4))) %>%
#   mutate(locale = gsub("^.{0,3}", "", locale)) %>%
#   separate(col = locale, into = c("locale", "sublocale"), sep = ": ") %>%
#   select(-moms, -dads, -families, -households, -no_internet) %>%
#   left_join(fin) %>% 
#   mutate(hpi = as.numeric(hpi))
# 
# write_csv(bonus, "bonus_data_v2.csv")

bonus <- read.csv("data/bonus_data_v2.csv") %>% 
  mutate(ncessch = as.numeric(ncessch))

disc <- read.csv("data/bonus source files/disc_drop.csv")

disc <- read_csv("data/disc_drop.csv") %>% 
  mutate(attnd_dist_inst_id = as.double(attnd_dist_inst_id))

## join data
data <- data %>% 
  left_join(bonus) %>% 
  left_join(disc)

set.seed(3000)
names(baked_train)
data_split <- initial_split(data) 

train <- training(data_split)
test <- data
test <- testing(data_split)

train_2 <- train

train <- rbind(train, train_2)
## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
              pupil_tch_ratio = as.numeric(pupil_tch_ratio),
              pupil_tch_ratio = case_when(pupil_tch_ratio < 18 ~ 1,
                                          pupil_tch_ratio < 25 ~ 2,
                                          pupil_tch_ratio < 30 ~ 3, 
                                          TRUE ~ 4),
              pupil_tch_ratio = as.factor(pupil_tch_ratio)) %>% 
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name, no_internet, fr_lnch_n, red_lnch_n) %>%
  step_mutate(hpi = as.numeric(hpi),
              lat = round(lat, 2),
              lon = round(lon, 2),
              median_income = log(median_income),
              median_rent = log(median_rent),
              frl_prop = fr_lnch_prop + red_lnch_prop,
              estimate = 100 - (sch_percent_level_1 + 2*sch_percent_level_2) +
                (sch_percent_level_3 + 2 * sch_percent_level_4)) %>% 
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

baked_test <- prep(rec) %>%
  bake(test)

# smallbake <- baked_train %>%
#   sample_frac(.40)
# 
# train_x = data.matrix(smallbake[, -67])
# train_y = data.matrix(smallbake[, 67])
## organize in to matrices
train_x = data.matrix(baked_train[, -69])
train_y = data.matrix(baked_train[, 69])
test_x = data.matrix(baked_test[, -69])
test_y = data.matrix(baked_test[, 69])

## set xgb matrices
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

## first xgb model
# 
# def_mod <- xgb.cv(
#   data = train_x,
#   label = train_y,
#   nrounds = 8000,
#   objective = "reg:squarederror",
#   early_stopping_rounds = 80,
#   nfold = 8,
#   verbose = 1,
#   eval_metric = "rmse",
#   params = list(
#     eta = .4,
#     gamma = 10,
#     subsample = 0.20,
#     colsample_bytree = 0.8,
#     nthread = 24
#   ))

# importance_matrix <- xgb.importance(model = bst)
# print(importance_matrix)
# xgb.plot.importance(importance_matrix = importance_matrix)
# 
bst <- xgboost(data = train_x,
                 label = train_y,
                 nrounds = 400,
                 verbose = 1,
                 params = list(
                   eta = .4,
                   gamma = 10,
                   subsample = 0.40,
                   colsample_bytree = 0.8,
                   nthread = 24)
)

saveRDS(bst, "model_test.Rds")
xgb_
test_x = data.matrix(baked_test[, -69])
test_y = data.matrix(baked_test[, 69])

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

# def_mod$evaluation_log[def_mod$best_iteration, ]
# # 
saveRDS(def_mod, "def_mod_v3.Rds")
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
