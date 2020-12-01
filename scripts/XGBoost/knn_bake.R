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
  step_knnimpute(all_numeric(), options = list(nthread = 24)) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)

# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

baked_test <- prep(rec) %>% 
  bake(test)

saveRDS(baked_train, "baked_train.rds")
saveRDS(baked_test, "baked_test.rds")
