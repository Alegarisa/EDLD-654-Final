library(tidyverse)
library(tidymodels)
library(xgboost)

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
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_zv(all_predictors()) %>%
  step_knnimpute(all_numeric()) %>% 
  step_unknown(all_nominal()) %>% 
  step_dummy(all_nominal())


# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

baked_test <- prep(rec) %>% 
  bake(test)

write_csv(baked_train, "baked_train.csv")
write_csv(baked_test, "baked_test.csv")
