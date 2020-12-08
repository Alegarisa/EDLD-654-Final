library(tidyverse)
library(tidymodels)
library(xgboost)
<<<<<<< Updated upstream
#
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
#   mutate(locale = gsub("^.{0,3}", "", locale)) %>%
#   separate(col = locale, into = c("locale", "sublocale"), sep = ": ") %>%
#   select(-moms, -dads, -families, -households, -no_internet) %>%
#   left_join(fin) %>% 
#   mutate(hpi = as.numeric(hpi))
# 
# write_csv(bonus, "bonus_data_v2.csv")

bonus <- read.csv("data/bonus_data_v2.csv") %>% 
  mutate(ncessch = as.numeric(ncessch))

# disc <- read.csv("data/bonus source files/disc_drop.csv")

disc <- read_csv("data/disc_drop.csv") %>% 
  mutate(attnd_dist_inst_id = as.double(attnd_dist_inst_id))

## join data
data <- data %>% 
  left_join(bonus) %>% 
  left_join(disc) 

set.seed(3000)

data_split <- initial_split(data) 
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
>>>>>>> Stashed changes

train <- training(data_split)

test <- testing(data_split)

## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
<<<<<<< Updated upstream
              pupil_tch_ratio = as.numeric(pupil_tch_ratio),
              pupil_tch_ratio = case_when(pupil_tch_ratio < 18 ~ 1,
                                          pupil_tch_ratio < 25 ~ 2,
                                          pupil_tch_ratio < 30 ~ 3,
                                          TRUE ~ 4),
              pupil_tch_ratio = as.factor(pupil_tch_ratio)) %>% 
  step_rm(contains("id"), ncessch, ncesag, lea_name, no_internet, sch_name, fr_lnch_n, red_lnch_n) %>%
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
  step_normalize(all_numeric(), -has_role("outcome")) %>% 
  step_knnimpute(all_numeric(), options = list(nthread = 24)) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)
=======
              pupil_tch_ratio = as.numeric(pupil_tch_ratio)) %>% 
  step_rm(contains("id"), ncessch, ncesag, lea_name, sch_name) %>%
  step_zv(all_predictors()) %>%
  step_knnimpute(all_numeric()) %>% 
  step_unknown(all_nominal()) %>% 
  step_dummy(all_nominal())

>>>>>>> Stashed changes

# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)

baked_test <- prep(rec) %>% 
  bake(test)

<<<<<<< Updated upstream
saveRDS(baked_train, "baked_train.rds")
saveRDS(baked_test, "baked_test.rds")
=======
write_csv(baked_train, "baked_train.csv")
write_csv(baked_test, "baked_test.csv")
>>>>>>> Stashed changes
