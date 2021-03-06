library(tidyverse)
library(tidymodels)
library(xgboost)
# 
## local
# git <- "~/Documents/GitHub/EDLD-654-Final"
# 
# data <- import(path(git, "data/train.csv")) %>%
#   select(-classification) %>%
#   mutate_if(is.character, factor) %>%
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


## basic recipe
rec <- recipe(score ~ ., train) %>%
  step_mutate(tst_dt = as.numeric(lubridate::mdy_hms(tst_dt)),
              lang_cd = case_when(lang_cd == "S" ~ "S", TRUE ~ "E"),
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


# bake recipe
baked_train <- prep(rec) %>% 
  bake(train)
# 
# baked_test <- prep(rec) %>%
#   bake(test)


train_x = data.matrix(baked_train[, -73])
train_y = data.matrix(baked_train[, 73])

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
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 1,
    params = list( 
      eta = 0.1,
      gamma = 50,
        max_depth = .x,
      min_child_weight = .y,
      nthread = 24
    ) 
  )  
}) 

saveRDS(tree_mods, "tree_mods_50.rds")
read
tree_mods <- readRDS("~/Documents/GitHub/EDLD-654-Final/scripts/XGBoost/tree_mods_50.rds")

tree_res <- map_df(tree_mods,
               ~.x$evaluation_log[.x$best_iteration, ])
tree_params <- map_df(tree_mods,
                   ~.x$params)

tree <- cbind(tree_res, tree_params)

