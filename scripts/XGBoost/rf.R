
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

data <- data %>% 
  sample_frac(.05)

# Initial Splits
data_split <- initial_split(data) 

train <- training(data_split)
test <- testing(data_split)



# CV split
cv_split <- vfold_cv(train) 

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
  step_other(all_nominal(), threshold = .01) %>% 
  step_medianimpute(all_numeric()) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>% 
  step_interact(~ exp_current_supp_serve_total.x:sp_ed_fg_Y) %>% 
  step_interact(~ lang_cd_S:p_hispanic_latino) %>% 
  step_nzv(all_predictors(), freq_cut = 995/5)


rfrst_tune <- 
rand_forest() %>% 
  set_engine("ranger",
             importance = "permutation",
             verbose = TRUE) %>% 
  set_mode("regression")  %>% 
  set_args(mtry = 24,
           trees = 1000,
           min_n = 50)


rfrst_wflow <-workflow() %>% 
  add_recipe(rec) %>% 
  add_model(rfrst_tune)

set.seed(345)
tune_res <- fit_resamples(
  rfrst_wflow,
  resamples = cv_split)

tune_res <- tune_grid(
  rfrst_wflow,
  resamples = cv_split)
tune_res <- rf_res
saveRDS(tune_res, "rf_res.rds")

tune_res %>%
  collect_metrics() %>%
  filter(.metric == "rmse") %>%
  select(mean, min_n, mtry) %>%
  pivot_longer(min_n:mtry,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

# start_rf <- Sys.time()
# rfrst_grid <- fit_resamples(rfrst_wflow,
#                             cv_split,
#                             metrics = yardstick::metric_set(rmse, rsq, huber_loss),
#                             control = control_resamples(extract = function(x) x))
# end_rf <- Sys.time()
# end_rf - start_rf
# 
# 
# show_best(rfrst_grid, metric = "rmse", n = 1)
# show_best(rfrst_grid, metric = "rsq", n = 1)
# show_best(rfrst_grid, metric = "huber_loss", n = 1)
# 
# collect_metrics(rfrst_grid)
# 
# 
# rf_tree_roots <- function(x){
#   map_chr(1:1000, 
#           ~ranger::treeInfo(x, tree = .)[1, "splitvarName"])
# }
# 
# rf_roots <- function(x){
#   x %>% 
#     select(.extracts) %>% 
#     unnest(cols = c(.extracts)) %>% 
#     mutate(fit = map(.extracts,
#                      ~.x$fit$fit$fit),
#            oob_rmse = map_dbl(fit,
#                               ~sqrt(.x$prediction.error)),
#            roots = map(fit, 
#                        ~rf_tree_roots(.))
#     ) %>% 
#     select(roots) %>% 
#     unnest(cols = c(roots))
# }
# 
# 
# rf_roots(rfrst_grid) %>% 
#   ggplot(aes(x = roots)) +
#   geom_bar() +
#   coord_flip()
# 
#  
# tic()
# rfrst_fit <- fit(rfrst_wflow,
#                  data = data_train)
# toc()
# 
# pluck(sqrt(rfrst_fit$fit$fit$fit$prediction.error))
