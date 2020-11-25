library(tidyverse)
library(tidymodels)
library(rio)
library(here)
library(glmnet)
library(lme4)

full_train <- read_csv(here::here("data","train.csv")) %>% 
  select(-classification)

str(full_train)


#library(rio)
frl <- import("https://nces.ed.gov/ccd/Data/zip/ccd_sch_033_1718_l_1a_083118.zip",
              setclass = "tbl_df")  %>% 
  janitor::clean_names()  %>% 
  filter(st == "OR")  %>%
  select(ncessch, lunch_program, student_count)  %>% 
  mutate(student_count = replace_na(student_count, 0))  %>% 
  pivot_wider(names_from = lunch_program,
              values_from = student_count)  %>% 
  janitor::clean_names()  %>% 
  mutate(ncessch = as.double(ncessch))

stu_counts <- import("https://github.com/datalorax/ach-gap-variability/raw/master/data/achievement-gaps-geocoded.csv",
                     setclass = "tbl_df")  %>% 
  filter(state == "OR" & year == 1718)  %>% 
  count(ncessch, wt = n)  %>% 
  mutate(ncessch = as.double(ncessch))

frl <- left_join(frl, stu_counts)
frl

frl_props <- frl %>%
  mutate(prop_fl = free_lunch_qualified/n,
         prop_rl = reduced_price_lunch_qualified/n) %>%
  select(ncessch, prop_fl, prop_rl)

str(frl_props)

d <- left_join(full_train, frl_props)
d

# library(tidymodels)

set.seed(89)

d_split <- initial_split(d)

d_train <- training(d_split)
d_test  <- testing(d_split)

str(d_train)

rec <- recipe(score ~ ., d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(contains("id"), ncessch, new_role = "id_vars") %>%
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors())

prepped_rec <- prep(rec)
prepped_rec

rec %>% 
  prep %>% 
  bake(d_train) 

# notice here that a lot of vars classified as "unknown may be easily identified... maybe. Also, consider using one-hot encoding for penalised regression (no need for refrence groups) 

# In the markdown chunk below, briefly explain what operations the recipe will apply (specifically, mentioning at least a few column names) to your data.

# The variable tst_dt was read as a string variable and it will be formatted as a date variable.
# Some variables are not really predictors, they are identifiers, so they will be assigned a new role called id_vars. 
# the variables that had zero variance like calc_admn_cd and lang_cd were removed as also were variables with near zero variance, like migrant_ed_fg and trgt_assist_fg, among others. 
# Missing data on categorical variables will be handled by recoding the missing values as an "unknown" category. 
# Missing data on mumerical variables will be handled using median imputation. Only numerical predictors with missing values, and not the outcome, will be median-inputed. It's important to have in mind that this method will not fix MNAR data. 
# All numeric variables that are not outcomes or identifiers will be centered and scaled, for instance, lat, lon, and the proportion variables representing students eligible for free and reduced price lunch. Centering will reduce collinearity and scaling will be needed when regularized regression models are applied. 
# All string variables will be dummy coded, for instance, gnder, econ_dsvntg, and enrl_grd, amomg others. 

# NOTE: For next step, prob not filter out language variable with step_zv, but make the n/a be english (not super sure how, but I think it's doable)

# Use  𝑘 -fold cross validation to fit a linear regression model with all predictor variables included in your recipe. Report the average RMSE across folds. Do not use regularized regression methods (yet).

set.seed(89)
cv_splits <- vfold_cv(d_train)

# model with parsnip package
mod_linear <- linear_reg() %>%
  set_engine("lm") %>%
  set_mode("regression") 

# fitting the linear model #1
mod_linear_fit_1 <- tune::fit_resamples(
  mod_linear,
  preprocessor = rec,
  resamples = cv_splits,
  metrics = yardstick::metric_set(rmse),
  control = tune::control_resamples(verbose = TRUE,
                                    save_pred = TRUE))
# metrics for linear model #1
mod_linear_fit_1 %>%
  collect_metrics() %>%
  filter(.metric == "rmse")


####################################################################
# more feature engineering
rec_2 <- recipe(score ~ ., d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(contains("id"), ncessch, new_role = "id_vars") %>%
  step_zv(all_predictors()) %>%
  step_unknown(all_nominal()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors()) %>%
  step_interact(terms = ~lat:lon)

prepped_rec_2 <- prep(rec_2)
prepped_rec_2

rec_2 %>% 
  prep %>% 
  bake(d_train) 


# fitting the linear model #2
mod_linear_fit_2 <- tune::fit_resamples(
  mod_linear,
  preprocessor = rec_2,
  resamples = cv_splits,
  metrics = yardstick::metric_set(rmse),
  control = tune::control_resamples(verbose = TRUE,
                                    save_pred = TRUE))
# metrics for linear model #2
mod_linear_fit_2 %>%
  collect_metrics() %>%
  filter(.metric == "rmse")


########### recipe 3 ###########
rec_3 <- recipe(score ~ ., d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt)) %>%
  update_role(contains("id"), ncessch, new_role = "id_vars") %>%
  step_unknown(all_nominal()) %>% # changed order of step_unknown to keep lang_cd
  step_zv(all_predictors()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors(), freq_cut = 98/2, unique_cut = 2) %>%
  step_interact(terms = ~lat:lon)

prepped_rec_3 <- prep(rec_3)
prepped_rec_3

rec_3 %>% 
  prep %>% 
  bake(d_train) 


# fitting the linear model #3
mod_linear_fit_3 <- tune::fit_resamples(
  mod_linear,
  preprocessor = rec_3,
  resamples = cv_splits,
  metrics = yardstick::metric_set(rmse),
  control = tune::control_resamples(verbose = TRUE,
                                    save_pred = TRUE)) # it says that rank-deficient model may be misleading

# metrics for linear model #3
mod_linear_fit_3 %>%
  collect_metrics() %>%
  filter(.metric == "rmse") # rmse decreased but it may be misleading because rank-deficient model.

########### recipe 4 ###########
rec_4 <- recipe(score ~ ., d_train) %>%
  step_mutate(tst_dt = lubridate::mdy_hms(tst_dt),
              lang_cd = factor(lang_cd),
              lang_cd = replace_na(lang_cd, "E")) %>% # trying to replace na to english to keep this variable
  update_role(contains("id"), ncessch, new_role = "id_vars") %>%
  step_unknown(all_nominal()) %>%
  step_zv(all_predictors()) %>%
  step_medianimpute(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_center(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_scale(all_numeric(), -all_outcomes(), -has_role("id_vars")) %>%
  step_dummy(all_nominal()) %>%
  step_nzv(all_predictors()) %>%
  step_interact(terms = ~lat:lon)

prepped_rec_4 <- prep(rec_4)
prepped_rec_4

rec_4 %>% 
  prep %>% 
  bake(d_train) # the mutate for lnag_cd didn't work

# fitting the linear model #4
mod_linear_fit_4 <- tune::fit_resamples(
  mod_linear,
  preprocessor = rec_4,
  resamples = cv_splits,
  metrics = yardstick::metric_set(rmse),
  control = tune::control_resamples(verbose = TRUE,
                                    save_pred = TRUE)) # problem with Problem with `mutate()` input `lang_cd`.

# metrics for linear model #4
mod_linear_fit_4 %>%
  collect_metrics() %>%
  filter(.metric == "rmse") # probably misleading rmse because of problem with `mutate()` input `lang_cd`.

