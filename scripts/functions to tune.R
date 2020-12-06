# function to tune random forest model

mtry_search <- seq(22, 26, 2)
min_n_search <- seq(2, 6, 2)

grd <- expand.grid(mtry_search, min_n_search)

mtry_rf_search <- function(mtry_val, min_n_val, wf) {
  mod <- rand_forest() %>% 
    set_engine("ranger",
               num.threads = cores,
               importance = "permutation",
               verbose = TRUE) %>% 
    set_mode("regression") %>% 
    set_args(mtry = {{mtry_val}},
             min_n = {{min_n_val}})
  
  wf <- wf %>% 
    update_model(mod)
  
  rmse <- fit(wf, data_train) %>% 
    extract_rmse()
  
  tibble(mtry = mtry_val, min_n = min_n_val, rmse = rmse, workflow = list(wf))
}

mtry_results <- map2_df(grd$Var1, grd$Var2, ~mtry_rf_search(.x, .y, rf_def_wkflw))

mtry_results %>% 
  arrange(rmse)