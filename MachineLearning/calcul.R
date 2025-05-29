# Getting things ready: loading up all the tools and packages we'll need.
library(DBI)
library(RMariaDB)
library(dplyr)
library(dbplyr)
library(caret)
# library(randomForest) # Random Forest is being removed
library(pROC)
library(ggplot2)
library(e1071)
library(class) # For kNN
library(rpart) # For Decision Trees
library(rpart.plot) # For plotting decision trees
library(cluster)
library(factoextra)

# Heads up! You'll need to fill in your database details here.
db_host <- "127.0.0.1"
db_port <- 3306
db_name <- "dw_final"
db_user <- "root"
db_password <- ""

# This is where we'll grab the data from the database.
sql_query <- "
SELECT
    FB.lead_time, FB.adults, FB.children, FB.babies,
    FB.stays_in_weekend_nights, FB.stays_in_week_nights,
    FB.adr, FB.days_in_waiting_list, FB.booking_changes,
    FB.parking_spaces, FB.special_requests,
    DH.hotel_type,
    DG.country, DG.is_repeated_guest, DG.previous_cancellations, DG.previous_bookings_not_cancelled,
    DR_res.room_type AS reserved_room_type,
    DM.meal_type,
    DSC.distribution_channel, DSC.market_segment,
    DD.deposit_type,
    DBS.booking_status,
    DT_arr.month_name AS arrival_month,
    DT_arr.year_num AS arrival_year
FROM
    fact_booking FB
JOIN dim_hotel DH ON FB.id_hotel = DH.id_hotel
JOIN dim_guest DG ON FB.id_guest = DG.id_guest
JOIN dim_room DR_res ON FB.id_reserved_room = DR_res.id_room
JOIN dim_meal DM ON FB.id_meal = DM.id_meal
JOIN dim_sales_channel DSC ON FB.id_sales_channel = DSC.id_channel
JOIN dim_deposit DD ON FB.id_deposit = DD.id_deposit
JOIN dim_booking_statuts DBS ON FB.id_booking_status = DBS.id_status
JOIN dim_time DT_arr ON FB.id_arrival_date = DT_arr.id_date
WHERE DBS.booking_status IN ('Canceled', 'Check-Out');
"

# Let's try to connect to the database and fetch the data.
# If it doesn't work, we'll use some sample data instead.
hotel_data_raw <- data.frame()
message("Attempting to connect to the database...")
tryCatch({
  con <- dbConnect(RMariaDB::MariaDB(),
                   host = db_host,
                   port = db_port,
                   dbname = db_name,
                   user = db_user,
                   password = db_password)
  message("Database connection successful.")
  
  message("Fetching data with SQL query...")
  hotel_data_raw <- dbGetQuery(con, sql_query)
  message(paste("Successfully fetched", nrow(hotel_data_raw), "rows and", ncol(hotel_data_raw), "columns."))
  
}, error = function(e) {
  message("ERROR during database connection or query: ", e$message)
  message("Continuing with built-in sample data for demonstration.")
}, finally = {
  if (!is.null(con) && dbIsValid(con)) {
    dbDisconnect(con)
    message("Database connection closed.")
  }
})

# If we couldn't get data from the database, we'll create some random sample data to work with.
if (!exists("hotel_data_raw") || nrow(hotel_data_raw) == 0) {
  message("Using sample data for demonstration as actual data was not loaded from the database.")
  set.seed(123)
  n_sample <- 500
  hotel_data_raw <- data.frame(
    lead_time = sample(0:365, n_sample, replace = TRUE),
    adults = sample(1:4, n_sample, replace = TRUE, prob = c(0.7, 0.2, 0.05, 0.05)),
    children = sample(0:3, n_sample, replace = TRUE, prob = c(0.8, 0.15, 0.04, 0.01)),
    babies = sample(0:2, n_sample, replace = TRUE, prob = c(0.9, 0.08, 0.02)),
    stays_in_weekend_nights = sample(0:4, n_sample, replace = TRUE),
    stays_in_week_nights = sample(0:10, n_sample, replace = TRUE),
    adr = round(runif(n_sample, 30, 500), 2),
    days_in_waiting_list = sample(0:50, n_sample, replace = TRUE, prob = c(0.9, rep(0.1/50, 50))),
    booking_changes = sample(0:5, n_sample, replace = TRUE, prob = c(0.8, 0.1, 0.05, 0.03, 0.01, 0.01)),
    parking_spaces = sample(0:1, n_sample, replace = TRUE, prob = c(0.94, 0.06)),
    special_requests = sample(0:3, n_sample, replace = TRUE, prob = c(0.5, 0.3, 0.15, 0.05)),
    hotel_type = sample(c("City Hotel", "Resort Hotel"), n_sample, replace = TRUE, prob = c(0.6, 0.4)),
    country = sample(c("PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "USA", "Other"), n_sample, replace = TRUE, prob = c(0.25,0.15,0.1,0.1,0.1,0.05,0.05,0.2)),
    is_repeated_guest = sample(c(0, 1), n_sample, replace = TRUE, prob = c(0.96, 0.04)),
    previous_cancellations = sample(0:3, n_sample, replace = TRUE, prob = c(0.92, 0.05, 0.02, 0.01)),
    previous_bookings_not_cancelled = sample(0:10, n_sample, replace = TRUE, prob = c(0.85,0.05,0.03,0.02,0.01,0.01,0.01,0.01,0.005,0.005,0.00)),
    reserved_room_type = sample(LETTERS[1:8], n_sample, replace = TRUE),
    meal_type = sample(c("BB", "HB", "SC", "FB", "Undefined"), n_sample, replace = TRUE),
    distribution_channel = sample(c("TA/TO", "Direct", "Corporate", "GDS"), n_sample, replace = TRUE),
    market_segment = sample(c("Online TA", "Offline TA/TO", "Direct", "Groups", "Corporate"), n_sample, replace = TRUE),
    deposit_type = sample(c("No Deposit", "Non Refund", "Refundable"), n_sample, replace = TRUE, prob = c(0.85, 0.14, 0.01)),
    booking_status = sample(c("Canceled", "Check-Out"), n_sample, replace = TRUE, prob = c(0.37, 0.63)),
    arrival_month = sample(month.name, n_sample, replace = TRUE),
    arrival_year = sample(c(2015, 2016, 2017), n_sample, replace = TRUE)
  )
}

# Now we clean up the data: create our target variable 'is_canceled',
# handle any missing values, and make sure data types are correct for modeling.
if (nrow(hotel_data_raw) > 0) {
  hotel_data <- hotel_data_raw
  hotel_data$is_canceled <- as.factor(ifelse(hotel_data$booking_status == "Canceled", 1, 0))
  hotel_data$booking_status <- NULL
  
  for(col_name in names(hotel_data)){
    if(is.numeric(hotel_data[[col_name]])){
      if(any(is.na(hotel_data[[col_name]]))) hotel_data[[col_name]][is.na(hotel_data[[col_name]])] <- median(hotel_data[[col_name]], na.rm = TRUE)
    } else if (is.factor(hotel_data[[col_name]]) || is.character(hotel_data[[col_name]])) {
      if(any(is.na(hotel_data[[col_name]]))) {
        if(is.factor(hotel_data[[col_name]]) && !"Unknown" %in% levels(hotel_data[[col_name]])) {
          levels(hotel_data[[col_name]]) <- c(levels(hotel_data[[col_name]]), "Unknown")
        }
        hotel_data[[col_name]][is.na(hotel_data[[col_name]])] <- "Unknown"
      }
    }
  }
  
  for (col_name in names(hotel_data)) {
    if (is.character(hotel_data[[col_name]])) {
      hotel_data[[col_name]] <- as.factor(hotel_data[[col_name]])
    }
  }
  if("is_repeated_guest" %in% names(hotel_data) && is.numeric(hotel_data$is_repeated_guest)) {
    hotel_data$is_repeated_guest <- as.factor(hotel_data$is_repeated_guest)
  }
  
  message("Data preprocessing complete.")
} else {
  message("No data loaded. Skipping further processing and modeling.")
}

# We'll store the accuracy of our models here.
accuracy_results <- data.frame(Algorithme = character(), Accuracy = numeric(), stringsAsFactors = FALSE)

# Time to build some models!
# First, we split our data into a training set (to build the models)
# and a test set (to see how well they perform on unseen data).
if (exists("hotel_data") && nrow(hotel_data) > 50) {
  set.seed(42) # So we get the same split every time
  trainIndex <- createDataPartition(hotel_data$is_canceled, p = .8, list = FALSE, times = 1)
  data_train <- hotel_data[trainIndex,]
  data_test <- hotel_data[-trainIndex,]
  
  # Making sure our target variable 'is_canceled' is in the right format for caret.
  levels(data_train$is_canceled) <- make.names(levels(data_train$is_canceled))
  levels(data_test$is_canceled) <- make.names(levels(data_test$is_canceled))
  
  # Setting up some common rules for how we train our models.
  train_control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = FALSE, allowParallel = TRUE)
  
  # Random Forest section has been removed.
  
  # Next up, K-Nearest Neighbors (K-NN).
  message("\n--- Training K-Nearest Neighbors (K-NN) ---")
  model_knn <- NULL
  tryCatch({
    tune_grid_knn <- expand.grid(.k = seq(3, 15, by = 2))
    
    model_knn <- train(is_canceled ~ ., data = data_train, method = "knn",
                       trControl = train_control,
                       preProcess = c("center", "scale", "nzv"), 
                       tuneGrid = tune_grid_knn,
                       metric = "ROC") 
    
    pred_knn_class <- predict(model_knn, newdata = data_test)
    cm_knn <- confusionMatrix(pred_knn_class, data_test$is_canceled, positive = make.names("1"))
    accuracy_results <- rbind(accuracy_results, data.frame(Algorithme = "K-NN", Accuracy = cm_knn$overall['Accuracy']))
    print(cm_knn)
  }, error = function(e) message("Error in K-NN: ", e$message))
  
  # Now for a Decision Tree ("Arbre manuel").
  message("\n--- Training Decision Tree (rpart) ---")
  model_dt <- NULL
  tryCatch({
    tune_grid_dt <- expand.grid(.cp = seq(0.001, 0.02, by = 0.002))
    
    model_dt <- train(is_canceled ~ ., data = data_train, method = "rpart",
                      trControl = train_control,
                      tuneGrid = tune_grid_dt,
                      metric = "ROC") 
    
    # Optional: Plot the best tree 
    # if (!is.null(model_dt$finalModel)) {
    #   prp(model_dt$finalModel, box.palette = "Reds", tweak = 1.2)
    # }
    
    pred_dt_class <- predict(model_dt, newdata = data_test)
    cm_dt <- confusionMatrix(pred_dt_class, data_test$is_canceled, positive = make.names("1"))
    accuracy_results <- rbind(accuracy_results, data.frame(Algorithme = "Arbre manuel (Decision Tree)", Accuracy = cm_dt$overall['Accuracy']))
    print(cm_dt)
  }, error = function(e) message("Error in Decision Tree: ", e$message))
  
  # Let's see how K-Means clustering aligns with our cancellation classes.
  # This is a bit different as K-Means is for finding groups, not direct prediction.
  message("\n--- Evaluating K-Means for Classification Alignment ---")
  tryCatch({
    data_test_kmeans <- data_test
    
    true_labels_kmeans <- data_test_kmeans$is_canceled
    data_test_kmeans$is_canceled <- NULL 
    
    dmy <- dummyVars(" ~ .", data = data_test_kmeans, fullRank = TRUE)
    data_test_numeric <- data.frame(predict(dmy, newdata = data_test_kmeans))
    
    data_test_scaled <- scale(data_test_numeric)
    
    data_test_scaled <- data_test_scaled[, colSums(is.na(data_test_scaled)) == 0]
    
    if (ncol(data_test_scaled) > 1 && nrow(data_test_scaled) > 2) {
      set.seed(123) 
      kmeans_result <- kmeans(data_test_scaled, centers = 2, nstart = 25)
      
      cluster_vs_true <- table(kmeans_result$cluster, true_labels_kmeans)
      
      correct_opt1 <- 0
      correct_opt2 <- 0
      
      if("X0" %in% colnames(cluster_vs_true) && "X1" %in% colnames(cluster_vs_true) && 
         1 %in% rownames(cluster_vs_true) && 2 %in% rownames(cluster_vs_true)){
        correct_opt1 <- cluster_vs_true["1", "X0"] + cluster_vs_true["2", "X1"]
        correct_opt2 <- cluster_vs_true["1", "X1"] + cluster_vs_true["2", "X0"]
      } else {
        message("Warning: Could not find expected labels (X0, X1) or clusters (1,2) in K-Means results for accuracy calculation. This can happen with very small or skewed test sets.")
      }
      
      kmeans_accuracy <- 0
      if(sum(cluster_vs_true) > 0) { 
        kmeans_accuracy <- max(correct_opt1, correct_opt2) / sum(cluster_vs_true)
      } else {
        kmeans_accuracy <- NA 
      }
      
      accuracy_results <- rbind(accuracy_results, data.frame(Algorithme = "K-Means (aligned)", Accuracy = kmeans_accuracy))
      message(paste("K-Means (aligned) Accuracy:", round(kmeans_accuracy, 6)))
      print("Cluster vs True Labels table:")
      print(cluster_vs_true)
    } else {
      message("Not enough valid columns/rows for K-Means after preprocessing.")
      accuracy_results <- rbind(accuracy_results, data.frame(Algorithme = "K-Means (aligned)", Accuracy = NA))
    }
    
  }, error = function(e) {
    message("Error in K-Means evaluation: ", e$message)
    accuracy_results <- rbind(accuracy_results, data.frame(Algorithme = "K-Means (aligned)", Accuracy = NA))
  })
  
} else {
  message("Skipping model training as data is insufficient or was not loaded.")
}

# Let's see a summary of how accurate each model was.
message("\n\n===== Tableau récapitulatif des accuracies =====")
if (nrow(accuracy_results) > 0) {
  accuracy_results_cleaned <- accuracy_results[!is.na(accuracy_results$Accuracy),]
  
  if(nrow(accuracy_results_cleaned) > 0){
    # Updated desired_order with Decision Tree and without Random Forest
    desired_order <- c("K-Means (aligned)", "K-NN", "Arbre manuel (Decision Tree)") 
    
    current_algos <- unique(accuracy_results_cleaned$Algorithme)
    levels_for_factor <- intersect(desired_order, current_algos)
    levels_for_factor <- c(levels_for_factor, setdiff(current_algos, levels_for_factor))
    
    accuracy_results_cleaned$Algorithme <- factor(accuracy_results_cleaned$Algorithme, levels = levels_for_factor, ordered = TRUE)
    accuracy_results_cleaned <- accuracy_results_cleaned[order(accuracy_results_cleaned$Algorithme), ]
    
    accuracy_results_cleaned$Algorithme <- as.character(accuracy_results_cleaned$Algorithme)
    
    print(accuracy_results_cleaned, row.names = FALSE)
    
    # And the winner is... Let's find the algorithm with the best accuracy.
    if(nrow(accuracy_results_cleaned) > 0){
      best_model_row <- accuracy_results_cleaned[which.max(accuracy_results_cleaned$Accuracy), ]
      message(paste("\nLe meilleur algorithme basé sur l'accuracy est :", best_model_row$Algorithme, 
                    "avec une accuracy de", round(best_model_row$Accuracy, 4)))
    } else {
      message("\nCould not determine the best algorithm as no valid accuracy results were found after cleaning.")
    }
    
  } else {
    message("No accuracy results to display after cleaning (all models might have failed or produced NA).")
  }
} else {
  message("No accuracy results to display.")
}


