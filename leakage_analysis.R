# =============================================================================
# Leakage analysis: is "Hazardous" defined by the predictors?
# Author: Fernando Ricaurte (extension to the STA 5703 team project)
#
# NASA labels an asteroid a Potentially Hazardous Asteroid (PHA) when:
#   Minimum Orbit Intersection Distance (MOID) <= 0.05 AU   AND
#   Absolute Magnitude (H)                     <= 22.0
# Both quantities are predictors in our dataset, so the target may be a
# deterministic rule applied to two features (target leakage by definition).
#
# This script answers two questions:
#   1. How well does NASA's rule alone, with no model, reproduce the label?
#   2. How well can a model predict the label WITHOUT the defining features?
#
# Run from the repository root:  Rscript leakage_analysis.R
# =============================================================================

suppressPackageStartupMessages({
  library(readxl); library(dplyr); library(caret); library(randomForest); library(pROC)
})

# ---- 1. Load data and recreate the exact split used in analysis.Rmd ---------
asteroids <- as.data.frame(read_excel("data/cleaned_neows_1950_2025.xlsx"))

drop <- c("Date", "Neo.Reference.ID", "Close.Approach.Date", "Orbit.Determination.Date",
          "Epoch.Date.Close.Approach", "Orbit.ID", "Perihelion.Time", "Orbiting.Body")
model_data <- asteroids[, !(names(asteroids) %in% drop)]
predictors <- setdiff(names(model_data), "Hazardous")
model_data <- model_data %>% mutate(across(all_of(predictors), ~ as.numeric(as.character(.))))
model_data$Hazardous <- factor(
  ifelse(as.character(model_data$Hazardous) %in% c("True", "TRUE"), "TRUE", "FALSE"),
  levels = c("FALSE", "TRUE"))

set.seed(5703)
train_idx <- createDataPartition(model_data$Hazardous, p = 0.6, list = FALSE)
train     <- model_data[train_idx, ]
heldout   <- model_data[-train_idx, ]
val_idx   <- createDataPartition(heldout$Hazardous, p = 0.5, list = FALSE)
val       <- heldout[val_idx, ]
test      <- heldout[-val_idx, ]
minority_n <- min(table(train$Hazardous))

# Helper: one row of test-set metrics for a set of class predictions
metrics_row <- function(name, pred_class, truth, prob = NULL) {
  cm <- confusionMatrix(pred_class, truth, positive = "TRUE")
  auc_value <- if (is.null(prob)) NA else
    as.numeric(auc(roc(truth, prob, levels = c("FALSE", "TRUE"), direction = "<", quiet = TRUE)))
  data.frame(Model = name,
             Accuracy    = round(cm$overall["Accuracy"], 4),
             Sensitivity = round(cm$byClass["Sensitivity"], 4),
             Specificity = round(cm$byClass["Specificity"], 4),
             Precision   = round(cm$byClass["Pos Pred Value"], 4),
             Kappa       = round(cm$overall["Kappa"], 4),
             F1          = round(cm$byClass["F1"], 4),
             AUC         = round(auc_value, 4),
             FN = cm$table["FALSE", "TRUE"], FP = cm$table["TRUE", "FALSE"],
             row.names = NULL)
}

# ---- 2. NASA's two-cutoff rule, no model ----------------------------------
nasa_rule <- function(d) {
  factor(ifelse(d$Minimum.Orbit.Intersection <= 0.05 & d$Absolute.Magnitude <= 22,
                "TRUE", "FALSE"), levels = c("FALSE", "TRUE"))
}

rule_all <- mean(nasa_rule(model_data) == model_data$Hazardous)
cat(sprintf("NASA rule matches the label on %.2f%% of all %d objects.\n\n",
            100 * rule_all, nrow(model_data)))

results <- metrics_row("NASA rule (no model)", nasa_rule(test), test$Hazardous)

# ---- 3. Random forest WITHOUT the defining features -------------------------
# Estimated diameter is computed from absolute magnitude, so it is removed too.
defining <- c("Minimum.Orbit.Intersection", "Absolute.Magnitude", "Est.Dia.in.KM.max.")
train_reduced <- train[, !(names(train) %in% defining)]

set.seed(5703)
rf_reduced <- randomForest(Hazardous ~ ., data = train_reduced, ntree = 500, mtry = 3,
                           sampsize = c("FALSE" = minority_n, "TRUE" = minority_n),
                           importance = TRUE)

# Choose the threshold on the validation set, then apply it unchanged to test
val_prob  <- predict(rf_reduced, val, type = "prob")[, "TRUE"]
threshold <- coords(roc(val$Hazardous, val_prob, quiet = TRUE), "best",
                    ret = "threshold", transpose = FALSE)$threshold[1]
test_prob <- predict(rf_reduced, test, type = "prob")[, "TRUE"]
test_pred <- factor(ifelse(test_prob >= threshold, "TRUE", "FALSE"), levels = c("FALSE", "TRUE"))

results <- rbind(results,
                 metrics_row("Random forest without MOID, H, diameter", test_pred,
                             test$Hazardous, test_prob))

cat("=== Test set comparison ===\n")
print(results)
cat(sprintf("\nValidation-selected threshold for the reduced model: %.3f\n", threshold))

cat("\nTop predictors once the defining features are removed:\n")
print(round(head(sort(importance(rf_reduced)[, "MeanDecreaseGini"], decreasing = TRUE), 6), 1))

write.csv(results, "leakage_results.csv", row.names = FALSE)
