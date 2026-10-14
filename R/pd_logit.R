# Cross-check in R: plain logistic regression (dummy-coded factors) on the same
# train/test split used in the Python notebook.
# Run from the repository root after the notebook:  Rscript R/pd_logit.R
# Base R only, no extra packages needed.

loans <- read.csv("data/raw/germancredit.csv", stringsAsFactors = TRUE)
loans$bad <- as.integer(loans$creditability == "bad")
loans$creditability <- NULL
loans$personal_status_and_sex <- NULL   # excluded for fairness, as in the notebook
loans$foreign_worker <- NULL

test_rows <- read.csv("outputs/test_index.csv")$row + 1   # Python index is 0-based
train <- loans[-test_rows, ]
test  <- loans[test_rows, ]

fit <- glm(bad ~ ., data = train, family = binomial(link = "logit"))
cat(sprintf("Training observations: %d | AIC: %.1f\n", nrow(train), AIC(fit)))

# AUC via the Mann-Whitney statistic (no packages needed)
auc <- function(y, p) {
  r <- rank(p)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
p_test <- predict(fit, newdata = test, type = "response")
a <- auc(test$bad, p_test)
cat(sprintf("Test AUC: %.3f | Gini: %.3f\n", a, 2 * a - 1))

# The ten most significant coefficients
co <- summary(fit)$coefficients
co <- co[order(co[, 4]), ][1:10, ]
print(round(co, 3))

write.csv(data.frame(model = "R glm, dummy-coded", test_auc = a, test_gini = 2 * a - 1),
          "outputs/r_glm_result.csv", row.names = FALSE)
