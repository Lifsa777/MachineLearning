# Modele bayesien et tests d'autres modeles

# Installation de packages
install.packages(c("rjags","coda", "ggplot2", "caret", "xgboost", "rpart","randomForest"))

library(coda)
library(rjags)
library(ggplot2)
library(caret)
library(xgboost)
library(randomForest)
library(rpart)


# Preparation des donnees
X <- hotel_data[, c("chambre", "service", "restauration", "type_chambre", "saison")]
X <- model.matrix(~ . - 1, data = X)  # Encodage des variables catÃ©gorielles
y <- hotel_data$satisfaction
y <- as.factor(y)

# Division des donnees en ensembles d'entrainement et de test
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Definition du modele bayesien
model_code <- "
model {
  for (i in 1:N) {
    y[i] ~ dbern(p[i])
    logit(p[i]) <- beta[1] + beta[2] * X[i, 1] + beta[3] * X[i, 2] + beta[4] * X[i, 3] +
                    beta[5] * X[i, 4] + beta[6] * X[i, 5] + beta[7] * X[i, 6] + beta[8] * X[i, 7]
  }
  
  beta[1] ~ dnorm(0, 0.1)
  beta[2] ~ dnorm(1, 0.5)
  beta[3] ~ dnorm(1, 0.5)
  beta[4] ~ dnorm(0.5, 0.5)
  beta[5] ~ dnorm(0, 0.1)
  beta[6] ~ dnorm(0, 0.1)
  beta[7] ~ dnorm(0, 0.1)
  beta[8] ~ dnorm(0, 0.1)
}
"

# Configuration des parametres MCMC
data_list <- list(N = nrow(X_train), X = X_train, y = as.numeric(y_train) - 1)
n_chains <- 3
n_iter <- 5000
n_burnin <- 1000
n_thin <- 2

# Entrainement du modele bayesien
model <- jags.model(textConnection(model_code), data = data_list, n.chains = n_chains)
update(model, n.iter = n_burnin)
samples <- coda.samples(model, variable.names = c("beta"), n.iter = n_iter, thin = n_thin)

# Prediction sur l'ensemble de test
X_test_matrix <- as.matrix(X_test)
n_test <- nrow(X_test_matrix)
if (nrow(X_test) != length(y_test)) {
  stop("Le nombre d'observations dans X_test et y_test ne correspond pas.")
}
pred_bayes_samples <- matrix(0, nrow = nrow(samples[[1]]), ncol = n_test)
for (i in 1:nrow(samples[[1]])) {
  beta <- samples[[1]][i, ]
  logit_pred <- beta[1] + X_test_matrix %*% beta[-1]
  pred_bayes_samples[i, ] <- plogis(logit_pred)
}
pred_bayes <- apply(pred_bayes_samples, 2, mean)
pred_bayes <- ifelse(pred_bayes > 0.5, "1", "0")

# Evaluation des performances
levels_order <- c("0", "1")
pred_bayes <- factor(pred_bayes, levels = levels_order)
y_test_bayes <- as.character(y_test)
y_test_bayes <- factor(y_test_bayes, levels = levels(pred_bayes))
confusionMatrix(pred_bayes, y_test_bayes)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Definir une graine pour la reproductibilite
set.seed(123)

# Supposons que 'y' est un vecteur de 0 et 1
# Creer l'index pour la partition des donnees
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)

# Diviser les donnees en ensembles d'entrainement et de test
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Creer des data frames pour l'entrainement et le test
train_data <- data.frame(X_train, y = y_train)
test_data <- data.frame(X_test)

# Ajuster le modele de regression logistique
model_glm <- glm(y ~ ., data = train_data, family = binomial)

# Predire sur l'ensemble de test
pred_glm <- predict(model_glm, newdata = test_data, type = "response")
pred_glm <- ifelse(pred_glm > 0.5, "satisfait", "insatisfait")

# Definir les niveaux de façon coherente
niveaux_ordre <- c("insatisfait", "satisfait")
pred_glm <- factor(pred_glm, levels = niveaux_ordre)
y_test_glm <- ifelse(y_test == 0, "insatisfait", "satisfait")
y_test_glm <- factor(y_test_glm, levels = niveaux_ordre)

# Creer la matrice de confusion
confusionMatrix(pred_glm, y_test_glm)  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------

# Creer des data frames pour l'entrainement et le test
train_data <- data.frame(X_train, y = factor(ifelse(y_train == 0, "insatisfait", "satisfait")))
test_data <- data.frame(X_test)

# Modele d'arbre de decision
model_arbre <- rpart(y ~ ., data = train_data)
pred_arbre <- predict(model_arbre, newdata = test_data, type = "class")

# Convertir y_test en facteur avec les memes niveaux que pred_arbre
niveaux_ordre <- levels(train_data$y)
y_test_factor <- factor(ifelse(y_test == 0, "insatisfait", "satisfait"), levels = niveaux_ordre)

# S'assurer que pred_arbre a les memes niveaux
pred_arbre <- factor(pred_arbre, levels = niveaux_ordre)

# Matrice de confusion pour l'arbre de decision
confusionMatrix(pred_arbre, y_test_factor)

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Verifier et traiter les valeurs manquantes dans les donnees d'entrainement
train_data <- na.omit(train_data)
# Modele de foret aleatoire
model_fa <- randomForest(y ~ ., data = train_data, na.action = na.roughfix)
# Verifier et traiter les valeurs manquantes dans les donnees de test
test_data <- na.roughfix(test_data)
# Prediction
pred_fa <- predict(model_fa, newdata = test_data)
# S'assurer que pred_fa a les memes niveaux
pred_fa <- factor(pred_fa, levels = niveaux_ordre)
# Matrice de confusion pour la foret aleatoire
confusionMatrix(pred_fa, y_test_factor)  

#--------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
# Nettoyage des donnees
clean_labels <- function(y) {
  # Convertir en caractere puis en numerique
  y_num <- as.numeric(as.character(y))
  
  # Remplacer les NA par 0
  y_num[is.na(y_num)] <- 0
  
  # Assurer que toutes les valeurs sont 0 ou 1
  y_num <- ifelse(y_num > 0, 1, 0)
  
  return(y_num)
}

y_train_clean <- clean_labels(y_train)
y_test_clean <- clean_labels(y_test)

# Verifier a nouveau apres nettoyage
cat("\ny_train nettoye:\n")
print(table(y_train_clean))
cat("\ny_test nettoye:\n")
print(table(y_test_clean))

# Preparation des donnees pour XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_clean)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test_clean)

# Definition des parametres du modele
params <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 3,
  eta = 0.1,
  nthread = 2,
  nrounds = 50
)

# Entrainement du modele XGBoost
model_xgb <- xgb.train(params = params, data = dtrain, nrounds = 50)

# Predictions
pred_xgb <- predict(model_xgb, newdata = dtest)
pred_xgb <- ifelse(pred_xgb > 0.5, "satisfait", "insatisfait")

# Conversion des predictions et des valeurs reelles en facteurs
niveaux_ordre <- c("insatisfait", "satisfait")
pred_xgb <- factor(pred_xgb, levels = niveaux_ordre)
y_test_factor <- factor(ifelse(y_test_clean == 0, "insatisfait", "satisfait"), levels = niveaux_ordre)

# Matrice de confusion pour XGBoost
confusionMatrix(pred_xgb, y_test_factor)