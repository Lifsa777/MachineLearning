# Generation de donnees

set.seed(123)

# Definition des parametres pour la generation des donnees
n_samples <- 1000
mean_chambre <- 3.5
sd_chambre <- 0.8
mean_service <- 4.0
sd_service <- 0.6
mean_restauration <- 3.8
sd_restauration <- 0.7
intercept <- -5
coef_chambre <- 0.8
coef_service <- 1.2
coef_restauration <- 0.5
coef_chambre_carre <- -0.1
coef_service_carre <- -0.15
coef_restauration_carre <- -0.08
bruit_explicatives <- 0.2
bruit_cible <- 0.1

# Generation des notes de qualite pour la chambre, le service et la restauration
chambre <- rnorm(n_samples, mean_chambre, sd_chambre)
service <- rnorm(n_samples, mean_service, sd_service)
restauration <- rnorm(n_samples, mean_restauration, sd_restauration)

# Ajout de termes non lineaires
chambre_carre <- chambre^2
service_carre <- service^2
restauration_carre <- restauration^2

# Ajout de variables categorielles
type_chambre <- sample(c("Standard", "Deluxe", "Suite"), n_samples, replace = TRUE, prob = c(0.6, 0.3, 0.1))
saison <- sample(c("Haute", "Basse"), n_samples, replace = TRUE, prob = c(0.4, 0.6))

# Encodage des variables categorielles
type_chambre_encode <- model.matrix(~ type_chambre - 1)
saison_encode <- model.matrix(~ saison - 1)

# Calcul de la probabilite de satisfaction basee sur un modele logistique plus complexe
logit_prob <- intercept + coef_chambre * chambre + coef_service * service + coef_restauration * restauration +
  coef_chambre_carre * chambre_carre + coef_service_carre * service_carre + coef_restauration_carre * restauration_carre +
  type_chambre_encode[, 2] + type_chambre_encode[, 3] + saison_encode[, 2] +
  rnorm(n_samples, 0, bruit_explicatives)

prob_satisfaction <- 1 / (1 + exp(-logit_prob))

# Desequilibrage des classes
prob_satisfaction <- ifelse(prob_satisfaction >= 0.8, 0.95, prob_satisfaction)
prob_satisfaction <- ifelse(prob_satisfaction <= 0.2, 0.05, prob_satisfaction)

# Generation de la variable de satisfaction basee sur la probabilite calculee avec ajout de bruit
satisfaction <- rbinom(n_samples, size = 1, prob = prob_satisfaction + rnorm(n_samples, 0, bruit_cible))

# Creation du data frame avec les donnees generees
hotel_data <- data.frame(chambre, service, restauration, type_chambre, saison, satisfaction)

# Aperçu des 5 premieres lignes du jeu de donnees
head(hotel_data, 5)

# Sauvegarde des donnees dans un fichier CSV
write.csv(hotel_data, "hotel_data_complex.csv", row.names = FALSE)