library(quantmod)
library(fGarch)
library(rugarch) 
library(sarima)
library(FinTS)
library(tseries)
library(forecast)
library(stats)
library(fBasics);
library(car)
library(PerformanceAnalytics)
library(TSA)
library(astsa)
library(tibble)
library(neuralnet)
library(caret)
library(arules)

library(anomalize)
data<-read.csv("E:\\PROJECTWORK\\rain3.csv")
tsdata<-ts(data$Rainfall,start =c(1901,1),end=c(2021,12),frequency = 12)
start(tsdata)
end(tsdata)
glimpse(tsdata) 

summary(tsdata)

plot(tsdata, main = "Graph of Rainfall", 
     col.main = "darkgreen") 
# Convert  monthly rainfall data to matrix format
rainfall_matrix <- matrix(tsdata, ncol = 1)

# Perform k-means clustering
k <- 7# Number of clusters (adjust as needed)
set.seed(123)  # For reproducibility
km_clusters <- kmeans(rainfall_matrix, centers = k)
par(mfrow=c(1,1))
# Visualize the clusters
plot(rainfall_matrix, col = km_clusters$cluster, main = "K-Means Clustering of Original Rainfall Data",
     xlab = "Time", ylab = "Rainfall")
points(km_clusters$centers, col = 1:k, pch = 8, cex = 2)
legend("topright", legend = paste("Cluster", 1:k), col = 1:k, pch = 8)
cluster_counts <- table(km_clusters$cluster)
cluster_counts

# Calculate the proportion of each cluster
cluster_proportions <- cluster_counts / length(km_clusters$cluster)

# Print the proportions
cat("Cluster Percentage:\n")
for (i in 1:k) {
  cat("Cluster", i, ":", cluster_proportions[i]*100,"%", "\n")
}
i=1
# Set up multiple plots in one row
par(mfrow=c(2,4))
for(i in 1:k) {
  cluster_indices <- which(km_clusters$cluster == i)
  cluster_data <- rainfall_matrix[cluster_indices, ]
  plot(cluster_data,type="l", main = paste("Cluster ", i), xlab = "Date", ylab = "Rainfall")
}
# Create empty vectors to store statistics for each cluster
min_values <- numeric(k)
max_values <- numeric(k)
mean_values <- numeric(k)
median_values <- numeric(k)
mode_values <- numeric(k)

# Iterate through each cluster
for (i in 1:k) {
  # Extract rainfall data for the current cluster
  cluster_indices <- which(km_clusters$cluster == i)
  cluster_data <- rainfall_matrix[cluster_indices, ]
  
  # Calculate statistics
  min_values[i] <- min(cluster_data)
  max_values[i] <- max(cluster_data)
  mean_values[i] <- mean(cluster_data)
  median_values[i] <- median(cluster_data)
  mode_values[i] <- as.numeric(names(sort(table(cluster_data), decreasing = TRUE)[1]))
  
  # Print statistics for the current cluster
  cat("\nCluster", i, "Statistics:\n")
  cat("Minimum:", min_values[i], "\n")
  cat("Maximum:", max_values[i], "\n")
  cat("Mean:", mean_values[i], "\n")
  cat("Median:", median_values[i], "\n")
  cat("Mode:", mode_values[i], "\n")
}
library(moments)  # Required for skewness and kurtosis calculations

# Additional statistical measures
standard_deviation <- numeric(k)
variance <- numeric(k)
iqr <- numeric(k)
skewness <- numeric(k)
kurtosis <- numeric(k)

for (i in 1:k) {
  cluster_indices <- which(km_clusters$cluster == i)
  cluster_data <- rainfall_matrix[cluster_indices, ]
  
  standard_deviation[i] <- sd(cluster_data)
  variance[i] <- var(cluster_data)
  iqr[i] <- IQR(cluster_data)
  skewness[i] <- skewness(cluster_data)
  kurtosis[i] <- kurtosis(cluster_data)
  
  cat("\nCluster", i, "Additional Statistical Measures:\n")
  cat("Standard Deviation:", standard_deviation[i], "\n")
  cat("Variance:", variance[i], "\n")
  cat("Interquartile Range (IQR):", iqr[i], "\n")
  cat("Skewness:", skewness[i], "\n")
  cat("Kurtosis:", kurtosis[i], "\n")
}
library(ggplot2)

# Set up multiple plots in one row
par(mfrow=c(2,4))

# Iterate through each cluster
for(i in 1:k) {
  cluster_indices <- which(km_clusters$cluster == i)
  cluster_data <- rainfall_matrix[cluster_indices, ]
  
  # Plot histogram of rainfall distribution for the current cluster
  hist(cluster_data, main = paste("Cluster ", i, " Rainfall Distribution"), 
       xlab = "Rainfall", ylab = "Frequency", col = "lightblue")
  
}


par(mfrow=c(1,1))
# Split tsdata into two partitions (70% and 30%)
total_length <- length(tsdata)
split_point <- round(0.7 * total_length)

# First partition (70%)
tsdata_train <- window(tsdata, end = c(1901 + floor((split_point - 1) / 12), (split_point - 1) %% 12 + 1))

# Second partition (30%)
tsdata_test <- window(tsdata, start = c(1901 + floor(split_point / 12), split_point %% 12 + 1))

# Display the start and end of each partition
cat("Training Data Start:", start(tsdata_train), "\n")
cat("Training Data End:", end(tsdata_train), "\n")
cat("Test Data Start:", start(tsdata_test), "\n")
cat("Test Data End:", end(tsdata_test), "\n")

# Plot the training and test partitions
par(mfrow = c(2, 1))
plot(tsdata_train, main = "Training Data (70% of Total)", col.main = "blue", ylab = "Rainfall")
plot(tsdata_test, main = "Test Data (30% of Total)", col.main = "red", ylab = "Rainfall")
# Fit ARIMA model to historical data
arima_model <- auto.arima(tsdata_train)
summary(arima_model)
tsdiag(arima_model)

# Forecast future values
forecast_values <- forecast(arima_model, h = length(tsdata_test))

# Print the forecasted values
print(forecast_values)
par(mfrow=c(1,1))
plot(forecast_values,main="Figure 3.12:Graph of forecast using ARIMA")
# Extract forecasted values
forecasted <- forecast_values$mean

# Extract actual values
actual <- window(tsdata_test, start = c(1985, 9))

# Calculate residuals
residuals <- actual-forecasted
# Extract date index
date_index <- as.Date(time(actual))
# Create a table
result_table <- data.frame(
  Date = date_index,
  Forecasted = forecasted,
  Actual = actual,
  Residuals = residuals
)

# Print the table
print(result_table)
# Plot forecasted vs actual values
plot(date_index, forecasted, type = "l", col = "blue", ylim = range(c(forecasted, actual)), xlab = "Date", ylab = "Rainfall", main = "Forecasted vs Actual Rainfall for SARIMA")
lines(date_index, actual, col = "red")
legend("topleft", legend = c("Forecasted", "Actual"), col = c("blue", "red"), lty = 1)

# Add a grid
grid()
cop.decom <- stl(tsdata_train, t.window=12, s.window="periodic", robust=TRUE)

plot(cop.decom, main = " STL Decomposition of  training set Rainfall")
summary(cop.decom)

stlde<-forecast(cop.decom,h=length(tsdata_test))
stlde
plot(stlde,main="Forecast using STL Decomposition")

# Extract forecasted values
stlforecasted <- stlde$mean



# Calculate residuals
stlresiduals <-  actual-stlforecasted 
# Extract date index
date_index <- as.Date(time(actual))
# Create a table
stlresult_table <- data.frame(
  Date = date_index,
  Forecasted = stlforecasted,
  Actual = actual,
  Residuals = stlresiduals
)

# Print the table
print(stlresult_table)
# Plot forecasted vs actual values
plot(date_index, stlforecasted, type = "l", col = "blue", ylim = range(c(stlforecasted, actual)), xlab = "Date", ylab = "Rainfall", main = "Forecasted vs Actual Rainfall using STL")
lines(date_index, actual, col = "red")
legend("topleft", legend = c("Forecasted", "Actual"), col = c("blue", "red"), lty = 1)

# Forecast future values using seasonal naive method
forecast_values_seasonal_naive <- snaive(tsdata_train, h = length(tsdata_test))
forecast_values_seasonal_naive
plot(forecast_values_seasonal_naive,main="Forecast using Seasonal Naive",)

# Extract forecasted values from seasonal naive
seasonal_naive <- forecast_values_seasonal_naive$mean
# Calculate residuals for seasonal naive
residuals_seasonal_naive <-  actual-seasonal_naive 
# Create a table for all methods
forecast_result_table <- data.frame( Date = date_index,seasonal_naive = seasonal_naive,
                                     Actual = actual,Residuals_Seasonal_Naive = residuals_seasonal_naive
)
# Print the table for all methods
print(forecast_result_table)
# Plot forecasted vs actual values
plot(date_index, seasonal_naive, type = "l", col = "blue", ylim = range(c(stlforecasted, actual)), xlab = "Date", ylab = "Rainfall", main = "Figure 3.18:Forecasted vs Actual Rainfall")
lines(date_index, actual, col = "red")
legend("topleft", legend = c("Forecasted", "Actual"), col = c("blue", "red"), lty = 1)
# Calculate MAE
mae_arima <- mean(abs(residuals))
mae_stl <- mean(abs(stlresiduals))
mae_seasonal_naive <- mean(abs(residuals_seasonal_naive))

# Calculate RMSE
rmse_arima <- sqrt(mean(residuals^2))
rmse_stl <- sqrt(mean(stlresiduals^2))
rmse_seasonal_naive <- sqrt(mean(residuals_seasonal_naive^2))
# Define a function to calculate MASE
calculate_mase <- function(actual, forecast) {
  n <- length(actual)
  mase <- mean(abs(actual - forecast)) / mean(abs(actual[2:n] - actual[1:(n-1)]))
  return(mase)
}


# Calculate MASE for each method
mase_arima <- calculate_mase(actual, forecasted)
mase_stl <- calculate_mase(actual, stlforecasted)
mase_seasonal_naive <- calculate_mase(actual, seasonal_naive)

# Print MAE and RMSE for each method
cat("MAE for ARIMA:", mae_arima, "\n")
cat("MAE for STL:", mae_stl, "\n")
cat("MAE for Seasonal Naive:", mae_seasonal_naive, "\n\n")

cat("RMSE for ARIMA:", rmse_arima, "\n")
cat("RMSE for STL:", rmse_stl, "\n")


cat("RMSE for Seasonal Naive:", rmse_seasonal_naive, "\n")



# Print MASE for each method
cat("MASE for ARIMA:", mase_arima, "\n")
cat("MASE for STL:", mase_stl, "\n")
cat("MASE for Seasonal Naive:", mase_seasonal_naive, "\n")
cop.decom2<- stl(tsdata, t.window=12, s.window="periodic", robust=TRUE)
plot(cop.decom2, main = "Figure 3.19:STL Decomposition of Rainfall")
summary(cop.decom2)

finalforecast<-forecast(cop.decom2,h=60)
finalforecast
plot(finalforecast,main="Figure 3.20:Forecast for the entire data ")
# Extract the forecasted values
forecast_values <- finalforecast$mean
# Generate forecasts for different horizons
horizons <- c(5, 10, 25,50, 100) * 12  # Convert years to months
forecasts <- lapply(horizons, function(h) forecast(cop.decom2, h = h)$mean)

# Perform k-means clustering on the original data
rainfall_matrix <- matrix(tsdata, ncol = 1)
original_clusters <- kmeans(rainfall_matrix, centers = 7)$cluster

# Function to perform clustering on combined data and compute ARI
compute_ari <- function(forecast_values, original_clusters, k = 7) {
  combined_data <- c(tsdata, forecast_values)
  combined_matrix <- matrix(combined_data, ncol = 1)
  combined_clusters <- kmeans(combined_matrix, centers = k)$cluster
  original_cluster_length <- length(original_clusters)
  ari <- adjustedRandIndex(original_clusters, combined_clusters[1:original_cluster_length])
  return(ari)
}

# Compute ARI for each forecast horizon
ari_values <- sapply(forecasts, compute_ari, original_clusters = original_clusters)

# Print the ARI values
names(ari_values) <- c("5 years", "10 years", "25 years", "50 years","100 years")
print(ari_values)
# Function to perform clustering on combined data and compute NMI
compute_nmi <- function(forecast_values, original_clusters, k = 7) {
  combined_data <- c(tsdata, forecast_values)
  combined_matrix <- matrix(combined_data, ncol = 1)
  combined_clusters <- kmeans(combined_matrix, centers = k)$cluster
  original_cluster_length <- length(original_clusters)
  nmi <- NMI(original_clusters, combined_clusters[1:original_cluster_length])
  return(nmi)
}

# Compute NMI for each forecast horizon
nmi_values <- sapply(forecasts, compute_nmi, original_clusters = original_clusters)

# Print the NMI values
names(nmi_values) <- c("5 years", "10 years", "25 years","50 years", "100 years")
print(nmi_values)

# Function to extract data points for a specific cluster
extract_cluster_data <- function(forecast_values, original_clusters, cluster_number) {
  combined_data <- c(tsdata, forecast_values)
  combined_matrix <- matrix(combined_data, ncol = 1)
  combined_clusters <- kmeans(combined_matrix, centers = length(unique(original_clusters)))$cluster
  
  # Extract data points for the specified cluster
  cluster_indices <- which(combined_clusters == cluster_number)
  cluster_data <- combined_matrix[cluster_indices, ]
  
  return(cluster_data)
}

# Define the forecast horizons
horizons <- c(5, 10, 25, 100) * 12  # Convert years to months

# Iterate through each cluster
for (cluster_number in 1:7) {
  # Create a new plotting window
  par(mfrow = c(1, length(horizons)+1), mar = c(4, 4, 2, 1))
  
  # Extract data points for the current cluster in the original data
  cluster_original <- extract_cluster_data(forecast_values = numeric(0), original_clusters = original_clusters, cluster_number = cluster_number)
  
  # Plot the histogram for the original data
  hist(cluster_original, main = paste("Cluster", cluster_number, "Original Data"), xlab = "Rainfall", ylab = "Frequency", col = "lightblue")
  
  # Iterate through each forecast horizon
  for (horizon in horizons) {
    # Extract data points for the current cluster and forecast horizon
    forecast_data <- forecasts[[which(horizons == horizon)]]
    cluster_combined <- extract_cluster_data(forecast_values = forecast_data, original_clusters = original_clusters, cluster_number = cluster_number)
    
    # Plot the histogram for the combined data
    hist(cluster_combined, main = paste("Cluster", cluster_number, "Original +", horizon/12, "Years Forecast"), xlab = "Rainfall", ylab = "Frequency", col = "lightblue")
  }
  
  # Save the plot as a separate file (optional)
   pdf(file = paste("Cluster_", cluster_number, "_histograms.pdf", sep = ""), width = 10, height = 6)
  par(mfrow = c(1, length(horizons)), mar = c(4, 4, 2, 1))

  dev.off()
}

# Extract data points for Cluster 1 for both original and combined data
cluster1_original <- extract_cluster_data(forecast_values = numeric(0), original_clusters = original_clusters, cluster_number = 1)
cluster1_combined <- extract_cluster_data(forecast_values = forecasts[[1]], original_clusters = original_clusters, cluster_number = 1)

# Plot histograms for Cluster 1 for original and combined data
par(mfrow = c(1, 2), mar = c(4, 4, 2, 1))
hist(cluster1_original, main = "Cluster 1 Original Data Distribution", xlab = "Rainfall", ylab = "Frequency", col = "lightblue")
hist(cluster1_combined, main = "Cluster 1 Combined Data Distribution", xlab = "Rainfall", ylab = "Frequency", col = "lightblue")

