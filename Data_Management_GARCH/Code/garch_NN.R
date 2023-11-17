library (dplyr)
library(tidyr)
library(rmgarch)
library(anytime)
library(timeSeries)
library (fBasics)
library (fExtremes)
library(FRAPO)
library(QRM)
library(copula)
library(moments)
library(tseries)
library(forecast)
library(mvnormalTest)
library(MVN)
library(spd)
library(lmtest)
library(copula)
library(QRM)
library(fitdistrplus)
library(logspline)
library(ggplot2)
library(cowplot)
library(xts)
setwd("/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management")
SPX_prices <- read.csv("S&P_prices_1996-2021.csv")
SPX_prices$date <- as.Date(SPX_prices$date, format = "%Y-%m-%d")
SPX_prices[,"log_return"]<- returnseries(SPX_prices[,"close"])
SPX_prices <- SPX_prices[-1,]



dens <- plot(density(SPX_prices[, "log_return"]), main = "")
mu <- mean(SPX_prices[, "log_return"])
sigma <- sd(SPX_prices[, "log_return"])
x <- seq(-max(SPX_prices[, "log_return"]), -min(SPX_prices[, "log_return"]), length = 100)
lines(-x, dnorm(x, mean = mu, sd = sigma), col = "red")
title("Empirical Density vs. Theoretical Normal Distribution")
legend("topleft", legend = c("Empirical Data", "Theoretical Normal"), col = c("black", "red"), lty = 1)

CullenFrey <- descdist(SPX_prices[,"log_return"])
adf = adf.test(SPX_prices$log_return)
acf(SPX_prices$log_return)
pacf(SPX_prices$log_return)


SPX_prices <- xts(subset(SPX_prices, select = -date), order.by = as.Date(SPX_prices$date))
model.arima = auto.arima(SPX_prices$log_return, max.order = c(3, 0 ,3) , stationary = TRUE , trace = T , ic = 'aicc')


spec = ugarchspec(variance.model = list(model = "fGARCH", garchOrder = c(2, 2), 
                                        submodel = "GARCH", external.regressors = NULL, variance.targeting = FALSE), 
                  mean.model = list(armaOrder = c(0, 1), include.mean = TRUE, archm = FALSE, 
                                    archpow = 1, arfima = FALSE, external.regressors = NULL, archex = FALSE), 
                  distribution.model = "std", start.pars = list(), fixed.pars = list())

###Roll model
in_sample_size = 250
out_of_sample_size = 50

roll <- ugarchroll(spec = spec, data = SPX_prices$log_return, n.ahead = 1, n.start = in_sample_size, refit.every = out_of_sample_size, refit.window = "moving")
roll_preds <- as.data.frame(roll)
colnames(roll_preds)[2] <- "garch_vol"
garch_vol <- xts(roll_preds$garch_vol, order.by = as.Date(rownames(roll_preds)))
plot(garch_vol)

#Garch vol vs Realized vol plot
garch_volatility <- coredata(garch_vol)
realized_volatility <- roll_preds$Realized


plot_data_roll <- data.frame(
  Date = index(garch_vol),
  GARCH_Volatility = garch_volatility,
  Realized_Volatility = realized_volatility
)

ggplot(data = plot_data_roll, aes(x = Date)) +
  geom_line(aes(y = Realized_Volatility, color = "Realized Volatility"), size = 1) +
  geom_line(aes(y = GARCH_Volatility, color = "GARCH Volatility"), size = 1) +
  labs(
    title = "Realized Volatility vs. GARCH Forecasted Volatility",
    x = "Date",
    y = "Volatility%",
    color = "Legend"
  ) +
  scale_color_manual(values = c("Realized Volatility" = "blue", "GARCH Volatility" = "red")) +
  theme(legend.position = c(0.15, 0.10))

roll_preds$garch_vol <- (roll_preds$garch_vol/100) * sqrt(252)
roll_preds <- subset(roll_preds, select = garch_vol)

roll_preds <- cbind(date = rownames(roll_preds), roll_preds)
write.csv(roll_preds, file = "garch_vol.csv", row.names = FALSE)





