library(tidyverse)
library(dplyr)
library (bizdays)
library(readxl)

setwd("/Users/timothee/Documents/IESEG/4a/Master Thesis/Data_Management")
SPX_prices_1996to2021 <- read.csv("S&P_prices_1996-2021.csv")
historical_volatility <- read.csv("SPXOptions_Historical_Vol.csv")
#yield_curve <- read_excel("Libor curve.xlsx")
yield_curve <- read.csv("zero_coupon_yield_curve.csv")
Overnight_Libor <- read.csv("LIBOR USD.csv")
#option_file <- read.csv("SPXOptions.csv")
option_file <- read.csv("SPXOptions_subset_3.csv")
#Option_file prep
option_file <- option_file %>% mutate(date=as.Date(date))
option_file <- option_file %>% mutate(exdate=as.Date(exdate))
#option_file$Time_to_expiry <- option_file$exdate - option_file$date 
option_file$Time_to_expiry <- bizdays(option_file$date, option_file$exdate, 
                                      cal = create.calendar('my_calendar', weekdays = c('saturday','sunday')))

option_file$strike_price <- option_file$strike_price/1000
option_file$mid_price <- (option_file$best_bid + option_file$best_offer)/2

option_file <- subset(option_file, volume != 0)
option_file <- option_file[option_file$Time_to_expiry >= 1, ]

SPX_prices_1996to2021 <- SPX_prices_1996to2021 %>% mutate(date=as.Date(date))
option_file$S0 <- SPX_prices_1996to2021$close[match(option_file$date, SPX_prices_1996to2021$date)]
option_file$S1 <- SPX_prices_1996to2021$close[match(option_file$exdate, SPX_prices_1996to2021$date)]
option_file$moneyness <- option_file$S0 / option_file$strike_price



#Yield Curve preparation
#yield_curve<-yield_curve %>% rename(date = Name)
yield_curve <- yield_curve %>% mutate(date=as.Date(date))
Overnight_Libor <- Overnight_Libor %>% mutate(date = as.Date(Date, format = "%d.%m.%Y"))

Overnight_Libor$Date <- Overnight_Libor$date
Overnight_Libor <- Overnight_Libor[,-10]
Overnight_Libor <- Overnight_Libor %>%
  rename(date = Date)

#Rename some columns in option file
option_file <- option_file %>% rename("K" = "strike_price", "C0" = "mid_price", "M" = "moneyness")


# Export the dataframe to a CSV file
write.csv(option_file, file = "option_file.csv", row.names = FALSE)
write.csv(Overnight_Libor, file = "Overnight_Libor.csv", row.names = FALSE)
write.csv(yield_curve, file = "yield_curve.csv", row.names = FALSE)