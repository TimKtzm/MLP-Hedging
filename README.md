# MLP-Hedging

Application of a Multi-Layer Perceptron-Based Model to Option Pricing & Hedging.
Here is the code I wrote to design the MLP model used to hedge European options as part of my Master's thesis at the IÉSEG School of Management.

[Link to my thesis]()

## Author

- [@TimKtzm](https://github.com/TimKtzm)


## FAQ

#### How do I run the code?

The code consists of 3 folders (1 for IV, 1 for GARCH forcasted volatility, 1 for delta-optimized MLP) each containing 14 files:
  -	Dataframe_preparation.R: Filters the original database and makes changes to the auxiliary files used to prepare the final dataframe
  -	garch_NN.R: implements a GARCH rolling model for volatility prediction (Data_Management_GARCH folder only)
  -	Dataframe_preparation.py: Prepares data for MLP
  -	training.py: MLP implementation
  -	Hedging.py: hedges the positions, computes and stores the various metrics
  -	pricing_results: creates a table of pricing results
  -	Hedging_results: creates a table of hedging results
  -	Other files with the "func" suffix contain functions only
  -	The remaining files provide additional information on results and model architectu

#### What files do I need?

You will need four files to run the code:
  -	S&P_prices_1996-2021.csv: contains the price history for the S&P 500 index
  -	SPXoptions.csv: contains historical prices for S&P 500 options, IV, Greeks
  -	zero_coupon_yield_curve.csv: historical interest rates used to interpolate risk-free rates
  -	Libor USD.csv: Option Metrics does not provide short-term interest rates, which are necessary to interpolate rates


## Feedback

If you have any feedback, please contact me at kuntzmanntimothee@gmail.com

    
          
