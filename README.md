# Trading
Using the Alpaca Markets API to to pull historical data, aggrigate said data in to
volumetric bars/candlesticks (s.t. the periods of each bar/candlestick is equal to a
volume of the users liking). Strip time from the equation, we are going to be testing
true price action.



## ToDo
* Determine appropriate paramaters for the HMM (e.g. first layer being (close-open)/open, second and third layers being (High-Open)/Open andn (Open-Low)/Low (Can help to identify certain candlestick patterns), and maybe even a fourth layer.])

## Updates
* Hello! This new project will try to analyze true volumetric price action data using a hidden markov model.
* THE HMM WILL NOT BE PUSHED. I USE ALOT OF CODE I WROTE IN A CLASS I TOOK IN THE PAST. PUBLISHING IS AGAINST HONOUR CODE.


# Old/Depreciated (TTM.py)

Using the Alpaca Markets API alongside the Polygon API this program will watch a stock or stocks in real time and actively utilize the TTM Squeeze Indicator to alert the user through email about significant market activity.

## ToDo
* Transition over to Day based data
* Implementing live trading function.?
* Multiple time frames being tracked at once.?

## Updates
* I might or might not continue to work on this project. Currently, indicators are useful to get a visual but when applied against the market results are noisy.
Perhaps moving over to day based data will work or greatly help, but that would strip the need for the program to be live.

