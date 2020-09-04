# Trading Strategy
Train various Gaussian-Hidden Markov Models to make predictions during common market trends (bearish, bullish, channels, etc) using large, historic, minute-based datasets of various companies compiled into standard, volumetric periods. After stripping time from the equation we both minimize the noise fed into the machine learning model and better see and respond to True Price Action.

## ToDo
* Implement paper trading functionality to test near 'alpha' truth of the strategy. This will not include bear periods as shorting is not an option at the current moment (too risky).
* Add gpu parallelization (with CuPy) to speedup as much as possible. Also, clean up logic.

## Updates
* Hello! This new project will try to analyze true volumetric price action data using a hidden markov model.
* After investigating the left-most portion of the bimodial graph (smaller volumetric periods), I achieved a consistent 65% success rate on both the bull and bear periods. (*check update section*)
* Currently the models seem to generate bimodial graphs during optimization with the peaks appearing on extremely small volumetric periods (~2-4 minutes on aversge - where True price actions lies) and on the larger time frames (~1.5 to 2 days - where institutional trading exists). 
* The smaller volumetric periods (~2 - 4 min average) see success rates (in regards to determining the direction of the next period) hovering around 60% - 65%. (*check update section*)
* The larger volumetric periods (~1.5 - 2 days average) see success rates hovering around 58%. (*check update section*)

# Old/Depreciated (TTM.py/backtest.py)

As the current project develops, I will begin to interweave the live trading/analysis functionality of the TTM.py project to determine which model to use/what the current state of the market is (this project was, after all, purely indicators!).

## ToDo
* Use EMA/other indicators to determine trends. Parse Historic data to generate sub-datasets that the future indicators would consider "bullish," "bearish," "channeling," etc.

## Updates
* It should be noted that these statistics are plagued by model overfitting. More or less, this is just a proof of concept.

* I might or might not continue to work on this project. Currently, indicators are useful to get a visual/wholistic understanding of the market, but when applied against the market results are noisy.
Perhaps moving over to day based data will work or greatly help, but that would strip the need for the program to be live.