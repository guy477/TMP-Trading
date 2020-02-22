# Trading Strategy
Train various Gaussian-Hidden Markov Models to make predictions during common market trends (bearish, bullish, channels, etc) using large, historic, minute-based datasets of various companies compiled into standard, volumetric periods. After stripping time from the equation we both minimize the noise fed into the machine learning model and better see and respond to True Price Action.

## ToDo
* Determine which volumetric periods the models respond to best. Check *Updates* for details
* Add gpu parallelization (with CuPy) to speedup as much as possible. Also, clean up logic.

## Updates
* Hello! This new project will try to analyze true volumetric price action data using a hidden markov model.
* Currently the models seem to generate bimodial graphs during optimization with the peaks appearing on extremely small volumetric periods (~2-4 minutes on aversge - where True price actions lies) and on the larger time frames (~1.5 to 2 days - where institutional trading exists). 
* The smaller volumetric periods (~2 - 4 min average) see success rates (in regards to determining the direction of the next period) hovering around 60% - 65%.
* The larger volumetric periods (~1.5 - 2 days average) see success rates hovering around 58%.
* Wrote an optimization function (using that word loosly) 
* Data used for proof of concept model: https://github.com/Zombie-3000/Bitfinex-historical-data

# Old/Depreciated (TTM.py/backtest.py)

As the current project develops, I will begin to interweave the live trading/analysis functionality of the TTM.py project to determine which model to use/what the current state of the market is (this project was, after all, purely indicators!).

## ToDo
* Use EMA/other indicators to determine trends. Parse Historic data to generate sub-datasets that the future indicators would consider "bullish," "bearish," "channeling," etc.

## Updates
* I might or might not continue to work on this project. Currently, indicators are useful to get a visual/wholistic understanding of the market, but when applied against the market results are noisy.
Perhaps moving over to day based data will work or greatly help, but that would strip the need for the program to be live.