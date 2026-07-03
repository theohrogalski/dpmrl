# TODO
**Bold for importance**
_Italics for external to the file itself_
## General
A general list of tasks meant to be completed in future, in order of urgency.

1. Cleaning code of poor structure
2. Adding desriptive comments to the code 
3. Rationalize the way data is collected
4. Add centralized training
5. Add more metrics for collection
6. Add dynamic visualization for agents (i.e. a video instead of a series of freeze-frames)
7. Fix graph creation structure
8. Use HPC for training and testing


## Specific Files
### training.py
* Convert training interface to parser-based rather than direct code changing (perhaps a UI in future?)
* Update to be universally compatible with centralized/decentralized agent algorithms
* Update deprecated save_marl_checkpoint and broken save_diagnostic_plots functions
* Remove deprecated text, add comments for clarity
* Add assert functions to confirm functionality
* Add hyperparemeter editing to collect further data
* _Peripheral_: Add generator function for automatic parsing
* Centralization function should batch observations 
### eval.py
* Update logging format to be more descriptive
* Update the format of the logger
* Consider splitting evaluation into "classical algorithms" and "RL Algorithms"
* _**Add centralized algorithm evaluation**_
* Rationalize model selection
* Add ability to evaluate algorithms from other papers 
### requirements.txt and alternatives
* Correct requirements.txt
* Use device params that incorporates both Nvidia and CPu for training/evaluation pipeline

### README.md
* Add detailed usage descriptions 
* Add visualizers
* Add example usage
* Link paper