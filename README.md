# Agent-Based model of Twitter


This project simulates a twitter network using Agent-Based Modeling. It aims to understand the factors that cause polarization and echo chamber formation within online communities. With this model, you will be able to watch animations of how information moves within a network and the effects it has on the user agents.

## Model Breakdown:
### User Agents:
**Attributes:**
* Opinion - value that ranges between -1 - 1
* Opinion flexibility - how likely a user will hange their opinions.
* timelines - a 'screen' where they can obtain information from their friends.
* endorsement threshold - how likely a user will interact with information that doesn't line up with ther own opinion
* Influence - The power a user holds in convincing others with their informating spreading

**Actions**
* Create and publish tweets
* Like and Retweet
* Read their timeline
* Update their opinion
* Drop worst neighbour - distance from those they disagree with


## Running the model:
**There are 2 ways to interact with the model:**

(1) Run the main.py file from the commandline. This way you can view the results of all expriments within a set at one go.

(2) Run model through GUI.py. Here you have a user interface that lets you input information relating to the described experimenrts as well as customizing own parameters for model runs

When an experiment is run, you will be navigated to a localilly hosted animation of the timesteps the model goes through. Additionally, Graphs will be produced to further interpret results.

### Dependencies:
Python\
Flask\
Plotly\
NetworkX
