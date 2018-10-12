# Toxical
Andrew Li (Pd. 1) and Justin Li (Pd.5)

## What It Does
Our project is a bot that automatically scrapes and analyzes messages sent on the Discord messaging platform. The bot 
will use a machine learning algorithm to detect the “sentiment”, or overall positivity or negativity, of messages sent 
through the platform. We will train the bot’s algorithm using datasets available online. The machine learning algorithm 
will use a process called word embedding which reduces every word into a multi-dimensional vector, representing the 
overall “profile” of the word (similar words will have similar directions and magnitudes in its vector representation). 

## Why
The purpose of this bot is to provide a platform for positive environment control. This framework could be used to 
direct online discussion on sites like Facebook or Twitter in a positive direction or be used by businesses to promote 
positive interaction between automated customer service bots and customers.

## API's Used
Tensorflow and Tensorboard (1.8.0)

NumPy

Discord.py

## Installation

### Windows:
If you do not have Python 3.6.x 64 bit installed, install it from www.python.org/downloads/release/python-365/

When installing, make sure to choose "Add Python 3.6 to PATH" on the first screen. Also choose to disable PATH length 
limit at the end of installation.

Download this repository from GitHub as a ZIP file and extract all. In the extracted folder, navigate to the 
`/CS-Final-2018-master/Data/` folder and extract the `ids_matrix.npy.zip` file to `/CS-Final-2018-master/Data/`. 
**Make sure you do NOT not extract to a new folder named `ids_matrix.npy`** If you do, move the `ids_matrix.npy` file 
inside the folder of the same name outside to `/CS-Final-2018-master/Data/`.

Run command prompt in administrator mode and run the following line:

```pip install tensorflow```

Navigate in command prompt to `/CS-Final-2018-master/Bot` and run the following line:

```python analyze_sentiment.py```

The script may take a few moments to load. Ignore any non-fatal warnings. The console will print the results of the test
of the neural network. After the tests finish, you may enter any input you want to test the network.

#### Interpreting output
The output will list a positive score, which represents how positive the message is, and a negative score, which 
represents how negative the message is. The composite score is the difference of the positive score and the negative
score. A positive composite score represents a positive message and vice-versa.

#### Errors
An error may ask you install Microsoft Visual C++ Redistributable 2015. Follow any error messages and run the script
again.