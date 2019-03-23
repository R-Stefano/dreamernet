# DQN-Dreamer
This is a network system that tries to replicate a concept that I had from a long time in my mind. It tries to emulate the entire process that we use when we try to learn something using the state of art algorithm that I thought could be the fundamental building blocks.

We create mental models based on the experience that we live. Then, we use these simulators to decide what to do.

This system tries to re-implement variations of world models and alphaZero in order to create a full and efficient planning agent.

## Get started
Clone the repo running
```
git clone https://github.com/R-Stefano/DQN-Dreamer.git
```

Install and setup a [**virtualenv**](https://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv/).

Once activated, navigate to the main folder of the project and install dependencies:
```
pip install -r requirements.txt
```

Finally, run
```
python main.py 
```

Other possible commands:
```
//train VAEGAN and RNN before train agent
python main.py --preprocessing=True

//train only VAEGAN before train agent
python main.py --preprocessing=True training_RNN=False

//train only RNN before train agent
python main.py --preprocessing=True training_VAEGAN=False

//observe the agent playing while training
python main.py --renderGame=True
```