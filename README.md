# :imp: Keras Masochism ༼ノಠل͟ಠ༽ノ ︵ ┻━┻
Keras for twisted minds. An extension of Keras that makes working with deep learning more painful then it needs to be. Perfect for those that enjoy self sabotage.

## Installation

```python
git clone https://github.com/simon-larsson/keras-masochism.git
cd keras-masochism
pip install -e
```

## Components

- **Russian Roulette** - russian roulette callback that might kill your model.
- **Crackmoid** - the opposite of a good activation function.
- **SGDIM** - a gradual descent into madness sub-optimizer that missteps.
- **GroupAbnormalization** - (Not impl.) denormalize activations of previous layers independent of batch size.
- **DepressNet** - (Not impl.) ResNet with depressingly long residual connections, Deep Pre-Activation Residual Networks.
- **DeficientNet** - (Not impl.) a malnourished version of EfficientNet.

### Russian Roulette :gun: :game_die: 
Do you want to add an element of danger to your training? Are you willing gamble when your models life is on the line? Then this russian roulette callback is just what you are looking for!

Play a game of russian roulette in the end of your training. If you lose your network dies and your weights are destroyed. But if you win you will remember to cherish every prediction it makes!

#### Arguments

- **rounds** - number of bullets that will be loaded, int.
- **chambers** - number of bullet chambers, int.
- **firings** - number of times the trigger will be pulled, int.

#### Example
```python
from masochism.callbacks import RussianRoulette

rr = RussianRoulette()

model.fit(X, y, epochs=5, verbose=1, callbacks=[rr])
```

Output without spoiling the end
```
Epoch 5/5
1000/1000 [==============================] - 0s 31us/step - loss: 0.6929
 ______________________________
|                              |
| LET´S PLAY RUSSIAN ROULETTE! |
|______________________________|

Inserting round

Spinning cylinder

.
.
.
.
.

Squeezing trigger
```

**Using model checkpoints when playing russian roulette is frowned upon** :unamused:

### Crackmoid
Being discontineous and undefined in zero is perhaps not what normal people look for in an activation function. But maybe you are not normal. Maybe you like seeing your models explode. :boom:

**f(x) = sigmoid(x)/x**

[![crackmoid](https://raw.githubusercontent.com/simon-larsson/keras-masochism/master/crackmoid.png)](https://raw.githubusercontent.com/simon-larsson/keras-masochism/master/crackmoid.png)



Crackmoid is losely based on the swish activation function from the paper [Searching for Activation Functions](https://arxiv.org/abs/1710.05941).

#### Examples
```python
import masochism

model.add(Dense(10, activation='crackmoid'))
```

```python
import masochism

model.add(Activation('crackmoid'))
```

```python
from masochism.layers import Crackmoid

model.add(Crackmoid())
```

### SGDIM - Stochastic Gradual Descent Into Madness
SGDIM is a sub-optimizer based on gradient descent. It works in much the same way except it will sometimes take random missteps in the wrong direction based on a madness factor.  

#### Arguments

- **learning_rate** - learning rate, float.
- **momentum** - accelerates SGD in the relevant direction and dampens oscillations, float.
- **nestrov** - flag for applying nestrov momentum, bool.
- **madness** - Likelihood of taking a misstep, float.


#### Examples
```python
from masochism.optimizers import SGDIM

model.compile(optimizer=SGDIM(madness=0.2),
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

```python
import masochism

model.compile(optimizer='sgdim',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```
