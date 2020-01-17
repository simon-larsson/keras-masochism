# Keras Masochism ༼ノಠل͟ಠ༽ノ ︵ ┻━┻
Masochistic deep learning with Keras.

### Russian Roulette
Do you want to add an element of danger to your training? Are you willing gamble when your models life is on the line? Then this russian roulette callback is just what you are looking for!

Play a game of russian roulette in the end of your training. If you lose your network dies. If you win you will cherish every prediction you make.

#### Arguments

- **rounds** - number of bullets that will be loaded, int.
- **chambers** - number of bullet chambers, int.
- **firings** - number of times the trigger will be pulled, int.

```python
from masochism.callbacks import RussianRoulette

rr = RussianRoulette(1, 6, 1)

model.fit(X, y, epochs=5, verbose=1, callbacks=[rr])
```

Output without spoiling the end
```python
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
