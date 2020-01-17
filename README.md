# Keras Masochism ༼ノಠل͟ಠ༽ノ ︵ ┻━┻
An extensions for Keras that make working with deep learning more painful then it needs to be. Made for those that

### Russian Roulette :gun: :game_die: 
Do you want to add an element of danger to your training? Are you willing gamble when your models life is on the line? Then this russian roulette callback is just what you are looking for!

Play a game of russian roulette in the end of your training. If you lose your network dies and your weights are destroyed. But if you win you will remember cherish every prediction it makes!

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

**Using models checkpoints when playing russian roulette is frowned upon** :unamused:
