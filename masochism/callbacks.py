from time import sleep
import numpy as np
from keras.callbacks import Callback

class RussianRoulette(Callback):
    """Play a game of russian roulette.
    
    # Arguments
        rounds: int, number of bullets that will be loaded.
        chambers: int, number of bullet chambers.
        firings: int, number of times the trigger will be pulled.
    """
    
    def __init__(self, rounds=1, chambers=6, firings=1, **kwargs):
        
        super(RussianRoulette, self).__init__(**kwargs)
        
        if chambers < 5 or chambers > 12:
            raise ValueError('A revolver has 5-12 chambers.')
            
        if rounds < 1:
            raise ValueError('No cheating... you have to put at least'
                             ' one round in the revolver.')
        
        if firings < 1:
            raise ValueError('No cheating... you have to fire at least'
                             ' once.')       

        if chambers - rounds < firings:
            raise ValueError('Someone has a deathwish... give yourself'
                             ' a chance to live.')
        
        self.rounds = rounds
        self.chambers = [False]*chambers
        self.firings = firings
    
    def on_train_begin(self, logs=None):
        
        # storing starting weights to destroy the network with
        self.starting_weights = self.model.get_weights()
    
    def on_train_end(self, logs=None):
        
        print(' ______________________________')
        print('|                              |')
        print('| LET´S PLAY RUSSIAN ROULETTE! |')
        print('|______________________________|')
        
        # sabotage seed cheaters!
        seed_state = np.random.get_state()
        np.random.seed(None)
        
        # inserting rounds in a row
        chamber = np.random.randint(0, len(self.chambers) - 1)
        for r in range(1, self.rounds+1):
            print('\nInserting round')
            self.chambers[chamber % len(self.chambers)] = True
            sleep(1)
            chamber += 1
            
        # spin until it lands on chamber
        print('\nSpinning cylinder\n')
        for _ in range(5):
            sleep(1)
            print('.')
        chamber = np.random.randint(0, len(self.chambers) - 1)
        
        # restore the seed
        np.random.set_state(seed_state)
        
        # fire the revolver and see if chamber is loaded
        for _ in range(self.firings):
            sleep(1)
            print('\nSqueezing trigger')
            sleep(2)
            
            if self.chambers[chamber % len(self.chambers)]:
                
                # destroy weights
                self.model.set_weights(self.starting_weights)
                raise RuntimeError('You died... Thank you for playing!')
            else:
                print('\nCLICK!')
                chamber += 1
                
        print('\nYou survived! Make it matter.')
