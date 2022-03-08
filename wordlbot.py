from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling2D,Conv2DTranspose,UpSampling2D, Reshape, Input
import numpy as np

#Finding closest word:
from scipy.spatial.distance import cdist
ALPHABET = 27

class Words:
    def __init__(self):

        """
        get_words.py
        Utility method to load the SBG words
        and retun them as a list of strings.
        """
        with open('sgb-words.txt','r') as f:
            ## This includes \n at the end of each line:
            #words = f.readlines()

            # This drops the \n at the end of each line:
            dictionary = f.read().splitlines()


        self.dic = dictionary
        self.data = []
        for word in dictionary:
            n = []
            for x in word:
                n.append(ord(x.lower()) - 96)
            self.data.append(np.array(n))
        self.un_norm_data = np.array(self.data)
        self.data = self.un_norm_data/ALPHABET # Normalize
        self.reduced_data_norm = self.un_norm_data/ALPHABET
        self.reduced_data = self.un_norm_data

    def get_sub_dic(self,letters = False, correct_spot = False, enemies = False):
        if letters:
            for l in letters:
                if l != '-':
                    self.reduced_data = self.reduced_data[ np.any(self.reduced_data == int(too_numb(l)), axis=1)]
                    #self.reduced_data_norm = self.reduced_data/ALPHABET
            for p in range(len(letters)):
                if letters[p]!= '-':
                    mask = (self.reduced_data[:, p] != int(too_numb(letters[p])))
                    self.reduced_data = self.reduced_data[mask, :]
        if correct_spot:
            rows = []
            for k in range(len(correct_spot)):
                if correct_spot[k]!= '-':
                    mask = (self.reduced_data[:, k] == int(too_numb(correct_spot[k])))
                    self.reduced_data = self.reduced_data[mask, :]
        if enemies:
            for e in enemies:
                self.reduced_data = self.reduced_data[ np.all(self.reduced_data != int(too_numb(e)), axis=1)]
        self.reduced_data_norm = self.reduced_data/ALPHABET
        print(f"We have totally {self.reduced_data.shape} words to guess from")

class Network:
    def __init__(
        self,
        force_learn: bool = True,
        file_name: str = "./bot/007",
    ):
        self.file_name = file_name
        self.force_learn = force_learn
        self.input_size = 5
        self.force_relearn = force_learn
        # This is the size of our encoded representations
        encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

        # This is our input image
        input_img = keras.Input(shape=(5,))

        encoded = Dense(200, activation='relu')(input_img)

        encoded = Dense(100, activation='relu')(encoded)
        encoded = Dense(100, activation='relu')(encoded)

        decoded = Dense(5, activation='relu')(encoded)

        # This model maps an input to its reconstruction
        autoencoder = keras.Model(input_img, decoded)

        autoencoder.compile(optimizer='Adam',
         loss='binary_crossentropy',
         metrics=["accuracy"])

        self.model = autoencoder
        self.done_training = self.load_weights()

    def load_weights(self):
        # noinspection PyBroadException
        try:
            self.model.load_weights(filepath=self.file_name)
            print(f"Read model from file, so I do not retrain")
            done_training = True
        except:
            print(
                f"Could not read weights for verification_net from file. Must retrain..."
            )
            done_training = False
        return done_training

    def train(self,X,epochs = 15):
        self.done_training = self.load_weights()
        if self.force_relearn or self.done_training is False:

            X =  X.astype('float32')
            #print(X.shape)
            self.model.fit(
                    X,
                    X,
                    batch_size=50,
                    shuffle=True,
                    epochs=epochs,
                    validation_data=(X, X),
                )
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
        return self.done_training

def search(matrix, search_vec):
    return (matrix[cdist(matrix, np.atleast_2d(search_vec)).argmin()])
def too_alpha(vec):
    word = ''
    for i in range(vec.shape[0]):
        #print(vec[i])
        word += chr(int(vec[i])+96)
    return word

def too_numb(string):
    n = []
    for x in string:
        n.append(ord(x.lower()) - 96)
    return np.array(n)

if __name__ == "__main__":
    print("What game are you plaing?")
    ans = -1

    file_name = 'bot/Wordlbot1'

    data = Words()
    dic = data.data
    net = Network(force_learn = False, file_name =file_name )
    print(net.model.summary())
    net.train(data.data, epochs = 5)

    print("write break too exit.")
    guess = 'a'
    noise = np.ones((1,5))
    first = 1
    print("AUTOMODE Y/N")
    auto = input().lower()
    while True:
        know1 = False
        know2 = False

        if auto == 'n' or first:
            print("What word do you guess?")
            while len(guess) != 5:
                guess = input()
        else:
            guess = last_pred
        if guess == "break":
            pass
        print("Write output as G - green, Y - yellow, B - black, eg BBGBY")
        out = input()
        known_at_all = ''
        enemies = ''
        known_at_right = ''
        for i in range(len(out)):
            if out[i].lower() == 'y':
                known_at_all += guess[i]
            else:
                known_at_all += '-'

            if out[i].lower() == 'g':
                known_at_right += guess[i]
            else:
                known_at_right += '-'
            if out[i].lower() == 'b':
                enemies += guess[i]
        data.get_sub_dic(known_at_all, known_at_right,enemies)
        dic = data.reduced_data_norm

        if first and (known_at_right or known_at_all):
            B = int(dic.shape[0]/10)
            if B == 0:
                B = 1
            net.model.fit(
                    dic,
                    dic,
                    batch_size=B,
                    shuffle=True,
                    epochs=10,
                )
        guess = guess.lower()
        guess = too_numb(guess)/ALPHABET
        guess = guess.reshape(1,5)# * noise + 1/(noise)*first
        pred = net.model.predict(guess)[0]

        print(f"preguss is {too_alpha(pred*ALPHABET)}")
        pred = search(dic,pred) # Dic
        pred = too_alpha(pred*ALPHABET)

        print(f"Next guess should be {pred}")
        noise = np.random.rand(1,5)
        if dic.shape[0] < 10:
            #print("Reminding words we can guess are:")
            for word in dic:
                pass
                # Write out dictonary, ... to boring
                #print(too_alpha(word*ALPHABET), end = '--')
            #print(" ")
        last_pred = pred
        first = 0
