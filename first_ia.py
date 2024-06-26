# https://www.youtube.com/watch?v=bzC5cdxZcOM&t=966s
import numpy as np 

# longueur, largeur
# x_enter = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]), dtype=float) # Normalement devrait être Rouge
# x_enter = np.array(([3,1.5],[4,1.5],[3.5,0.5],[5.5,1],[2,1],[3,1],[2,0.5],[1,1],[4.5,1]), dtype=float) # Normalement devrait être Rouge
x_enter = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[1.5,1]), dtype=float) # Normalement devrait être Bleu
# donnée de sortie 1 = rouge / 0 = bleu 
# y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float) 
# y = np.array(([1],[1],[1],[1],[0],[0],[0],[0]), dtype=float) 
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float) 

# Doit mettre a la même valeur entre 0 et 1
# Donc on divise chaque valeur par la valeur max du array
x_enter = x_enter / np.amax(x_enter, axis=0)

# Valeur ou on a toutes les informations
X = np.split(x_enter, [8])[0]
# Valeur qui nous manque
xPrediction = np.split(x_enter, [8])[1]

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # Matrice 2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # Matrice 3x1

    def forward(self,x):
        self.z = np.dot(x, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o
 
    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    # Dérivée de notre sigmoid
    def sigmoidPrime(self, s):
        return s * (1-s)
    
    # Fonction de retropropagation
    def backForward(self, X, y, o):
        # Calcul de l'erreur output error
        self.o_error = y - o # valeur d'entrée moins la valeur de sortie
        # Erreur delta
        self.o_delta = self.o_error * self.sigmoidPrime(o) # erreur mutliplié par la dérivée de notre sortie

        # Calcul des neuronnes caché
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        # Mettre à jour les poids
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
        
    # Fonction qui va entrainer la machine a mettre à jour son poid    
    def train(self, X, y):
        o = self.forward(X)
        self.backForward(X, y, o) 

    def predict(self):
        print("Donnée prédite après entrainement: ")
        print("Entrée : \n" + str(xPrediction))
        print("Sortie : \n" + str(self.forward(xPrediction)))

        if (self.forward(xPrediction) < 0.5):
            print("La fleur est Bleue ! \n")
        else:
            print("La fleur est Rouge ! \n")


NN = Neural_Network()

for i in range(30000):
    print("# " + str(i) + "\n")  
    print("Valeurs d'entrées: \n" + str(X))
    print("Sortie actuelle: \n" + str(y))
    print("Sortie prédite: \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")  
    NN.train(X, y)

NN.predict()