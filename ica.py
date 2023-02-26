import numpy as np
from scipy.io import wavfile as wf
import matplotlib.pyplot as plt

np.random.seed(0)
s1_file = "./talk.wav"
s2_file = "./music.wav"


def mix_sources(sources):
    for i in range(len(sources)):
        max_val = np.max(sources[i])
        if(max_val > 1 or np.min(sources[i]) < 1):
            sources[i] = sources[i] / (max_val / 2) - 0.5
            
    mixture = np.c_[[source for source in sources]]
    return mixture

# Leemos los archivos
_, s1 = wf.read(s1_file)
sampling_rate, s2 = wf.read(s2_file)

# Imprimimos las muestras
print(s1.shape, s2.shape)

# Mezclamos ambos audios y nos regresa una matriz que podemos procesar
x = mix_sources([s1, s2[:s1.shape[0]]])
# guardamos la mezcla procesada
wf.write('./talk_and_music.wav', sampling_rate, x.mean(axis=0).astype(np.float32))


# Imprimimos la gráfica de la muestra mezclada
fig = plt.figure()
for shape in x:
        plt.plot(shape)
plt.title("Mezcla")
fig.tight_layout()
plt.show()



def center(x):
    x = np.array(x)
    return x - x.mean(axis=1, keepdims=True)

def whiten(x):
    eigen_values, eigen_vectors = np.linalg.eigh(np.cov(x))
    D = np.diag(eigen_values)
    sqrt_inverse_D = np.sqrt(np.linalg.inv(D))
    x_whiten = eigen_vectors @ (sqrt_inverse_D @ (eigen_vectors.T @ x))
    
    return x_whiten, D, eigen_vectors

X_whiten, D, E = whiten(
    center(x)
)
D, E

def objFunc(x):
    return np.tanh(x)

def dObjFunc(x):
    return 1 - (objFunc(x) ** 2)

def calc_w_hat(W, X):
    # Implementation of the eqn. Towards Convergence
    w_hat = (X * objFunc(W.T @ X)).mean(axis=1) - dObjFunc(W.T @ X).mean() * W
    w_hat /= np.sqrt((w_hat ** 2).sum())
    
    return w_hat

def ica(X, iterations, tolerance=1e-5):
    num_components = X.shape[0]
    
    W = np.zeros((num_components, num_components), dtype=X.dtype)
    distances = {i: [] for i in range(num_components)}
    
    for i in np.arange(num_components):
        w = np.random.rand(num_components)
        for j in np.arange(iterations):
            w_new = calc_w_hat(w, X)
            if(i >= 1):
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            
            w = w_new
            if(distance < tolerance):
                print(f'Convergence attained for the {i+1}/{num_components} component.')
                print(f'Component: {i+1}/{num_components}, Step: {j}/{iterations}, Distance: {distance}\n')
            
                break;
                
            distances[i].append(distance)
            
            if(j % 50 == 0):
                print(f'Component: {i+1}/{num_components}, Step: {j}/{iterations}, Distance: {distance}')
            
         
                
        W[i, :] = w
    S = np.dot(W, X)
    plt.plot(S[0],'--', color='orange')
    plt.title("Música")
    fig.tight_layout()
    plt.show()   

    plt.plot(S[1])
    plt.title("Hablando")
    fig.tight_layout()
    plt.show()   
    
    return S, distances

S, distances = ica(X_whiten, iterations=100)
wf.write('s1_predicted.wav', sampling_rate, S[0].astype(np.float32))
wf.write('s2_predicted.wav', sampling_rate, S[1].astype(np.float32))



