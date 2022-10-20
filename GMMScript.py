import DataLoader.DataLoaderAll as DataLoaderAll

X,Y,speech,noise,mix = DataLoaderAll.data_loader()
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

gmm = GaussianMixture(n_components=52)
for i in tqdm(range(0,100)):
    gmm.fit(X[i][0])
print(gmm.fit_predict(X[0][0],Y[0]))