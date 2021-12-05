from sentence_transformers import SentenceTransformer
import torch
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import numpy as np

def read_file():
	f = open('syscall_list.txt')
	next(f)
	sentences = []
	syscalls = []
	for line in f.readlines():
		tokens = line[:-1].split(',')
		syscalls.append(tokens[1])
		sentences.append(tokens[2])
	f.close()
	return syscalls,sentences


def main():
	# read file('syscall_list.txt')
	syscalls,sentences = read_file()

	# sentence BERT
	model = SentenceTransformer('all-MiniLM-L6-v2')
	sentences_vec = model.encode(sentences)
	print(sentences_vec.shape)

	# PCA
	pca = PCA(n_components=16)
	pca_sentences = pca.fit_transform(sentences_vec)
	print(pca_sentences.shape)

	# save file('syscall_vec.npy')
	f = np.savez('syscall_vec.npz',name=syscalls,vec=pca_sentences)

	for i in range(len(syscalls)):
		print(syscalls[i],pca_sentences[i])


if __name__ == '__main__':  
	main()


