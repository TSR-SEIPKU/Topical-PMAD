
import numpy as np
#number of topic
num_topic = 20

#two group of attributes:a and b

#number of attribute types in group a
num_voc_a = 200
#number of attribute types in group b
num_voc_b = 400

#number of samples
num_normal_samples = 9900
num_abnormal_samples = 100

thetas = np.random.dirichlet(np.zeros([num_topic])+2.0/num_topic,num_normal_samples)
thetas_ab = np.random.dirichlet(np.zeros([num_topic])+2.0/num_topic,num_abnormal_samples*2)

phi_as = np.random.dirichlet(np.zeros([num_voc_a])+2.0/num_voc_a,num_topic)
phi_bs = np.random.dirichlet(np.zeros([num_voc_b])+2.0/num_voc_b,num_topic)

#length of each document
lengths_a = np.random.normal(100, 50, num_normal_samples + num_abnormal_samples)
lengths_b = np.random.normal(5, 2, num_normal_samples + num_abnormal_samples)

file_a = open('fa-part-sp', 'w')
file_b = open('fb-part-sp', 'w')

def write_doc(isAb, i, lan, theta, phis, length, f):
	if length < 1:
		length = 1
	f.write(isAb + str(i) + "\t" + lan + "\t")
	topic_counts = np.random.multinomial(length, theta)
	t = 0
	word_counts = np.zeros([len(phis[0])])
	for count in topic_counts:
		phi = phis[t]
		if count != 0:
			word_counts += np.random.multinomial(count, phi)
		t += 1
	w = 0
	doc = ''
	for wcount in word_counts:
		if wcount != 0:
			string = (lan+str(w)+' ')*wcount
			doc += string
		w += 1
	f.write(doc.strip(' ')+"\n")
	pass

# generate normal samples
for i in range(0,num_normal_samples):
	theta = thetas[i]
	write_doc('', i, 'a', theta, phi_as, lengths_a[i], file_a)
	write_doc('', i, 'b', theta, phi_bs, lengths_b[i], file_b)


# generate anomalous samples
for i in range(0,num_abnormal_samples):
	theta_a = thetas_ab[i*2]
	write_doc('ab-', i, 'a', theta_a, phi_as, lengths_a[num_normal_samples+i], file_a)
	theta_b = thetas_ab[i*2+1]
	write_doc('ab-', i, 'b', theta_b, phi_bs, lengths_b[num_normal_samples+i], file_b)

	