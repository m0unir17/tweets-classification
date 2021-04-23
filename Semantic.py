#!usr/bin/python2.7
# -*- coding: utf-8 -*-
from __future__ import division
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys

# Parameters to the algorithm. Currently set to values that was reported
# in the paper to produce "best" results.
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

######################### word similarity ##########################

def get_best_synset_pair(word_1, word_2):
    """ 
    Choose the pair with highest path similarity among all pairs. 
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               if sim > max_sim:
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic 
    ontology (Wordnet in our case as well as the paper's) between two 
    synsets.
    """
    l_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])        
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)

def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that 
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxint
    if synset_1 is None or synset_2 is None: 
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if hypernyms_1.has_key(lcs_candidate):
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if hypernyms_2.has_key(lcs_candidate):
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) / 
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))
    
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) * 
        hierarchy_dist(synset_pair[0], synset_pair[1]))

######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if sim > max_sim:
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim
    
def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if not brown_freqs.has_key(word):
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if not brown_freqs.has_key(lookup_word) else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))
    
def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are 
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec                
            
def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the 
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the 
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last 
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
        (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)
        
######################### main / test ##########################

# the results of the algorithm are largely dependent on the results of 
# the word similarities, so we should test this first...
"""
word_pairs = [
  ["asylum", "fruit", 0.21],
  ["autograph", "shore", 0.29],
  ["autograph", "signature", 0.55],
  ["automobile", "car", 0.64],
  ["bird", "woodland", 0.33],
  ["boy", "rooster", 0.53],
  ["boy", "lad", 0.66],
  ["boy", "sage", 0.51],
  ["cemetery", "graveyard", 0.73],
  ["coast", "forest", 0.36],
  ["coast", "shore", 0.76],
  ["cock", "rooster", 1.00],
  ["cord", "smile", 0.33],
  ["cord", "string", 0.68],
  ["cushion", "pillow", 0.66],
  ["forest", "graveyard", 0.55],
  ["forest", "woodland", 0.70],
  ["furnace", "stove", 0.72],
  ["glass", "tumbler", 0.65],
  ["grin", "smile", 0.49],
  ["gem", "jewel", 0.83],
  ["hill", "woodland", 0.59],
  ["hill", "mound", 0.74],
  ["implement", "tool", 0.75],
  ["journey", "voyage", 0.52],
  ["magician", "oracle", 0.44],
  ["magician", "wizard", 0.65],
  ["midday", "noon", 1.0],
  ["oracle", "sage", 0.43],
  ["serf", "slave", 0.39]
]
for word_pair in word_pairs:
    print "%s\t%s\t%.2f\t%.2f" % (word_pair[0], word_pair[1], word_pair[2], 
                                  word_similarity(word_pair[0], word_pair[1]))
"""
"""
import time
start_time = time.time()
sentence_pairs = [
    ["I like that bachelor.", "I like that unmarried man.", 0.561],
    ["John is very nice.", "Is John very nice?", 0.977],
    ["Red alcoholic drink.", "A bottle of wine.", 0.585],
    ["Red alcoholic drink.", "Fresh orange juice.", 0.611],
    ["Red alcoholic drink.", "An English dictionary.", 0.0],
    ["Red alcoholic drink.", "Fresh apple juice.", 0.420],
    ["A glass of cider.", "A full cup of apple juice.", 0.678],
    ["It is a dog.", "That must be your dog.", 0.739],
    ["It is a dog.", "It is a log.", 0.623],
    ["It is a dog.", "It is a pig.", 0.790],
    ["Dogs are animals.", "They are common pets.", 0.738],
    ["Canis familiaris are animals.", "Dogs are common pets.", 0.362],
    ["I have a pen.", "Where do you live?", 0.0],
    ["I have a pen.", "Where is ink?", 0.129],
    ["I have a hammer.", "Take some nails.", 0.508],
    ["I have a hammer.", "Take some apples.", 0.121]
]
for sent_pair in sentence_pairs:
    print "%s\t%s\t%.3f\t%.3f\t%.3f" % (sent_pair[0], sent_pair[1], sent_pair[2],similarity(sent_pair[0], sent_pair[1], False),
        similarity(sent_pair[0], sent_pair[1], True))

interval = time.time()-start_time
print "Total time  : ",interval
"""

### el program del general 
"""

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
   def _fromUtf8(s):
       return s

try:
   _encoding = QtGui.QApplication.UnicodeUTF8
   def _translate(context, text, disambig):
       return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
   def _translate(context, text, disambig):
       return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form(object):
    def setupUi(self, Form):
       Form.setObjectName(_fromUtf8("Form"))
       Form.resize(530, 367)
       self.label = QtGui.QLabel(Form)
       self.label.setGeometry(QtCore.QRect(160, 150, 191, 41))
       self.label.setObjectName(_fromUtf8("label"))
       self.label_2 = QtGui.QLabel(Form)
       self.label_2.setGeometry(QtCore.QRect(150, 190, 221, 71))
       self.label_2.setObjectName(_fromUtf8("label_2"))
       self.label_3 = QtGui.QLabel(Form)
       self.label_3.setGeometry(QtCore.QRect(40, 250, 341, 61))
       self.label_3.setObjectName(_fromUtf8("label_3"))
       self.pushButton = QtGui.QPushButton(Form)
       self.pushButton.setGeometry(QtCore.QRect(370, 330, 99, 27))
       self.pushButton.setObjectName(_fromUtf8("pushButton"))
       self.label_4 = QtGui.QLabel(Form)
       self.label_4.setGeometry(QtCore.QRect(230, 60, 191, 71))
       self.label_4.setObjectName(_fromUtf8("label_4"))
       self.lineEdit = QtGui.QLineEdit(Form)
       self.lineEdit.setGeometry(QtCore.QRect(350, 160, 113, 27))
       self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
       self.lineEdit_2 = QtGui.QLineEdit(Form)
       self.lineEdit_2.setGeometry(QtCore.QRect(360, 210, 113, 27))
       self.lineEdit_2.setObjectName(_fromUtf8("lineEdit_2"))
       self.lineEdit_3 = QtGui.QLineEdit(Form)
       self.lineEdit_3.setGeometry(QtCore.QRect(360, 270, 113, 27))
       self.lineEdit_3.setObjectName(_fromUtf8("lineEdit_3"))
       self.lineEdit.setText("%d"%(interval))
       self.retranslateUi(Form)
       QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")), exit)
       QtCore.QMetaObject.connectSlotsByName(Form)
    def retranslateUi(self, Form):
       Form.setWindowTitle(_translate("Form", "Form", None))
       self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#00007f;\">Taux de reussite = </span></p></body></html>", None))
       self.label_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#00007f;\">Temps d\'execution = </span></p></body></html>", None))
       self.label_3.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:16pt; font-weight:600; color:#00007f;\">Temps d\'execution d\'un tweet = </span></p></body></html>", None))
       self.pushButton.setText(_translate("Form", "OK", None))
       self.label_4.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:20pt; font-weight:600; color:#00007f;\">Résultats</span></p></body></html>", None))
"""


### EL Nosotros Programe 
import sys
import csv
reload(sys)
sys.setdefaultencoding("utf-8")


##### ----------------KNN TEST -----------------
def knn_test(tweet,appr,k):
  tab = [([""] * 3) for i in range(len(appr))]
  
  bar2= ProgressBar(total=len(appr))
  
  print " dok tebda la boucle "
  bar2.show()
  ma=len(appr)
  cp=0
  for appr in appr:
   
    print "boucle ",(cp)
    print appr[1]
    d=similarity(tweet[1],appr[1],False)
    tab[cp][0]=appr[0]
    tab[cp][1]=appr[1]
    tab[cp][2]=d
    bar2.update_progressbar(cp)
    print "2 %s"%( tab[cp])
    i+=1
    cp+=1

  
  tab=sorted(tab, key=lambda colonnes: colonnes[2],reverse=True)
  print tab
  po=0
  no=0
  
  for hh in k:
    if tab[hh][0]=="POLIT":
     po=po+1
    else:
     no=no+1
    hh+=1
  politic="POLIT"
  nott="NOT"
  if po>no:
    tweet[0]=politic
  else:
    tweet[0]=nott
  f=open("/home/mounir/Bureau/tweetjdid.txt",'a')
  f.write("%s\t%s\n"%(tweet[0],tweet[1]))
  f.close()
########## INTERFACE - resultat ---######




 


# El main ############# -- Hada howa el main aaaaaa ---#########



###### interface ------------ interface ------- INTERFAACE #########

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(598, 330)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.frame = QtGui.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 601, 281))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.label = QtGui.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(20, 50, 301, 61))
        self.label.setObjectName(_fromUtf8("label"))
        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(110, 100, 211, 61))
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.label_3 = QtGui.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(280, 150, 41, 61))
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.pushButton = QtGui.QPushButton(self.frame)
        self.pushButton.setGeometry(QtCore.QRect(450, 220, 99, 27))
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.pushButton_2 = QtGui.QPushButton(self.frame)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 220, 99, 27))
        self.pushButton_2.setObjectName(_fromUtf8("pushButton_2"))
        self.pushButton_3 = QtGui.QPushButton(self.frame)
        self.pushButton_3.setGeometry(QtCore.QRect(360, 70, 111, 27))
        self.pushButton_3.setObjectName(_fromUtf8("pushButton_3"))
        self.pushButton_4 = QtGui.QPushButton(self.frame)
        self.pushButton_4.setGeometry(QtCore.QRect(360, 120, 111, 27))
        self.pushButton_4.setObjectName(_fromUtf8("pushButton_4"))
        self.lineEdit = QtGui.QLineEdit(self.frame)
        self.lineEdit.setGeometry(QtCore.QRect(360, 170, 113, 27))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 598, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuA_propos = QtGui.QMenu(self.menubar)
        self.menuA_propos.setObjectName(_fromUtf8("menuA_propos"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.actionAide = QtGui.QAction(MainWindow)
        self.actionAide.setObjectName(_fromUtf8("actionAide"))
        self.menuA_propos.addAction(self.actionAide)
        self.menubar.addAction(self.menuA_propos.menuAction())
        self.label.setBuddy(self.pushButton_3)
        self.label_2.setBuddy(self.pushButton_4)
        self.label_3.setBuddy(self.lineEdit)
        global text
        text=self.lineEdit.text()

        self.retranslateUi(MainWindow)
        QtCore.QObject.connect(self.pushButton, QtCore.SIGNAL(_fromUtf8("clicked()")),knn )
        QtCore.QObject.connect(self.pushButton_2, QtCore.SIGNAL(_fromUtf8("clicked()")), exit)
        QtCore.QObject.connect(self.pushButton_4, QtCore.SIGNAL(_fromUtf8("clicked()")), selectFileTest)
        QtCore.QObject.connect(self.pushButton_3, QtCore.SIGNAL(_fromUtf8("clicked()")), selectFileApp)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Classification", None))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600; color:#0000ff;\">Ensemble d\'apprentissage</span></p></body></html>", None))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600; color:#0000ff;\">Ensemble de test</span></p></body></html>", None))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:18pt; font-weight:600; color:#0000ff;\">K = </span></p></body></html>", None))
        self.pushButton.setText(_translate("MainWindow", "OK", None))
        self.pushButton_2.setText(_translate("MainWindow", "Annuler", None))
        self.pushButton_3.setText(_translate("MainWindow", "Parcourir", None))
        self.pushButton_4.setText(_translate("MainWindow", "Parcourir", None))
        self.menuA_propos.setTitle(_translate("MainWindow", "a propos", None))
        self.actionAide.setText(_translate("MainWindow", "aide", None))
############  Interface du resultat --------


from Tkinter import *  #Pour python3.x Tkinter devient tkinter
 
class ApplicationBasic():
	'''Application principale'''
	def __init__(self):
		'''constructeur'''
		self.fen = Tk()
                self.fen.configure(width=500,height=400)
		self.fen.title('Resultat')
 
		self.message1 = Label(self.fen, text="Resultat").grid(row=0, column=1)

                self.message2 = Label(self.fen, text="Taux de réussite").grid(row=1)

                self.message5 = Label(self.fen, text="%.2f %%"%(taux)).grid(row=1, column=2)

                self.message3 = Label(self.fen, text="Temps d'execution pour 1 tweet ").grid(row=2)

                self.message6 = Label(self.fen, text="%d s"%(tweettime)).grid(row=2, column=2)

                self.message4 = Label(self.fen, text="Temp d'execution globale ").grid(row=5)

                self.message7 = Label(self.fen, text="%d s"%(interval)).grid(row=5, column=2)
 
		self.bou_quitter = Button(self.fen)
		self.bou_quitter.config(text='Quitter', command=self.fen.destroy)
		self.bou_quitter.grid(row=6, column=2)
 
	

	
 
 


############ PROGRESS BAR ---------

class ProgressBar(QtGui.QWidget):
    def __init__(self, parent=None, total=20):
        super(ProgressBar, self).__init__(parent)
        self.name_line = QtGui.QLineEdit()

        self.progressbar = QtGui.QProgressBar()
        self.progressbar.setMinimum(1)
        self.progressbar.setMaximum(total)

        main_layout = QtGui.QGridLayout()
        main_layout.addWidget(self.progressbar, 0, 0)

        self.setLayout(main_layout)
        self.setWindowTitle("Progress")

    def update_progressbar(self, val):
        self.progressbar.setValue(val)   

### main te3 interface ########## 


from PyQt4.QtGui import QFileDialog

def selectFileTest():
      global pathTest
      pathTest=QFileDialog.getOpenFileName()

def selectFileApp():
      global pathApp
      pathApp=QFileDialog.getOpenFileName()

def knn():
   import time 
   k=text
   print "hello 1"
   
   appr = list(csv.reader(open(pathApp, 'rb'), delimiter='\t'))

   test= list(csv.reader(open(pathTest, 'rb'), delimiter='\t'))
   
   start_time=time.time()
   print "dok yebda knn " 
   bar = ProgressBar(total=len(test))
   
#ndiro boucle pour tous le dataset du test

   testtaille=len(test)
   print " hadi fi khater test"
   l=0
   bar.show()
   for tes in test:
      
        bar.update_progressbar(l)
        st=time.time()
        knn_test(tes,appr,k)
        global tweettime
        tweettime=time.time()-st
        l+=1
   res = list(csv.reader(open("/home/mounir/Bureau/tweetjdid.txt", 'rb'), delimiter='\t'))
   i=0
   cpt=0
   test = list(csv.reader(open(pathTest, 'rb'), delimiter='\t'))
   for tes in res:
    if tes[0]==test[i][0]:
     cpt+=1
   global taux
   taux=(cpt/len(test))*100
   print "startrek"
   global interval
   interval = time.time()-start_time
   if __name__ == '__main__':
	app = ApplicationBasic()
        app.fen.mainloop() 
        
   print "c'est fait en %d " %(interval)
   
if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


