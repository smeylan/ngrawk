import random
import os
import itertools
import pandas as pd
import numpy as np
import io
import sys
sys.path.append('/home/stephan/python/espeak')
import espeak
sys.path.append('/home/stephan/python/ngrok')
import ngrok
import pdb
import scipy.stats
srilmPath = "/usr/share/srilm/bin/i686-m64"
os.environ['PATH'] = os.environ['PATH']+':'+srilmPath

#shuffle_word_contents: permute order of phones or characters within a word; same length and unigram probability statistics

#shuffle_word_assignments: permute coupling of phonological form and word

#draw_contents_with_length: keep the phonological inventory; draw n characters (without replacement) to make new words


def writeTokenFile(df, outputpath, word_column, count_column):
    words = df[word_column]
    counts = df[count_column]
    with io.open(outputpath, 'w', encoding='utf-8') as outfile:
        for i in range(df.shape[0]):
            word = words[i]
            for j in range(counts[i]):
                outfile.write(word+'\n')


def runLanguage(lang):	
	lex = pd.read_csv('/shared_ssd/ss/may8_tokenWeighted/OPUS/'+lang+'/00_lexicalSurprisal/opus_meanSurprisal.csv', encoding='utf-8')
	sublex = pd.read_csv('/shared_ssd/ss/may8_tokenWeighted/OPUS/'+lang+'/01_sublexicalSurprisal/25000_sublex.csv', encoding='utf-8')
	merged = lex.merge(sublex)


	#itertools trick to get the space
	paramDF = pd.DataFrame(list(itertools.product(*[['draw_contents_with_length'],['type','token'],['character'],range(5),[lang]])))
	paramDF.columns = ['permutationType','weightingType','substrate','run_index','lang']
	run_args = paramDF.T.to_dict().values()

	correlations = [computeCorrelationsUnderPemutation(merged, x) for x in run_args]

	correlationDF = pd.DataFrame(correlations)

	correlationDF['lang'] = lang
	correlationDF.to_csv(lang+'_permuted.csv')
	return(correlationDF)


def computeCorrelationsUnderPemutation(merged, run_args):
	# PERMUTATION TYPE
	if run_args['permutationType'] == 'shuffle_word_assignments':
		merged['permuted_word'] = np.random.permutation(merged['word']) # length and unigram prob free
	elif run_args['permutationType'] == 'shuffle_word_contents':	
		merged['permuted_word'] = [''.join(random.sample(s,len(s))) for s in merged['word']] # length and unigram prob fixed
	elif run_args['permutationType'] == 'draw_contents_with_length':	

		#make a single list of all the characters; draw ipa_n or ortho_n at each instance
		all_chars = list(''.join(merged['word']))
		random.shuffle(all_chars)

		#chop all_chars with the lengths in ortho_n
		slices = np.cumsum(merged.ortho_n)[0:len(merged.ortho_n)-1]
		merged['permuted_word'] = [''.join(x) for x in np.split(np.array(all_chars), indices_or_sections=slices)] 
	else:
		raise NotImplementedError	

	# TYPE OF TOKEN WEIGHTING
	permutedFile = '/home/stephan/scratch/permute_test.txt'
	letterizedFile= '/home/stephan/scratch/letterize_test.txt'
	lm_path = '/home/stephan/scratch/letterize_test.LM'	

	if run_args['weightingType'] == 'token':
		writeTokenFile(merged, permutedFile, 'permuted_word', 'frequency')

	elif run_args['weightingType'] == 'type':		
		
		merged[['permuted_word']].to_csv(permutedFile, header=None,encoding='utf-8')		
	else: 	
		raise NotImplementedError

	
	if run_args['substrate'] == 'ipa':	
		ngrok.letterize2(permutedFile, letterizedFile, filterfile='/shared_hd0/corpora/OPUS/2013_OPUS/intermediatecount/'+run_args['lang']+'_2013_collapsed.txt', splitwords=True, espeak_lang=run_args['lang'], phonebreak=' ', par=True, espeak_model=None)
	elif run_args['substrate'] == 'character':		
		ngrok.letterize2(permutedFile, letterizedFile, filterfile='/shared_hd0/corpora/OPUS/2013_OPUS/intermediatecount/'+run_args['lang']+'_2013_collapsed.txt', splitwords=True, espeak_lang=None, phonebreak=' ', par=True, espeak_model=None)
	else: 	
		raise NotImplementedError


	LM = ngrok.trainTokenModel(letterizedFile, 5, lm_path)

	merged['permuted_pic'] = [ngrok.getSublexicalSurprisal(list(x), LM, 5, 'letters', returnSum=True) for x in merged['permuted_word']]

	rv = run_args
	rv['permuted_cor'] = scipy.stats.spearmanr(merged['frequency'], merged['permuted_pic'])[0]

	return(rv)


languages = ['en','fr','de','he','es','ru','it','pt','sv','cs','pl','ro', 'nl']

all_DF = pd.concat([runLanguage(lang) for lang in languages])

all_DF.to_csv('all_languages_permutations.csv')