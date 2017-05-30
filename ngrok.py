#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, subprocess, re, sys, itertools, codecs, gzip, glob, unicodedata, click, pandas, srilm, pdb, json, multiprocessing, time, tempfile, math, scipy, warnings, codecs, aspell, unidecode, espeak
from zs import ZS
from scipy import stats
import joblib, multiprocessing
from joblib import delayed, Parallel
import numpy as np
import io

class cgWorker(multiprocessing.Process):
    '''single-thread worker for parallelized cleanGoogle function'''  
    def __init__(self,queue,myList):
        super(cgWorker, self).__init__()
        self.queue = queue
        self.myList = myList
        
    def run(self):    	
        for job in iter(self.queue.get, None): # Call until the sentinel None is returned
        	try:
        		cleanGoogle(job['inputfile'], job['outputfile'], job['collapseyears'],job['filetype'], job['order'])        
        	except ValueError:
        		print 'Problems encountered in cleaning '+job['inputfile']
			self.myList.append(job['inputfile'])

def cleanGoogleDirectory(inputdir, outputdir, collapseyears, order):
	'''Parallelized, load-balanced execution of cleanGoogle, starting with the largest files'''
	start_time =  time.time()

	# Put the manager in charge of how the processes access the list
	mgr = multiprocessing.Manager()
	myList = mgr.list() 
    
	# FIFO Queue for multiprocessing
	q = multiprocessing.Queue()
    
	# Start and keep track of processes
	procs = []
	for i in range(12):
		p = cgWorker( q,myList )
		procs.append(p)
		p.start()
	              
	files = glob.glob(os.path.join(inputdir,'*.gz')) + glob.glob(os.path.join(inputdir,'*.zip')) 
	if len(files) > 0:
		print('File type is gz')	
		filetype = 'gz'
	else:
		files = glob.glob(os.path.join(inputdir,'*.bz2'))
		if len(files) > 0:
			print('File type is bz2')	
			filetype = 'bz2'	
		else:
			raise ValueError('No files found')		
		
	filesizes = [(x, os.stat(x).st_size) for x in files]
	filesizes.sort(key=lambda tup: tup[1], reverse=True)
	
	extension = '.yc' if collapseyears else '.output'
	# Add data, in the form of a dictionary to the queue for our processeses to grab    
	[q.put({"inputfile": file[0], "outputfile": os.path.join(outputdir, os.path.splitext(os.path.basename(file[0]))[0]+extension),"collapseyears": collapseyears, 'filetype':filetype, 'order':order}) for file in filesizes] 
      
	#append none to kill the workers with poison pills		
	for i in range(12):
		q.put(None) #24 sentinels to kill 24 workers
        
	# Ensure all processes have finished and will be terminated by the OS
	for p in procs:
		p.join()     
        
	for item in myList:
		print(item)

	print('Done! processed '+str(len(myList))+' files; elapsed time is '+str(round(time.time()-start_time /  60., 5))+' minutes') 	


def collapseNgrams(inputfile, outputfile):	
	'''aggregate across dates from a google-formatted ngram file'''
	bufsize = 10000000
	print('Collapsing years...')	
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	off = codecs.open(outputfile, 'w', encoding='utf-8')		
	firstLine ='\n' #handle any lines that are blank at the beginning of the text
	#need to confirm that there is anything in the file
	while firstLine == '\n' or firstLine == '':
		firstLine = iff.readline()	

	lineSplit = firstLine.split('\t')
	prev_ngram = lineSplit[0]

	if len(lineSplit) == 5:		
		print('5 tab-delineated columns, assuming first is the ngram, second is the year, third is the token, and the fourth the context count')
		ncols = 5
		cached_count = int(lineSplit[2])
	if len(lineSplit) == 4:		
		print('4 tab-delineated columns, assuming first is the ngram, second is the year, third is the token, and the fourth the context count')
		ncols = 4
		cached_count = int(lineSplit[2])
	elif len(lineSplit) == 2:
		print('2 tab-delineated columns, assuming first is the ngram, second is the token count')	
		ncols = 2
		cached_count = int(lineSplit[-1])		
				
	rows =[]

	for c,l in enumerate(iff):
		line = l.split('\t')		
		if len(line) != ncols:
			print 'Mismatch in line length and ncols, line was '+line[0]
			continue

		ngram = line[0]
		if ncols == 2:
			count = int(line[1]) #second column is the token count		
		elif ncols in (4,5) :
			count = int(line[2]) #third column is the token count	
				
		if(ngram != prev_ngram): #new ngram, write out the cached one			
			#after appending row to the buffer, reset the storage
			rows.append('\t'.join([prev_ngram, str(cached_count)]))

			prev_ngram = ngram
			cached_count = count
			
		else:
			cached_count += count			

		if c % bufsize == 0:	
			off.write('\n'.join(rows)+'\n')
			rows =[] 
	
	rows.append('\t'.join([prev_ngram, str(cached_count)])) # catch the last record			

	off.write('\n'.join(rows)+'\n')	#catch any records since the last buffered write						 	
	iff.close()
	off.close()
	print('Finished collapsing years, output in file '+str(outputfile))

def cleanGoogle(inputfile, outputfile, collapseyears, filetype, order):
	'''Clean google trigram file. This is a highly streamlined version of process google that finds only non POS-tagged lines, with no punctuation, and makes them lowercase, using grep to find lines without punctuation (including _, which excludes lines with POS tags) and perl to lowercase the string, while maintaining the unicode encoding. If collapseyears is true, combine the year counts into a single record'''
	tempfile0 = inputfile+'_temp0'
	if collapseyears:		
		tempfile1 = inputfile+'_temp1'
		
		if filetype in ('gz', 'csv.zip'):
			cleanGoogleCommand = "zcat "+inputfile+" | LC_ALL=C grep -v '[]_,.!\"#$%&()*+-/:;<>=@^{|}~[]' | perl -CSD -ne 'print lc' > "+tempfile1
		elif filetype == 'bz2':
			cleanGoogleCommand = "bzcat "+inputfile+" | LC_ALL=C grep -v '[]_,.!\"#$%&()*+-/:;<>=@^{|}~[]' | perl -CSD -ne 'print lc' > "+tempfile1	
		
		os.system(cleanGoogleCommand)
		if os.stat(tempfile1).st_size > 0 :	
			collapseNgrams(tempfile1, tempfile0) # this means that there are separate records for lowercase and uppercase items
		else:
			return(None)
			'Temp file has no content; safe to remove.'	
		os.remove(tempfile1)
	else:	
		if filetype in ('gz', 'csv.zip'):
			cleanGoogleCommand = "zcat "+inputfile+" | LC_ALL=C grep -v '[]_,.!\"#$%&()*+-/:;<>=@^{|}~[]' | perl -CSD -ne 'print lc' > "+tempfile0
		elif filetype == 'bz2':
			cleanGoogleCommand = "bzcat "+inputfile+" | LC_ALL=C grep -v '[]_,.!\"#$%&()*+-/:;<>=@^{|}~[]' | perl -CSD -ne 'print lc' > "+tempfile		
		os.system(cleanGoogleCommand)

	fixPunctuation(tempfile0, outputfile, order)	#remove the punctuation
	os.remove(tempfile0)
	return(outputfile)

def remove_punctuation(text, tbl):
	'''remove punctuation from UTF8 strings given a character table'''
	return text.translate(tbl)

def combineFiles(inputdir, pattern, outputfile):
	'''combines a set of text files in directory with filenames terminating with pattern into a single file; a wrapper for GNU cat'''
	print('Combining the cleaned files...')	
	catCommand = 'cat '+os.path.join(inputdir,pattern)+' > '+outputfile
	subprocess.call(catCommand, shell=True)
	print('Done!')

def sortNgramFile(inputfile, outputfile):
	'''sorts an ngram file; basically a wrapper for GNU sort'''
	print('Sorting the combined file...')	
	sortCommand = 'env LC_ALL=C sort --compress-program=lzop '+inputfile+'  -o '+outputfile+' --parallel=24'
	subprocess.call(sortCommand, shell=True)
	print('Done!')
	
def makeLanguageModel(inputfile, outputfile, metadata, codec):
	'''Take the cleaned and sorted file and put it into ZS file'''		
	print('Building the language model...')
	zs_command = 'zs make \''+json.dumps(metadata)+'\' --codec='+codec+' '+inputfile + ' ' + outputfile 
	subprocess.call(zs_command, shell=True)

def reverseGoogleFile(inputfile, outputfile):
	'''Reverse the order of the ngram in a Google-formatted ngram file. Note that this is a different procedure than rearranging the ngram files that are output by AutoCorpus'''
	print('Reversing existing model')		
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	off = codecs.open(outputfile, 'w', encoding='utf-8')		
	for l in iff:
		strArray = l.split('\t')		
		if len(strArray) > 0: #this cleans any empty lines that are produced by the cleaning process
			ngram = strArray[0].split(' ')
			if len(ngram) > 0: #only retain proper ngrams
				strArray[0] = ' '.join(ngram[::-1])
				off.write('\t'.join(strArray))
	iff.close()
	off.close()
	print('Done!')


def reorderGoogleFile(inputfile, outputfile, index):
	'''Reorder the columns in a Google-formatted ngram file to put the word at targetWordIndex as the last item. This supports the reordering of columns so that the context is the preceding + following word, for example.'''
	if index < 1:
		raise ValueError('targetWordIndex should be indexed from 1 (like the order argument)')

	#indexing from 1, the target word index can either be 1, the length of the array, or length of array +1/2 (for the center embedded trigram. The function should return an error if there are an even number of items and the target word is not the first or last
	print('Reordering existing model')		
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	firstLine = ''
	while firstLine == '\n' or firstLine == '':
		firstLine = iff.readline()
	numWords = len(firstLine.split('\t')[0].split(' '))	
	if not index in (1, numWords, (numWords+1.)/2.):
		raise ValueError('targetWordIndex needs to be the first, last, or center item')
	iff.close()
	
	iff = codecs.open(inputfile, 'r', encoding='utf-8')			
	off = codecs.open(outputfile, 'w', encoding='utf-8')			
	for l in iff:
		strArray = l.split('\t')		
		if len(strArray) > 0: #this cleans any empty lines that are produced by the cleaning process
			ngram = strArray[0].split(' ')
			if len(ngram) > 0 and ngram != [u'\n']: #only retain proper ngrams	
				targetWord = [ngram[index-1]]
				context = ngram
				del context[index-1]
				strArray[0] = ' '.join(context+targetWord)
				off.write('\t'.join(strArray))
	iff.close()
	off.close()
	print('Done!')



def deriveFromHigherOrderModel(intermediatefiledir, n, direction):
	'''Search for a pre-computed model from which the desired counts can be derived either through reversing or marginalization'''
	if direction == 'forwards':
		oppositeDirection = 'backwards'
	elif direction == 'backwards':
		oppositeDirection = 'forwards'		
	
	#first look for a model in the same direction that is larger than the desired n

	availableModels = glob.glob(os.path.join(intermediatefiledir,'*'+direction+'-collapsed.txt'))
	modelOrders = [os.path.basename(x)[0] for x in availableModels if x > int(n)]
	if len(modelOrders) > 0:
		print 'Higher order model of same direction found; will marginalize counts...'
		NtoUse = min(modelOrders)
		inputfile = os.path.join(intermediatefiledir,str(NtoUse)+'gram-'+direction+'-collapsed.txt')
		outputfile = os.path.join(intermediatefiledir,str(n)+'gram-'+direction+'-collapsed.txt')
		#!!! sort before marginalization! may be okay
		marginalizeNgramFile(inputfile, outputfile, n, 'alphabetic')
		return(outputfile)
	else: #no models in the same direction, may need to reverse one
		availableModels = glob.glob(os.path.join(intermediatefiledir,'*'+oppositeDirection+'-collapsed.txt'))	 #look for ones of the opposite direction		
		modelOrders = [int(os.path.basename(x)[0]) for x in availableModels if x > int(n)]

		if len(modelOrders) > 0: # if there is at least one higher-order opposite-direction model
			print 'Higher order model of different direction found; will reverse, sort, and marginalize'
			NtoUse = min(modelOrders)						

			#reverse it-- the higher order model MUST be reversed before marginalization, or some low frequency trigrams are lost
			startingModel = os.path.join(intermediatefiledir,str(NtoUse)+'gram-'+oppositeDirection+'-collapsed.txt')
			desiredDirectionStartingFile = os.path.join(intermediatefiledir,str(NtoUse)+'gram-'+direction+'-combined.txt')
			reverseGoogleFile(startingModel, desiredDirectionStartingFile)

			#sort it
			sortedFile = os.path.join(intermediatefiledir,str(n)+'gram-'+direction+'-sorted.txt')
			sortNgramFile(desiredDirectionStartingFile, sortedFile)

			#marginalize it							
			marginalizedfile = os.path.join(intermediatefiledir,str(n)+'gram-'+direction+'-marginalized.txt')
			marginalizeNgramFile(sortedFile,marginalizedfile, n, 'alphabetic')
						
			collapsedFile = os.path.join(intermediatefiledir,str(n)+'gram-'+direction+'-collapsed.txt')

			os.system('cp '+marginalizedfile+' '+collapsedFile)
			return(collapsedFile)
		else:
			print 'No appropriate models found, proceeding to cleaning the source trigrams.'
			return(None)


def getGoogleBooksLanguageModel(corpusSpecification, n, direction, collapseyears, filetype):
	'''Metafunction to create a ZS language model from Google Ngram counts. Does a linear cleaning, merges the file into a single document, sorts it, collapses identical prefixes, and builds the ZS file.'''
	startTime = time.time()
	lexSurpDir = os.path.join(corpusSpecification['faststoragedir'], corpusSpecification['analysisname'],corpusSpecification['corpus'],corpusSpecification['language'],'00_lexicalSurprisal')

	if not collapseyears: #keeping dates is too large to keep the intermediate files on the ssd			
		intermediateFileDir = os.path.join(corpusSpecification['slowstoragedir'],corpusSpecification['corpus'],corpusSpecification['language'])
	else:
			intermediateFileDir	= lexSurpDir
	
	zs_metadata = {  #!!! should this be the corpusSpecification dictionary
		"corpus": corpusSpecification['corpus'],
		"language": corpusSpecification['language'],
		"n": n,
		"direction": direction
	}
	print zs_metadata
	
	zsFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'.zs')
	if not os.path.exists(zsFile):
		print('Checking if there are appropriate cleaned text files to create lower-order language model...')	
		tryHigher = deriveFromHigherOrderModel(intermediateFileDir, n, direction)	
		
		if tryHigher is not None:
			print('Derived model from higher order model, results are at '+str(tryHigher))
			collapsedfile = tryHigher
		else:
			print('No higher-order or reversible models found. Cleaning the input files... If n > 3 and the language is English, this is a good time to grab a coffee, this will take a few hours.')
			
			#find only lines without POS tags and make them lowercase
			inputdir = os.path.join(corpusSpecification['inputdir'],corpusSpecification['corpus'],corpusSpecification['language'],str(n))
			outputdir = os.path.join(corpusSpecification['slowstoragedir'],corpusSpecification['analysisname'], corpusSpecification['corpus'],corpusSpecification['language'],str(n)+'-processed')	
		
			combinedfile = os.path.join(intermediateFileDir,str(n)+'gram-'+direction+'-combined.txt')	

			if collapseyears:				
				cleanFileProp = checkForMissingFiles(inputdir, '*.'+filetype, outputdir, '*.yc')	
				if cleanFileProp < .2:
					cleanGoogleDirectory(inputdir,outputdir, collapseyears, n)
					checkForMissingFiles(inputdir, '*.'+ filetype, outputdir, '*.yc')	
				combineFiles(outputdir, '*.yc', combinedfile)	

			else:					
				cleanFileProp = checkForMissingFiles(inputdir, '*.'+filetype, outputdir, '*.output')
				if cleanFileProp < .2:
					cleanGoogleDirectory(inputdir,outputdir, collapseyears, n)
					checkForMissingFiles(inputdir, '*.'+filetype, outputdir, '*.output')
				combineFiles(outputdir, '*.output', combinedfile)		
			
			#reorder the columns if specified, e.g. to get center-embedded trigrams	
			if corpusSpecification['target'] != corpusSpecification['order']:				
				reorderedfile = os.path.join(intermediateFileDir,str(n)+'gram-'+direction+'-reordered.txt')
				reorderGoogleFile(combinedfile, reorderedfile, int(corpusSpecification['target']))
				fileToReverse = reorderedfile				
			else:
				fileToReverse = combinedfile

			#reverse if specified
			if direction == 'backwards':
				reversedfile = os.path.join(intermediateFileDir,str(n)+'gram-'+direction+'-reversed.txt')
				reverseGoogleFile(fileToReverse, reversedfile)
				fileToSort = reversedfile
			elif direction == 'forwards':	
				fileToSort = reorderedfile
			
			#sort it	
			sortedfile = os.path.join(intermediateFileDir,str(n)+'gram-'+direction+'-sorted.txt')
			sortNgramFile(fileToSort, sortedfile)		

			#collapse after the sorting: this deals with different POS treatments 
			collapsedfile = os.path.join(intermediateFileDir,str(n)+'gram-'+direction+'-collapsed.txt')
			collapseNgrams(sortedfile, collapsedfile)
								
		#build the language model	
		zsFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'.zs')	
		makeLanguageModel(collapsedfile, zsFile, zs_metadata, codec="none")

		print('Done! Completed file is at '+zsFile+'; elapsed time is '+str(round(time.time()-startTime, 5))+' seconds') 
	
	else:
		print('ZS file already exists at '+zsFile) 
	return(zsFile)

def makeDirectoryStructure(faststoragedir, slowstoragedir, analysisname, corpus, language, n):		
	print('Creating fast storage directory at '+os.path.join(faststoragedir, analysisname, corpus, language)+'...')	

	corpusLanguagePath = os.path.join(faststoragedir, analysisname, corpus, language)				
	lexSurpDir = os.path.join(faststoragedir, analysisname,corpus,language,'00_lexicalSurprisal')
	sublexSurpDir = os.path.join(faststoragedir, analysisname,corpus,language,'01_sublexicalSurprisal')
	correlationsDir = os.path.join(faststoragedir, analysisname, corpus,language,'02_correlations')

	if not os.path.exists(corpusLanguagePath):
		os.makedirs(corpusLanguagePath)
	if not os.path.exists(lexSurpDir):
		os.makedirs(lexSurpDir)	
	if not os.path.exists(sublexSurpDir):
		os.makedirs(sublexSurpDir)
	if not os.path.exists(correlationsDir):
		os.makedirs(correlationsDir)
	print('Fast directories created!')

	processedDir = os.path.join(slowstoragedir, analysisname, corpus, language)
	print('Creating slow storage directory at '+processedDir+'...')	
	if not os.path.exists(processedDir):
		os.makedirs(processedDir)

	#create directories for all n, n-1, 1	
	ordersToMake = [n, n-1, 1]
	for i in ordersToMake:
		pathToMake = os.path.join(processedDir, str(i)+'-processed')
		if not os.path.exists(pathToMake):
			os.makedirs(pathToMake)

	return lexSurpDir, sublexSurpDir, correlationsDir, processedDir


def analyzeCorpus(corpusSpecification):
	'''Conducts the analysis on a given dataset (corpus + language).'''	
	corpus = corpusSpecification['corpus'] 
	language = corpusSpecification['language'] 
	n = corpusSpecification['order'] 
	print('Processing '+corpus+':'+language)

	lexSurpDir, sublexSurpDir, correlationsDir, processedDir = makeDirectoryStructure(corpusSpecification['faststoragedir'], corpusSpecification['slowstoragedir'], corpusSpecification['analysisname'], corpusSpecification['corpus'], corpusSpecification['language'], int(corpusSpecification['order']))	
	
	#write model metadata in the form of a json of the corpusSpecification to the faststoragedir
	with open(os.path.join(corpusSpecification['faststoragedir'],
		corpusSpecification['analysisname'], corpusSpecification['corpus'], corpusSpecification['language'], 'model_details.json'), 'w') as outfile:
		json.dump(corpusSpecification, outfile)
	# read formatted version with cat model_details.json | python -m json.tool	

	if (corpus == 'GoogleBooks2012'):
		if (language in ('eng-all', 'spa-all', 'fre-all','ger-all','rus-all','test','heb-all','ita-all')):					
			print('Checking if input files exist...')			
			
			print('Building language models...')
			# get backwards-indexed model of highest order (n)
			backwardsNmodel = getGoogleBooksLanguageModel(corpusSpecification, int(n), direction='backwards', collapseyears=True, filetype='gz')
			# get forwards-indexed model of order n-1 (text file  built as a consequence)
			forwardsNminus1model = getGoogleBooksLanguageModel(corpusSpecification, int(n)-1, direction='forwards', collapseyears=True, filetype='gz')				
		else:
			raise NotImplementedError		
	elif(corpus == 'Google1T'):
		if (language in ('SPANISH','FRENCH','DUTCH','GERMAN','SWEDISH','CZECH','ROMANIAN','POLISH','PORTUGUESE','ITALIAN')):
			backwardsNmodel = getGoogleBooksLanguageModel(corpusSpecification, int(n), direction="backwards", collapseyears=True, filetype='bz2')
			forwardsNminus1model = getGoogleBooksLanguageModel(corpusSpecification, int(n)-1, direction="forwards", collapseyears=True, filetype='bz2')
		elif language in ('ENGLISH'):
			backwardsNmodel = getGoogleBooksLanguageModel(corpusSpecification, int(n), direction='backwards', collapseyears=True, filetype='gz')
			forwardsNminus1model = getGoogleBooksLanguageModel(corpusSpecification, int(n)-1, direction='forwards', collapseyears=True, filetype='gz')
	elif(corpus == 'GoogleBooks2009'):
		if (language in ('eng-all')):					
			print('Checking if input files exist...')			
			
			print('Building language models...')
			# get backwards-indexed model of highest order (n)
			backwardsNmodel = getGoogleBooksLanguageModel(corpusSpecification, int(n), direction='backwards', collapseyears=True, filetype='csv.zip')
			# get forwards-indexed model of order n-1 (text file  built as a consequence)
			forwardsNminus1model = getGoogleBooksLanguageModel(corpusSpecification, int(n)-1, direction='forwards', collapseyears=True, filetype='csv.zip')				
		else:
			raise NotImplementedError		


	elif(corpus in ('BNC', 'OPUS')):
		if (language in ('en', 'ru', 'es','fr','de','he','eng-half','eng','cs','pt', 'pl','ro','it','sv','nl','en-permuted')):

			print('Checking if input files exist...')
			#!!! check if file extant; if not, then download

			#!!! does buildZSfromPlaintext.py preserve unicode?	
			print('Building language models')
			# get backwards-indexed model of highest order (n)
			backwardsNmodel = getPlaintextLanguageModel(corpusSpecification, n, direction='backwards', cleaningFunction='cleanLine_BNC')
			# get forwards-indexed model of order n-1 (text file  built as a consequence)
			forwardsNminus1model = getPlaintextLanguageModel(corpusSpecification, int(n)-1, direction='forwards', cleaningFunction='cleanLine_BNC')
			#get unigrams to be able to take top N words in the analysis
			
		else:
			raise NotImplementedError	
	else:
		pdb.set_trace()
		raise NotImplementedError		
	
	#to use most frequent words from Google for the sublexical surprisal model
	forwardBigramPath = os.path.join(lexSurpDir, '2gram-forwards-collapsed.txt')
	unigramCountFilePath = os.path.join(lexSurpDir, 'unigram_list.txt')

	marginalizeNgramFile(forwardBigramPath, unigramCountFilePath, 1, 'numeric') 	

	#to use OPUS for the sublexical surprisal model:
	#unigramCountFilePath = corpusSpecification['wordlist']	

	print('Getting mean lexical surprisal estimates for types in the langauge...')
	forwardsNminus1txt = os.path.join(lexSurpDir,str(int(n)-1)+'gram-forwards-collapsed.txt')

	lexfile = os.path.join(lexSurpDir, 'opus_meanSurprisal.csv')	

	getMeanSurprisal(backwardsNmodel, forwardsNminus1txt, corpusSpecification['filterlist'],corpusSpecification['retrievallist'], 0,lexfile, corpusSpecification['country_code'], corpusSpecification['retrievalcount'], corpusSpecification['dictionary_filter'])	

	sublexFilePath = os.path.join(sublexSurpDir, str(corpusSpecification['modelcount'])+'_sublex.csv')

	for sublex_measure in corpusSpecification['sublex_measures']:

		if sublex_measure == 'ipa':
			print('Getting sublexical surprisal estimates for types in the language, using IPA...')
		elif sublex_measure == 'token_ipa':
			print('Getting sublexical surprisal estimates for tokens in the language, using IPA...')			
		elif sublex_measure == 'ortho':	
			print('Getting sublexical surprisal estimates for types in the language, using orthography...')					
		elif sublex_measure == 'character':	
			print('Getting sublexical surprisal estimate for types in the language, building it over the characters')
		elif sublex_measure == 'token_character':
			print('Getting sublexical surprisal estimates for tokens in the language, using characters...')
		else:
			raise ValueError('Sublexical measure not implmented. Use ipa, ortho, or character')	
		
		addSublexicalSurprisals(corpusSpecification['retrievallist'], corpusSpecification['retrievalcount'], corpusSpecification['modellist'], corpusSpecification['modelcount'], sublexFilePath, sublex_measure, corpusSpecification['country_code'], corpusSpecification['dictionary_filter'], corpusSpecification['smoothing'], corpusSpecification['token_corpus'], corpusSpecification['token_filter'])
	

def getPlaintextLanguageModel(corpusSpecification, n, direction, cleaningFunction):	
	'''This metafunction produces a ZS language model from a large plaintext document using the program "ngrams" from the AutoCorpus Debian package to count the n-gram frequencies for a specified order (n). Example use: for producing a ZS file from the BNC.'''
	startTime = time.time()
	lexSurpDir = os.path.join(corpusSpecification['faststoragedir'], corpusSpecification['analysisname'],corpusSpecification['corpus'],corpusSpecification['language'],'00_lexicalSurprisal')	

	zs_metadata = corpusSpecification
	zs_metadata["zs_n"] = n
	zs_metadata["direction"] = direction
	print zs_metadata
	
	tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
	if unicodedata.category(unichr(i)).startswith('P'))
	
	inputfile = os.path.join(corpusSpecification['inputdir'],corpusSpecification['corpus'],corpusSpecification['language'],corpusSpecification['filename'])
	cleanedFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'-cleaned.txt')
	countedFile= os.path.join(lexSurpDir,str(n)+'gram-'+direction+'-counted.txt')
	countMovedFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'-countMoved.txt')
	sortedFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'-sorted.txt')
	collapsedFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'-collapsed.txt')
	zsFile = os.path.join(lexSurpDir,str(n)+'gram-'+direction+'.zs')	

	cleanTextFile(inputfile, cleanedFile, cleaningFunction)
	countNgrams(cleanedFile, countedFile, n)
	rearrangeNgramFile(countedFile, countMovedFile, direction)
	sortNgramFile(countMovedFile, sortedFile)
	os.system ("cp "+sortedFile+" "+collapsedFile) #this just copies it, so filenames are equivalent to the google procedure
	makeLanguageModel(collapsedFile, zsFile, zs_metadata, 'none')

	print('Done! Completed file is at '+zsFile+'; elapsed time is '+str(round(time.time()-startTime, 5))+' seconds') 
	return(zsFile)


def readCSVorTxt(filename): 
	'''load a CSV or a TXT file without complaining'''
	file_path, file_extension = os.path.splitext(filename)
	if(file_extension=='.txt'):
		df = pandas.read_table(filename, encoding='utf-8', keep_default_na=False, na_values=[]).dropna()
	elif(file_extension=='.csv'):
		df = pandas.read_csv(filename, encoding='utf-8', keep_default_na=False, na_values=[]).dropna()
	return(df)

def rearrangeNgramFile(inputfile, outputfile, direction):	
	print('Rearranging the ngrams...')	
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	off = codecs.open(outputfile, 'w', encoding='utf-8')	
	for l in iff:		
		l = l.replace('\n','')
		strArray = l.split('\t')
		if (len(strArray) == 1):
			continue
		else:	
			count = strArray.pop(0)	
			ngram = strArray[0].split(' ')
			if direction == 'backwards':
				ngram.reverse()
			strArray = [' '.join(ngram)]
			strArray.append(count) #move the count to the end, reverse ngram				
			off.write('\t'.join(strArray)+'\n')
	iff.close()
	off.close()


def marginalizeNgramFile(inputfile, outputfile, n, sorttype):
	'''collapse counts from inputfile for sequences of length n'''
	print('Marignalizing over counts from higher-order ngram file to produce counts of '+str(n)+'-grams')	
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	
	tf_path = os.path.join(os.path.dirname(inputfile),next(tempfile._get_candidate_names()))
	tf = open(tf_path, 'w')
	tf = codecs.open(tf_path, 'w', encoding='utf-8')	

	firstLine ='\n' #handle any lines that are blank at the beginning of the text
	while firstLine == '\n' or firstLine == '':
		firstLine = iff.readline()

	linesplit = firstLine.split('\t')
	cachedNgram = ' '.join(linesplit[0].split(' ')[0:n])
	cachedCount = int(linesplit[1]) 
	ncols =  len(linesplit)		

	print('Collapsing counts...')
	for l in iff:
		parts = l.split('\t')
		ngram = ' '.join(parts[0].split(' ')[0:n])		
		if ncols == 2:
			count = int(parts[1])
		elif ncols == 4:	
			count = int(parts[2])
		
		#if the ngram isn't the same, print out the last trigram and add it to the aggregate count, and restart the count
		if ngram != cachedNgram:
			#print('Added to total '+cachedNgram+': '+str(cachedCount))
			if (sorttype == 'numeric'):
				tf.write(str(cachedCount)+'\t'+cachedNgram+'\n')
			elif (sorttype == 'alphabetic'):
				tf.write(cachedNgram+'\t'+str(cachedCount)+'\n')	

			#restart the count, for the next ngram
			cachedNgram = ngram
			cachedCount = count			
		else:
			#if it is the same, add it to the aggregate count
			cachedCount += count			
			#print('Increased '+cachedNgram+' count to : '+str(cachedCount))
	#obligate write of final cached value at the end 
	if sorttype == 'alphabetic':               
		tf.write(cachedNgram+'\t'+str(cachedCount)+'\n')		
	else:
		tf.write(str(cachedCount)+'\t'+cachedNgram+'\n')		

	iff.close()
	tf.close()

	print('Sorting new counts...')
	#then run sort on the output file
	if (sorttype == 'numeric'):
		os.system("sort -n -r "+tf_path+' > '+outputfile) # sorted by descending frequency
		addCommand = "sed -i '1s/^/count\\tword\\n/' " # add labels, do this post hoc so we can sort the file		
		os.system(addCommand + outputfile)
		##df = pandas.read_table(outputfile, sep='\t', encoding='utf-8')
		#df.to_csv(outputfile, encoding='utf-8') #overwrite the file

	elif (sorttype == 'alphabetic'):
		os.system("env LC_ALL=C sort "+tf_path+' > '+outputfile) # sorted alphabetically, suitable for putting into a ZS file         
	os.remove(tf_path)	

	print('Done!')

def countNgrams(inputfile, outputfile, n):
	'''Produces an ngram count for a text file using the ngrams command from Autocorpus'''
	print('Counting the ngrams...')
	ngramsCommand = 'cat '+inputfile+' | /usr/bin/ngrams -n '+str(n)+' > '+outputfile
	subprocess.call(ngramsCommand, shell=True)


def cleanTextFile(inputfile, outputfile, cleaningFunction):
	'''Cleans a plaintext file line by line with the function specified in cleaningFunction'''
	print('Cleaning the plaintext file...')

	tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                      if unicodedata.category(unichr(i)).startswith('P'))
	tbl.pop(ord(u"'")) #remove apostrophe from the list of punctuation

	def cleanLine_BNC(l):
		return remove_punctuation(l.lower(), tbl)
	cleanLineOptions = {'cleanLine_BNC': cleanLine_BNC}


	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	off = codecs.open(outputfile, 'w', encoding='utf-8')	

	for line in iff:		
		off.write(cleanLineOptions[cleaningFunction](line))

	iff.close()
	off.close()

def permuteTextFile(inputfile, outputfile):
	'''creates a permuted ordering of a text'''
	import numpy	
	print('Permuting the plaintext file '+ inputfile)

	print('Retrieving the uniphone and space counts...')
	charcounts = {}	
	lineLengths = []
	spaces = 0 # keep spaces separate

	iff = codecs.open(inputfile, 'r', encoding='utf-8')	
	for line in iff:		
		letters = list(line)
		lineLengths.append(len(letters))
		for letter in letters:
			if letter == u' ':
				spaces += 1
			elif letter == u'\n':
				pass #don't do anything with an end of line character	
			elif letter in charcounts:
				charcounts[letter] += 1

			else:
				charcounts[letter] = 1	

	print('Generating first and last letter of every sentence...')
	letters = numpy.array(charcounts.keys())			
	letterIndices = numpy.array(range(len(letters)), dtype =numpy.uint8)
	counts = numpy.array(charcounts.values(), dtype =numpy.float)			
	probs = counts/numpy.sum(counts)

	
	bookends = []
	for line in range(len(lineLengths)):
		line_bookends = numpy.random.choice(letterIndices,size=2,p=probs)	#this returns a number			
		bookends.append(line_bookends)

		for j in line_bookends:
			counts[j] -= 1.			
		probs = counts/numpy.sum(counts) # recompute the probs	

	print('Generating the center of every sentence...')
	# now add back the spaces and draw from indices 1: n-1, including space; 
	# retreive the first and last letter from bookends	
	letters_withSpace = numpy.array(charcounts.keys() + [u' '])				
	spaceIndex = len(letters_withSpace)-1
	letterIndices_withSpace = range(len(letters_withSpace))		
	counts_withSpaces = numpy.hstack([counts, numpy.float(spaces)])			
	probs_withSpaces = counts_withSpaces/numpy.sum(counts_withSpaces)

	off = codecs.open(outputfile, 'w', encoding='utf-8')	
	for lineIndex in range(len(lineLengths)):
		lineLength = lineLengths[lineIndex]
				
		middle_content = []
		lastLetter = -1
		for index in range(lineLength-2): #omitting the first and the last letter
			newLetter = numpy.random.choice(letterIndices_withSpace, p=probs_withSpaces)
			redrawCount = 0
			while newLetter == spaceIndex and lastLetter == spaceIndex:
				#print('drawing a double space')				
				redrawCount += 1
				if redrawCount > 100:
					print(counts_withSpaces)
					break
				newLetter = numpy.random.choice(letterIndices_withSpace, p=probs_withSpaces)

			middle_content.append(newLetter)	
			lastLetter = newLetter
			counts_withSpaces[newLetter] -=1.			
			# update the probablities	
			probs_withSpaces = counts_withSpaces/numpy.sum(counts_withSpaces)		


		stringToWrite = ''.join([letters_withSpace[x] for x in ([bookends[lineIndex][0]]+middle_content +[bookends[lineIndex][1]])]) 

		off.write(stringToWrite+u'\n')		

	iff.close()
	off.close()
	print('Finished permuting plaintext file.')


def getMeanSurprisal(backwards_zs_path, forwards_txt_path, filter_file, retrieval_file, cutoff, outputfile, language, retrieval_count, dictionary_filter):		
	start_time = time.time()
	'''producing mean surprisal estimates given a backwards n-gram language model and a forwards text file (to be read into a hash table) for order n-1. Produces mean information content (mean log probability, weighted by the frequency of each context)'''

	if os.path.exists(outputfile):
		print('Using the existing lexical surprisal estimate')			
	else:		
		print('Loading the backwards ZS file for order n...')
		backward_zs = ZS(backwards_zs_path, parallelism=0)

		print('Loading the forwards hash table for order n-1...')
		
		bigrams = {}
		f = codecs.open(forwards_txt_path, encoding='utf-8')
		for line in f:
			lineElements = line.split('\t')
			if len(lineElements) > 1:			
				key = lineElements[0]+u' ' 						
				val = int(lineElements[1])
				bigrams[key] = val
			else:
				pdb.set_trace()	

		#!!! be careful with loading text vs. csv files	
		print('Loading retrieval file...')	#use the column 'word', sort desceding by frequency or count
		retrievalDF = readCSVorTxt(retrieval_file)
		if 'word' not in retrievalDF.columns:
			raise ValueError('Retrieval file must have "word" column')
		if ('count' in retrievalDF.columns):
			retrievalDF = retrievalDF.rename(columns={'count': 'retrieval_count'})
		elif ('frequency' in filterDF.columns):
			retrievalDF = retrievalDF.rename(columns={'frequency': 'retrieval_count'})
		else:
			raise ValueError('Filter file must contain "count" or "frequency" column')	
		
		print('Loading filter file...')	#use the column word	
		if filter_file is not None:
			filterDF = readCSVorTxt(filter_file)

			if 'word' not in filterDF.columns:
				raise ValueError('Retrieval file must have "word" column')
			if ('count' in filterDF.columns):
				filterDF = filterDF.rename(columns={'count': 'filter_count'})
			elif ('frequency' in filterDF.columns):
				filterDF = filterDF.rename(columns={'frequency': 'filter_count'})
			else:
				raise ValueError('Filter file must contain "count" or "frequency" column')	

			merged = retrievalDF.merge(filterDF, left_on='word', right_on='word')
		else: 
			merged = retrievalDF

		merged = merged.sort_values(by='retrieval_count', ascending=False)
		
		# Dictionary-based filtering
		merged = filterByDictionary(merged, dictionary_filter, language)
		
		#take the top n items after dictionary-based exclusion
		wordsToRetrieve = merged['word'].head(retrieval_count).tolist()

		print('Retrieving lexical surprisal estimates...')
		surprisalEstimates = [get_mean_surp(bigrams, backward_zs, w, cutoff) for w in wordsToRetrieve]

		df = pandas.DataFrame(surprisalEstimates)
		df.columns = ['word','mean_surprisal_weighted','mean_surprisal_unweighted','frequency','numContexts','retrievalTime']
		df.to_csv(outputfile, index=False, encoding='utf-8')	
		print('Done! Completed file is at '+outputfile+'; elapsed time is '+str(round(time.time()-start_time /  60., 5))+' minutes') 

def filterByDictionary(merged, dictionary_filter, language):		
	if dictionary_filter is None :	
		print('Not limiting words to a spelling dictionary')
		pass
	elif dictionary_filter in ('lowerInDictionary', 'inDictionary'):			
		aspellLang = language

		if aspellLang == 'pt':
			aspellLang = 'pt-BR'

		speller = aspell.Speller(('lang',aspellLang),('encoding','utf-8'))

		merged['aspell_lower'] = [speller.check(x.lower().encode('utf-8')) == 1 for x in merged['word']]

		if dictionary_filter == 'lowerInDictionary':
			if aspellLang == 'de':
				raise ValueError('German must use inDictionary filter setting because all nouns are capitalized')
			print('Limiting to words with lower-case in spelling dictionary')
			#German nouns are capitalized, so need to check upper case
			merged = merged[merged['aspell_lower']] #only take the upper case one
		elif dictionary_filter == 'inDictionary':			
			print('Limiting to words with lower-case or upper-case in spelling dictionary')			
			merged['aspell_upper'] = [speller.check(x.title().encode('utf-8')) == 1 for x in merged['word']]
			#this should be checking if speller.check has x.upper
			merged = merged[merged['aspell_upper'] | merged['aspell_lower']]	
	else:
		raise ValueError('Dictionary specification not recognized. Choose None, "lowerInDictioanry" or "inDictionary"')	
	return(merged)

def get_mean_surp(bigrams_dict,zs_file_backward, word, cutoff):	
	start_time = time.time()	
	total_freq = 0
	surprisal_total = 0
	num_context = 0
	unweightedSurprisal = 0	
	searchTerm = word+u" " #need a trailing space
	print 'Retrieving context probabilities for '+searchTerm	
	for record in zs_file_backward.search(prefix=searchTerm.encode('utf-8')):
		r_split = record.decode("utf-8").split(u"\t")
		ngram = r_split[0].split(u' ')
		#print r_split[0]
		count = int(r_split[1])
		if count >= cutoff:
			total_freq += count
			context =u" ".join(ngram[1:][::-1])+u' '
			num_context += 1
			if context in bigrams_dict:
				total_context_freq = bigrams_dict[context]
				cond_prob = -1 * math.log(count / float(total_context_freq))
				#print cond_prob
				surprisal_total += (count * cond_prob) #this is weighted by the frequency of this context
				unweightedSurprisal +=  cond_prob #this is not
			else:
				print('Missing context: '+ context) 
				#pdb.set_trace()
				#there should not be any missing values			
		else:
			continue	
	stop_time = time.time()
	st = None if total_freq == 0 else surprisal_total / float(total_freq)
	uwst = None if num_context == 0 else unweightedSurprisal / float(num_context)
	return (word, st, uwst, total_freq, num_context, (stop_time-start_time))

def addSublexicalSurprisals(retrieval_file, retrieval_count, model_file,model_count, output_file, column, language, dictionary_filter, smoothing, token_corpus, token_filter, overwriteTokens=False):
	'''get the probability of each word's letter sequence using the set of words in the language
		#first argument is the set of types over which the model will be computed, in this case the 2013 subtitle data
		#second argument is the name of the file to augment. If it doesn't exist, a new file is created
		#third is the kind of model to build
		#number of types in the model specifies how much of the first argument to use, e.g. 5k
		#fifth is the country code, which is used in the call to espeak and/or aspell
	''' 	
	print('Retrieving sublexical surprisal estimates...')

	# Load the model DF
	modelDF = readCSVorTxt(model_file)
	if ('count' in modelDF.columns):
		modelDF = modelDF.rename(columns={'count': 'retrieval_count'})
	elif ('frequency' in lex.columns):
		modelDF = modelDF.rename(columns={'frequency': 'retrieval_count'})
	else:
		raise ValueError('Model file must contain "count" or "frequency" column')	
	modelDF = modelDF.sort_values(by='retrieval_count', ascending=False)	
	modelDF = filterByDictionary(modelDF, dictionary_filter, language).head(model_count)

	# Load the retrieval DF
	retrievalDF = readCSVorTxt(retrieval_file)
	if ('count' in retrievalDF.columns):
		retrievalDF = retrievalDF.rename(columns={'count': 'retrieval_count'})
	elif ('frequency' in lex.columns):
		retrievalDF = retrievalDF.rename(columns={'frequency': 'retrieval_count'})
	else:
		raise ValueError('Retrieval file must contain "count" or "frequency" column')	
	retrievalDF = retrievalDF.sort_values(by='retrieval_count', ascending=False)
	retrievalDF = filterByDictionary(retrievalDF, dictionary_filter, language).head(retrieval_count)

	sublexLMfileDir = os.path.join(os.path.dirname(output_file), column)
	if not os.path.exists(sublexLMfileDir):
		os.makedirs(sublexLMfileDir)

	if column == 'character':				
		retrieval_pm = retrievalDF
		model_pm = modelDF

		retrieval_pm['character'] = [list(x) for x in retrieval_pm['word']]
		model_pm['character'] = [list(x) for x in model_pm['word']]


		LM = trainSublexicalSurprisalModel(model_pm, column, order=5, smoothing=smoothing, smoothOrder=[3,4,5], interpolate=True, sublexlmfiledir = sublexLMfileDir)	
		retrieval_pm[column+'_ss_array']   = [getSublexicalSurprisal(transcription, LM, 5, 'letters', returnSum=False) for transcription in list(retrieval_pm[column])]	

	elif column == "token_character":	
		retrieval_pm = retrievalDF
		retrieval_pm['character'] = [list(x) for x in retrieval_pm['word']]
		retrieval_pm['token_character'] = retrieval_pm['character']

		char_path = os.path.join(sublexLMfileDir, 'token_character.txt')
		if os.path.exists(char_path) and not overwriteTokens:			
			print('Using previously generated token files')
			char_file = char_path

		else:
			print('Generating token files')				

			# should write both character and IPA model from letterize
			char_file = letterize2(
				inputfile = token_corpus,
				outputfile =  char_path,
				filterfile = token_filter,				
				splitwords = True,
				espeak_lang = None,
				phonebreak = '" "',
				par = True,
				espeak_model = None)

		#then train models with these as the input
		char_LM = trainTokenModel(char_file, order=5, outputfile=os.path.join(sublexLMfileDir,'token_character.LM'))
			
		retrieval_pm[column+'_ss_array'] = [getSublexicalSurprisal(transcription, char_LM, 5, 'letters', returnSum=False) for transcription in list(retrieval_pm['character'])]
			#note that the queries are from the character column


	elif column in ('ipa','token_ipa'):			
		
		#get the IPA representation from espeak
		if language == u'en':
			espeak_lang = u'en-US'
		elif language == u'he':
			print 'No Hebrew support for Espeak, returning None for IPA'
			return None
		else:
			espeak_lang = language	
		
		

		print('Retrieving IPA for all words in sample...')
		n = multiprocessing.cpu_count()			
		retrieval_pronunciations = Parallel(n_jobs=n)(delayed(espeak.espeak)(*i) for i in [(espeak_lang, x) for x in retrievalDF['word']])

		retrieval_pdf = pandas.DataFrame(retrieval_pronunciations) #this has a column "ipa"				
		retrieval_pm = retrievalDF.merge(retrieval_pdf, left_on="word", right_on="word")		

		#exclude items where pronunctiation is more than twice as long as the number of characters. This filters out many abbreviations  
		retrieval_pm['nSounds'] = [len(x) for x in retrieval_pm['ipa']]	
		retrieval_pm['suspect'] = retrieval_pm.apply(lambda x: (x['nSounds']/2.) > len(x['word']), axis=1)
		retrieval_pm = retrieval_pm.ix[~retrieval_pm['suspect']]

		if column  == 'ipa':
			print('Building a type-weighted model')
			print('Retrieving IPA for words in model...')


			espeak_results = Parallel(n_jobs=n)(delayed(espeak.espeak)(*i) for i in [(espeak_lang, x) for x in modelDF['word']]) 		
			#vocab['espeak'] = [x['ipa'] for x in espeak_results]
			#model_pronunciations = [espeak.espeak(espeak_lang,x) for x in modelDF['word']]		
			model_pdf = pandas.DataFrame(espeak_results)
			model_pm = modelDF.merge(model_pdf, left_on="word", right_on="word")	

			
			LM = trainSublexicalSurprisalModel(model_pm, column, order=5, smoothing=smoothing, smoothOrder=[3,4,5], interpolate=True, sublexlmfiledir=sublexLMfileDir)	
			retrieval_pm[column+'_ss_array']   = [getSublexicalSurprisal(transcription, LM, 5, 'letters', returnSum=False) for transcription in list(retrieval_pm[column])]

		elif column == 'token_ipa':
			
			# input paths:
			# token_corpus : plaintext corpus from which we want token frequencies
			# token_filter : wordlist for limiting the token corpus

			# output paths
			# sublexLMfileDir

			phone_path = os.path.join(sublexLMfileDir, 'token_ipa.txt')
			if os.path.exists(phone_path) and not overwriteTokens:				
				print('Using previously generated token files')
				phone_file = phone_path

			else:
				print('Generating token files')	

				# should write both character and IPA model from letterize
				phone_file = letterize2(
					inputfile = token_corpus,
					outputfile =  phone_path,
					filterfile = token_filter,					
					splitwords = True,
					espeak_lang = espeak_lang,
					phonebreak = '" "',
					par = True,
					espeak_model = None)

			#then train models with these as the input
			phone_LM = trainTokenModel(phone_file, order=5, outputfile=os.path.join(sublexLMfileDir,'token_ipa.LM'))

			retrieval_pm['token_ipa'] = retrieval_pm['ipa']
			retrieval_pm[column+'_ss_array']   = [getSublexicalSurprisal(transcription, phone_LM, 5, 'letters', returnSum=False) for transcription in list(retrieval_pm['ipa'])]
			#note that the queries are from ipa rather than token_ipa
	
	elif column == 'ortho':
		retrieval_pm = retrievalDF
		retrieval_pm['ortho'] = [list(x) for x in retrieval_pm['word']]
		retrieval_pm[column+'_ss_array'] = [[1]*len(x) for x in retrieval_pm['ortho']]

		#use pm['word']
	elif column == 'sampa':		
		raise ValueError('Out of date procedure for obtaining SAMPA model')	
		retrieval_pm =  retrievalDF
		if not 'sampa' in retrieval_pm.columns:
			print 'Must have SAMPA column to compute sublexical model for SAMPA'
			return None #can't compute SAMPA on the fly
		retrieval_pm['sampa'] = [x.split(' ') for x in retrieval_pm['sampa']]
		LM = trainSublexicalSurprisalModel(modelDF, column, order=5, smoothing=smoothing, smoothOrder=[3,4,5], interpolate=True, sublexlmfiledir= sublexLMfileDir)	
		retrieval_pm[column+'_ss_array']   = [getSublexicalSurprisal(transcription, LM, 5, 'letters', returnSum=False) for transcription in list(retrieval_pm[column])]
	else:
		raise ValueError('Acceptable column types are sampa, character, and ortho')	
	
	
	retrieval_pm[column+'_ss'] = [sum(x) if x is not None else 0 for x in retrieval_pm[column+'_ss_array']]	
	retrieval_pm[column+'_n'] = [len(x) if x is not None else 0 for x in retrieval_pm[column+'_ss_array']]
	
	#add the new results to the output_file and write it out
	if os.path.exists(output_file):
		aug = pandas.read_csv(output_file, encoding='utf-8').dropna()	
		if column in aug.columns:
			#columns already exist in the file, so we want to overwrite it
			retrieval_pm[['word', column, column+'_ss_array', column+'_ss', column+'_n']].to_csv(output_file, index=False, encoding='utf-8')			
		else:				
			aug.merge(retrieval_pm[['word', column, column+'_ss_array', column+'_ss', column+'_n']], left_on="word", right_on="word").to_csv(output_file, index=False, encoding='utf-8')
	else: 
		retrieval_pm[['word', column, column+'_ss_array', column+'_ss', column+'_n']].to_csv(output_file, index=False, encoding='utf-8')			
	print('Done!')

def trainTokenModel(inputfile, order, outputfile):
	''' Train an n-gram language model using a token inventory 

		inputfile: path to a file with one word per line, separated by a space
		order: integer representing the highest order encoded in the language model
		
		outputfile: path to the resulting language model
	'''	
	commandString = 'ngram-count -text '+inputfile+' -order ' + str(order) + ' -lm ' + outputfile

	subprocess.call(commandString, shell=True)
	
	# load the language model and return it
	lm = srilm.LM(outputfile, lower=True)
	return(lm)	

def trainSublexicalSurprisalModel(wordlist_DF, column, order, smoothing, smoothOrder, interpolate, sublexlmfiledir):	
	''' Train an n-gram language model using a list of types 

		wordList_DF: a pandas data DataFrame
		column: the name of the pandas data frame to use 
		order: integer representing the highest order encoded in the language model
		smoothing: Smoothing technique: 'wb' or 'kn'
		smoothOrder: list of integers, indicating which orders to smooth
		interpolate: boolean, indicating whether to use interpolation or not
		sublexlmfiledir: where should the type file and the language model be stored?

	'''

	# ensure that ngram-count is on the path. shouldn't need to do this from the command line	
	#os.environ['PATH'] = os.environ['PATH']+':'+srilmPath
	#generate the relevant filenames
	typeFile = os.path.join(sublexlmfiledir, 'typeFile.txt')
	modelFile = os.path.join(sublexlmfiledir, 'types.LM')

	# write the type inventory to the outfile
	outfile = codecs.open(typeFile, 'w',encoding='utf-8')
	sentences=[u' '.join(transcription) for transcription in wordlist_DF[column]] 
	print >> outfile, '\n'.join(sentences)
	outfile.close()

	# train a model with smoothing on the outfile
	if smoothing is not None:
		discounting = ' '.join([''.join(['-', smoothing,'discount', str(x)]) for x in smoothOrder])
		commandString = 'ngram-count -text '+typeFile+' -order ' + str(order) + ' ' + discounting + (' -interpolate' if interpolate else '') + ' -lm ' + modelFile
	else:	
		commandString = 'ngram-count -text '+typeFile+' -order ' + str(order) + ' -lm ' + modelFile

	subprocess.call(commandString, shell=True)

	# load the language model and return it
	lm = srilm.LM(modelFile, lower=True)
	return(lm)

def getSublexicalSurprisal(targetWord, model, order, method, returnSum):		

	''' Get the sublexical surprisal for a word
		targetWord: type for which surprisal is calculated
		model: pysrilm LM object
		order: specify n of n-gram model. e.g. 1 for unigrams
		method: get probability of sounds or letters. 
				if sounds, input must be a list of phones
		returnSum: if true, return sum of surprisal values
				otherwise, return a list of surprisal values		
	'''
	print  'Getting sublexical surprisal: '+''.join(targetWord).encode('utf-8')	
	if (method == 'sounds'):		
		#throw an error if the variable word is not already a list
		word = ['<s>'] + targetWord + ['</s>'] #append an end symbol
		infoContent = list()	
		raise NotImplementedError	
	elif (method == 'letters'):		
		# if type(targetWord) is not str:
		# 	pdb.set_trace()		

		if(len(targetWord) == 0):
			return(None)
			#proceed to the next one
		else:
			word = ['<s>'] + targetWord + ['</s>'] 
			infoContent = list()

	for phoneIndex in range(len(word)):
		if(phoneIndex - order < 0):
			i = 0 #always want the start to be positive 
		else:
			i = phoneIndex - order + 1 					
		target=word[phoneIndex].encode('utf-8') 				 		
		preceding=[x.encode('utf-8') for x in word[i:phoneIndex][::-1]] #pySRILM wants the text in reverse 		
		phonProb = model.logprob_strings(target,preceding)
		#print('Target: '+target,': preceding: '+' '.join(preceding)+'; prob:'+num2str(10**phonProb,5))
		infoContent.append(-1*phonProb)

	infoContent = infoContent[1:-1] #remove the beginning and end marker									
	if (all ([ x is not None for x in infoContent])):
		if returnSum:
			return(sum(infoContent))
		else:
			return(infoContent)	
	else:
		return(None)

def analyzeSurprisalCorrelations(lexfile, sublexfile):
	'''get correlations and plot the relationship between lexical and sublexical surprisal'''
	lex_DF = pandas.read_csv(lexfile, encoding='utf-8')
	sublex_DF = pandas.read_csv(sublexfile, encoding='utf-8')
	#wordlist_DF = pandas.read_table(wordlist_csv, encoding='utf-8')	

	df_selected = lex_DF.merge(sublex_DF, on='word').sort('frequency', ascending=False).dropna()
	#df_selected = wordlist_DF.merge(df, on='word').sort('frequency', ascending=False)
	df_selected['unigramSurprisal']  = -1 * np.log(df_selected['frequency'] / np.sum(df_selected['frequency']))
		
	trigram_ss = scipy.stats.spearmanr(df_selected['mean_surprisal_weighted'], df_selected['character_ss'])
	trigram_nchar = scipy.stats.spearmanr(df_selected['mean_surprisal_weighted'], df_selected['character_n'])
	unigram_ss = scipy.stats.spearmanr(df_selected['unigramSurprisal'], df_selected['character_ss'])
	unigram_nchar = scipy.stats.spearmanr(df_selected['unigramSurprisal'], df_selected['character_n'])

	print ('number of words in analysis: ' + str(len(df_selected)) + ' types')
	print ("trigram ~ ss:" + str(trigram_ss))
	print ("unigram ~ ss:" + str(unigram_ss))
	print ("trigram ~ nchar:" + str(trigram_nchar))
	print ("unigram ~ nchar:" + str(unigram_nchar))
	
	#df_selected.to_csv(outfile, index=False, encoding='utf-8')

def checkForMissingFiles(directory1, pattern1, directory2, pattern2):
	'''check which files from directory1 are not in directory2'''

	raw_files = glob.glob(os.path.join(directory1,pattern1))
	raw_filenames = [os.path.splitext(os.path.basename(x))[0] for x in raw_files]
	if len(raw_filenames) == 0:
		raise ValueError('No files matching search terms found in first directory')	
	print('Directory 1 contains '+str(len(raw_filenames)) + ' files')
	processed_files = glob.glob(os.path.join(directory2,pattern2))
	processed_filenames = [os.path.splitext(os.path.basename(x))[0] for x in processed_files]
	
	if len(raw_filenames) != len(processed_filenames):
		print('Differing number of raw and processed files')

		missing = []
		[missing.append(file) for file in raw_filenames if file not in processed_filenames]
		warnings.warn(('Missing files'))
		print(missing)		
	else:
		print('Same number of raw and processed files')	
	return (len(processed_filenames) /  (len(raw_filenames) * 1.))

def checkForBinary(command):
	test = os.popen("which "+command).read()
	if test != '':
		print(command +' found at '+test)
	else:
		raise ValueError('binary for '+command +' not found')	


def downloadCorpus(language, order, inputdir, release):
	import httplib2
	from bs4 import BeautifulSoup, SoupStrainer	
	import urllib
	from datetime import datetime

	language_input = language
	corpora_dir = inputdir
	old_cwd = os.getcwd()
	os.chdir(corpora_dir)
	start_time = datetime.now()
	http = httplib2.Http()
	status, response = http.request('http://storage.googleapis.com/books/ngrams/books/datasetsv2.html')
	for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
		if link.has_key('href'):
			url = link['href']
			# IF we match what we want:
			if re.search(order+"gram.+"+release, url):
				# Decode this
				m = re.search(r"googlebooks-([\w\-]+)-(\d+)gram.+"+release,url)
				language, n = m.groups(None)
				# Only download some language
				#set(["eng-us-all", "fre-all", "ger-all", "heb-all", "ita-all", "rus-all", "spa-all", "chi-sim" ])
				if language != language_input: continue
				filename = re.split(r"/", url)[-1] # last item on filename split
				# Make the directory if it does not exist
				if not os.path.exists(language):       os.mkdir(language)
				if not os.path.exists(language+"/"+n): os.mkdir(language+"/"+n)
				if not os.path.exists(language+"/"+n+"/"+filename):
					print "# Downloading %s to %s" % (url, language+"/"+n+"/"+filename)
					urllib.urlretrieve(url, language+"/"+n+"/"+filename )
				else:
					print('File already exists')	
				print "opening url:", url
				site = urllib.urlopen(url)
				meta = site.info()
				print "Content-Length:", meta.getheaders("Content-Length")[0]
				if(os.path.getsize(language+"/"+n+"/"+filename)!= int(meta.getheaders("Content-Length")[0])):
					print("error: "+filename)
				sys.stdout.flush()
	os.chdir(old_cwd)
	print("It took " + str(datetime.now() - start_time))

def downloadCorpusWrapper(corpusSpecification):
	downloadCorpus(corpusSpecification['language'], corpusSpecification['order'],corpusSpecification['inputdir'])

def validateCorpus(corpusSpecification):
	import httplib2
	from bs4 import BeautifulSoup, SoupStrainer	
	import urllib
	from datetime import datetime

	language_input = corpusSpecification['language']
	order = corpusSpecification['order']
	corpora_dir = os.path.join(corpusSpecification['inputdir'], corpusSpecification['corpus'])
	old_cwd = os.getcwd()
	os.chdir(corpora_dir)
	start_time = datetime.now()
	http = httplib2.Http()
	status, response = http.request('http://storage.googleapis.com/books/ngrams/books/datasetsv2.html')
	for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
		if link.has_key('href'):
			url = link['href']
			# IF we match what we want:
			if re.search(order+"gram.+20120701", url):
				# Decode this
				m = re.search(r"googlebooks-([\w\-]+)-(\d+)gram.+",url)
				language, n = m.groups(None)
				# Only download some language
				#set(["eng-us-all", "fre-all", "ger-all", "heb-all", "ita-all", "rus-all", "spa-all", "chi-sim" ])
				if language != language_input: continue
				filename = re.split(r"/", url)[-1] # last item on filename split
				# Make the directory if it does not exist
				if not os.path.exists(language): print("no directory for " + language)
				if not os.path.exists(language+"/"+n): print("no directory for "+language+"/"+n)
				if not os.path.exists(language+"/"+n+"/"+filename): 
					print("no file: "+language+"/"+n+"/"+filename)
					print "# Downloading %s to %s" % (url, language+"/"+n+"/"+filename)
					#urllib.urlretrieve(url, language+"/"+n+"/"+filename )
				site = urllib.urlopen(url)
				meta = site.info()
				print(meta.getheaders("Content-Length")[0])
				if(os.path.getsize(language+"/"+n+"/"+filename)!= int(meta.getheaders("Content-Length")[0])):
					print("error(wrong file size): "+filename)
					print "# Downloading %s to %s" % (url, language+"/"+n+"/"+filename)
					#urllib.urlretrieve(url, language+"/"+n+"/"+filename )
					#gunzip -t # 
					#pigz#
					#gzrecover #
				sys.stdout.flush()
	os.chdir(old_cwd)

def cleanString(string): 
		return(''.join(e for e in string if e.isalpha() or e in ("'") or e.isspace()))	

def cleanUnigramCountFile(inputfile, outputfile, n, language, filterByDictionary):	
	'''filter the unigram count file, and reduce the number of items in it'''	

	df = pandas.read_table(inputfile, encoding='utf-8')	
	df.columns = ['word','count']
	#take some multiple of items to run the filters on
	
	#discard purely numeric items
	df_nonnumeric = df[[type(x) is unicode for x in df['word']]]	

	#discard the <s> string
	df_clean = df_nonnumeric[[x != u'</s>' for x in df_nonnumeric['word']]]

	#delete apostrophes, numbers
	df_clean['word'] = [re.sub(u"’|'|\d",'',x) for x in df_clean['word']]

	#check for any empty strings
	df_clean = df_clean[[x != '' and x is not  None for x in df_clean['word']]]		
	
	df_clean['word'] = [cleanString(x) for x in df_clean['word']] 

	#check whether the upper and lower case is in the dictionary
	aspellLang = language
	if aspellLang == 'pt':
		aspellLang = 'pt-BR'
	speller = aspell.Speller(('lang',aspellLang),('encoding','utf-8'))
	df_clean['aspell_upper'] = [speller.check(x.lower().encode('utf-8')) == 1 for x in df_clean['word']]
	df_clean['aspell_lower'] = [speller.check(x.title().encode('utf-8')) == 1 for x in df_clean['word']]
	
	#Convert anything that can be lower case to lower case
	df_clean['word'][df_clean['aspell_lower']] = [x.lower() for x in df_clean['word'][df_clean['aspell_lower']]]

	if filterByDictionary:
		#check the rejected words
		#df_clean.ix[~df_clean['aspell']]	
		if language == 'de':
			#German nouns are capitalized
			df_clean = df_clean.ix[np.logical_or(df_clean['aspell_lower'],df_clean['aspell_upper'])]		
		else:
			df_clean = df_clean.ix[df_clean['aspell_lower']]		


	to_write = df_clean.drop(['aspell_lower','aspell_upper'], axis=1)
	to_write['word'] = [x.lower() for x in to_write['word']]
	to_write.to_csv(outputfile, sep='\t', index=False, header=False, encoding='utf-8')
	print('Wrote to file: '+outputfile)


def fixPunctuation(inputfile, outputfile, order):
	'''remove symbols except apostrophes and replace right quotation mark with apostrophe'''
	bufsize = 10000000
	print('Fixing the punctuation...')
	iff = codecs.open(inputfile, 'r', encoding='utf-8')
	off = codecs.open(outputfile, 'w', encoding='utf-8')	
	firstLine ='\n' #handle any lines that are blank at the beginning of the text
	#need to confirm that there is anything in the file
	while firstLine == '\n' or firstLine == '':
		firstLine = iff.readline()
	
	rows =[]
	lineSplit = firstLine.split('\t')
	ngram = lineSplit[0]	
	ngram_split = [x for x in ngram.split(' ') if x != u"'" and x != u'']
	count = int(lineSplit[1])
	
	if len(lineSplit) == 4:		
		print('3 tab-delineated columns, assuming first is the ngram, second is the token and the third the context count')								
		ncols = 4	
		context_count = int(lineSplit[2])
		if len(ngram_split) == order:
			rows.append('\t'.join([ngram, str(count), str(context_count)]))	
	elif len(lineSplit) == 2:
		print('2 tab-delineated columns, assuming first is the ngram, second is the token count')	
		ncols = 2		
		if len(ngram_split) == order:
			rows.append('\t'.join([ngram, str(count)]))	
	
	for c,l in enumerate(iff):
		line = l.split('\t')
		if len(line) != ncols:			
			print 'Mismatch in line length and ncols, line was '+line
			continue
		ngram = line[0]
		ngram_split = [x for x in ngram.split(' ') if x != u"'" and x != u'']
		if len(ngram_split) != order:
			continue
		count = int(line[1])
		if ncols == 4: 
			context_count = int(line[2])
		
		#fix the ngram
		ngram = ngram.replace(u'’',u"'")
		ngram = ''.join(e for e in ngram if e.isalpha() or e in (u"'") or e.isspace())
		
		ngram_split = [x for x in ngram.split(' ') if x != u"'" and x != u'']
		if len(ngram_split) != order:
			continue #there aren't N items after cleaning; exclude from the output

		if ncols == 4:
			rows.append('\t'.join([ngram, str(count), str(context_count)]))
		elif ncols == 2:	
			rows.append('\t'.join([ngram, str(count)]))
			
		if c % bufsize == 0:#refresh the buffer	
			off.write('\n'.join(rows)+'\n')
			rows =[] 
		
	off.write('\n'.join(rows)+'\n')	#catch any records since the last buffered write						 	
	iff.close()
	off.close()
	print('Finished fixing the punctuation, output in file '+str(outputfile))

def utfify(unicode):
	'''take a perfectly good string and pepper it with unicode	in an attempt to break our architecture'''
	remap = {u's':u'š',u'e':u'ë',u'a':u'æ',u'z':u'ž',u'y':u'ÿ',u'c':u'ç',u'n':u'ñ'}
	return(u''.join([remap[x] if x in remap.keys() else x for x in list(unicode)]))


def letterize2(inputfile, outputfile, filterfile, splitwords, espeak_lang, phonebreak, par, espeak_model):
	'''take textfile, split by words, and output a list of letters or phones (phones if espeak_lang is not None) separated by phonebreak. This can then be used as input to SRILM for various models'''

	if espeak_lang == 'None': # make sure None is not treated as a string
		espeak_lang = None

	if (espeak_lang is not None) and (espeak_model is None):
		print('building an espeak model...')
		
		# vocab is produced in the OPUS stack, e.g. OPUS/main.py 	
		vocab = pandas.read_table(filterfile, skiprows=2, header=None, encoding='utf-8') 
		vocab.columns = ['word','count']
		#vocab['espeak'] = [espeak.espeak(espeak_lang,x)['ipa'] for x in vocab['word']]

		print('Retrieving espeak transcriptions...')
		n = multiprocessing.cpu_count()
		espeak_results = Parallel(n_jobs=n)(delayed(espeak.espeak)(*i) for i in [(espeak_lang, x) for x in vocab['word']]) 		
		vocab['espeak'] = [x['ipa'] for x in espeak_results]

		espeak_model = dict(zip(vocab['word'], vocab['espeak']))
		print('Espeak model ready!')		

	if par:
		print 'Calling parallelized version of letterize'
		arguments = {
			'inputfile':inputfile,
			'outputfile':outputfile,
			'filterfile': filterfile,			
			'splitwords':splitwords,
			'espeak_lang': espeak_lang,
			'phonebreak':phonebreak,
			'espeak_model': espeak_model}
		return(embpar(letterize2, arguments))
	else:	
		print 'Executing single-thread version of letterize'
				
		input_f = codecs.open(inputfile, 'r', encoding='utf-8')		
		output_f = codecs.open(outputfile, 'w', encoding='utf-8')		

		phonebreak = phonebreak.replace('"','').replace("'",'') #get rid of quotes, if present


		for c,l in enumerate(input_f):
			if l == '\n':
				pass
			else:				
				if splitwords: #output each word on a separate line		
					words = l.replace('\n','').lower().split(' ')					
					for word in words:
						if espeak_lang is None:
							output_f.write(phonebreak.join(list(word))+'\n')
						else:
							if word in espeak_model: 
								output_f.write(phonebreak.join(espeak_model[word])+'\n')
				else: #all on a single line
					if espeak_lang is None:
						output_f.write(phonebreak.join(list(l.replace(' ','')))+'\n')		
					else:
						words = l.replace('\n','').lower().split(' ')
						translatedWords = []
						for word in words:							
							if word in espeak_model: #check if that token is cached already
								translatedWords.append(phonebreak.join(espeak_model[word]))	
							output_f.write(u' '.join(translatedWords)+'\n')	

		input_f.close()		
		output_f.close()
		return(outputfile)		


def letterize(inputfile, outputfile, splitwords, espeak_lang, phonebreak, par,):
	'''take textfile, split by words, and output a list of letters or phones (phones if espeak_lang is not None) separated by phonebreak. This can then be used as input to SRILM for various models'''
		#!!! a more efficient way to do the ipa translation would be to make a hashmap of the whole dictionary


	if par:
		print 'Calling parallelized version of letterize'
		arguments = {'inputfile':inputfile,'outputfile': outputfile, 'splitwords':splitwords, 'espeak_lang': espeak_lang, 'phonebreak':phonebreak}
		embpar(letterize, arguments)
	else:	
		print 'Executing single-thread version of letterize'
		iff = codecs.open(inputfile, 'r', encoding='utf-8')
		off = codecs.open(outputfile, 'w', encoding='utf-8')	

		phonebreak = phonebreak[1:-1] #get rid of quotes

		if espeak_lang == 'None':
			espeak_lang = None

		if espeak_lang:
			#make a cached for the frequently used tokens
			espeak_cache = {}

		for c,l in enumerate(iff):
			if l == '\n':
				pass
			else:				
				if splitwords: #output each word on a separate line					
					words = l.split(' ')
					for word in words:
						if not espeak_lang:
							off.write(phonebreak.join(list(word))+'\n')
						else:								
							#translate to espeak
							if word in espeak_cache: #check if that token is cached already
								lexItem =  espeak_cache[word]	
							else:		
								lexItem = espeak.espeak(espeak_lang,word)['ipa']						
								espeak_cache[word] = lexItem
							off.write(phonebreak.join(lexItem)+'\n')
				else: 
					if not espeak_lang: #all on a single line
						off.write(phonebreak.join(list(l.replace(' ','')))+'\n')		
					else:
						words = l.split(' ')
						translatedWords = []
						for word in words:							
							if word in espeak_cache: #check if that token is cached already
								lexItem = espeak_cache[word]								
								translatedWords.append(phonebreak.join(lexItem))	
							else:		
								lexItem = espeak.espeak(espeak_lang,word)['ipa']						
								espeak_cache[word] = lexItem
								translatedWords.append(phonebreak.join(lexItem))			
						off.write(u' '.join(translatedWords)+'\n')	

		iff.close()		
		off.close()
		return(outputfile)	


def filterByWordList(inputfile, outputfile, loweronly, vocabfile,n, par):
	'''take a textfile, split by words, and check if each word is in the provided vocabfile'''
	if(par):
		print 'Calling parallelized version of filterByWordList'
		arguments = {'inputfile':inputfile,'outputfile': outputfile, 'loweronly':loweronly, 'vocabfile':vocabfile, 'n':n}
		embpar(filterByWordList, arguments)
	else:		
		print 'Executing single-thread version of filterByWordList'
		def filterWords(l, loweronly, vocab):		
			if loweronly:
				words = l.replace('\n','').split(' ')
				return(u' '.join([word for word in words if word in vocab]))
			else: 	
				words = l.replace('\n','').lower().split(' ')
				return(u' '.join([word for word in words if word in vocab]))

		vocab = set(pandas.read_table(vocabfile, encoding='utf-8', sep='\t')['word'][0:n])

		iff = codecs.open(inputfile, 'r')
		off = codecs.open(outputfile, 'w', encoding='utf-8')	

		bufsize = 100000	
		lineStore = []

		for c,l in enumerate(iff):
			lineStore.append(l)
			if c % bufsize == 0:#refresh the buffer	
				rows = [filterWords(l,loweronly,vocab) for l in lineStore]
				off.write('\n'.join(rows)+'\n')
				lineStore =[] 
				print 'Processed '+os.path.basename(inputfile)+' through line '+ str(c)
		rows = [filterWords(l,loweronly,vocab) for l in lineStore]	
		off.write('\n'.join(rows)+'\n')

		iff.close()		
		off.close()
		return(outputfile)		

def par_filterByWordList(idc):
	filterByWordList(idc['inputfile'], idc['outputfile'], idc['loweronly'], idc['vocabfile'],idc['n'], par=False)

def par_letterize(idc):
	letterize(idc['inputfile'], idc['outputfile'], idc['splitwords'], idc['espeak_lang'], idc['phonebreak'], par=False)

def par_letterize2(idc):
	return(letterize2(idc['inputfile'], idc['outputfile'], idc['filterfile'], idc['splitwords'], idc['espeak_lang'], idc['phonebreak'], False, idc['espeak_model']))	


functionMappings = {
	'filterByWordList' : par_filterByWordList,
	'letterize' : par_letterize,
	'letterize2' : par_letterize2
}

def embpar(functionName, arguments):
	#dict wrappers for functions that can be called with embpar

	def file_len(fname):
	    with open(fname) as f:
	        for i, l in enumerate(f):
	            pass
	    return i + 1

	def split_seq(numItems, numRanges):
		newseq = []
		splitsize = 1.0/numRanges*numItems
		for i in range(numRanges):
			newseq.append((int(round(i*splitsize)),int(round((i+1)*splitsize))))
		return newseq

	def splitfile(inputfile, n):
		'''divide inputfile into n approximately equal-sized parts.'''			
		fileLength = file_len(arguments['inputfile'])
		lineRanges = split_seq(fileLength, n)
		rangeStarts = set([x[0] for x in lineRanges])	

		iff = io.open(arguments['inputfile'], 'r',encoding='utf-8')
		filenames = []
		
		for c,l in enumerate(iff):			
			if c in rangeStarts: #switch the output file
				filename = arguments['inputfile']+'-'+str(c)
				filenames.append(filename)
				off = io.open(filename, 'w', encoding='utf-8')	
			off.write(l)	
		off.close()	
		return(filenames)

	n = multiprocessing.cpu_count()	
	print 'Splitting file: '+arguments['inputfile']
	subfiles = splitfile(file_len(arguments['inputfile']), n)	
	#subfiles = glob.glob(os.path.join(os.path.dirname(arguments['inputfile']),'*.txt-*'))

	#get the string of the function name
	if functionName.__name__ is 'filterByWordList':
		#build the inputs
		print 'Building inputs for parallelization'
		inputs = [{'inputfile':subfiles[x],
					'outputfile':subfiles[x]+'_out',
					'loweronly': arguments['loweronly'],
					'vocabfile': arguments['vocabfile'],
					'n': arguments['n']} for x in range(0,n)]	
	elif functionName.__name__ in ('letterize','letterize2'):				
		inputs = [{'inputfile':subfiles[x],
					'outputfile':subfiles[x]+'_out',
					'filterfile': arguments['filterfile'],
					'splitwords':arguments['splitwords'],
					'espeak_lang':arguments['espeak_lang'],
					'phonebreak':arguments['phonebreak'],
					'espeak_model':arguments['espeak_model'],
					} for x in range(0,n)]

	print 'Starting parallelized execution...'					
	resultfiles = Parallel(n_jobs=n)(delayed(functionMappings[functionName.__name__])(i) for i in inputs)  
	#!!! resultfiles above is not giving back the appropriate filenames
	#resultfiles = glob.glob(os.path.join(os.path.dirname(arguments['inputfile']),'*_out'))

	print('Combining files from parallelization...')
	combineFiles(os.path.dirname(arguments['inputfile']), '*_out', arguments['outputfile'])	
	print('Deleting temporary files from parallelization...')

	[os.remove(file) for file in subfiles]	
	[os.remove(file) for file in resultfiles]
	return(arguments['outputfile'])
	