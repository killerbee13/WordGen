import sys
import yaml
import random
import time
import re
import decimal
import gmpy2
import optparse
import copy

from collections import deque
from string import *


# sep = re.compile()

class Path:
	"""A path to a word."""
	def __next__(self, depth=None):
		if depth:
			self._depth = depth
		# Attempt to increment rightmost-deepest, check for successors
		q = self.asList()
		# (self._branch,len(self._data[self._root]))
		for n,d in reversed(q):
			if len(n._data[n._root]) > 1:
				j = n._branch + 1
				if j == len(n._data[n._root]):
					continue
				while not n._data[n._root][j].get("freq",decimal.Decimal("1")):
					j = n + 1
					if j == len(n._data[n._root]):
						continue
				t = n._data[n._root][j]
				if t.get("freq",decimal.Decimal("1")):
					L = [{"freq":0}] * n._branch
					L.append(t)
					tmp = chooseFrom(n._data, L, self._depth-d+1)
					print(formatWord(applyRE(n._data, tmp), {
						"ipa":True,
					}))
					n.__init__(n._data, n._root, tmp["path"])
					return self
		raise StopIteration
	def asList(self):
		"""Generates a heap-like (effectively). Used for breadth-first traversal"""
		q = list([(self,0)])
		i = 0
		while i < len(q):
			# print(q[i])
			n,d = q[i]
			q.extend([(c,d-1) for c in n._children])
			i = i + 1
		return q
	def iter(self):
		return self
	def getWord(self):
		"""return word corresponding to self suitably for formatWord()"""
		# print('get:  '+self._root)
		Node = self._data[self._root][self._branch]
		sumFreq = sum([decimal.Decimal(decimal.Decimal(x.get("freq",decimal.Decimal('1')))) for x in self._data[self._root]])
		SNode = {"val": Node.get("val",""), "ipa": Node.get("ipa",""), "freq":(decimal.Decimal(Node.get("freq",decimal.Decimal('1')))/sumFreq)}
		rets = {"val":"", "ipa":"", "freq":decimal.Decimal('1')}
		# print(self._root+'['+str(self._branch)+'] SNode'+str(SNode))
		for i,s in enumerate(Formatter().parse(SNode["val"])):
			#Reference
			if s[1]:
				#Generate subword from subpath
				# print('s'+str(s))
				# print(i)
				tmp = Path(self._data, s[1], self._children[i]).getWord()
				
				rets["val"] = rets["val"] + s[0] + tmp["val"]
				rets["freq"] = rets["freq"]*tmp["freq"]
				if s[0]:
					#If reference+literal text, insert 
					rets["ipa"] = rets["ipa"] + SNode["ipa"] + tmp["ipa"]
				else:
					rets["ipa"] = rets["ipa"] + tmp["ipa"]
			#No reference, only literal text
			else:
				rets["val"] = rets["val"] + s[0]
				rets["ipa"] = rets["ipa"] + SNode["ipa"]
		return rets
	def _str(self):
		"""print Path as string"""
		def recurse(self):
			ret = gmpy2.mpz(self._branch).digits(62)
			for a in self._children:
				ret = ret + '[' + (recurse(a)) + ']'
			return ret
		return '+' + recurse(self)
		#return root + '+' + recurse(self)
	def _list(self):
		"""Create nested-list from Path"""
		l = [self._branch]
		l.extend([c._list() for c in self._children])
		return l
	def __init__(self, Data, root, path):
		"""build Path from string or list"""
		self._data = Data
		self._root = root
		self._depth = -16
		# print('root: '+self._root)
		if isinstance(path, list):
			self._branch = path[0]
			self._children = []
			# Extract root for each child Path
			#for c in path[1:]:
			for i,tnode in enumerate(Formatter().parse(Data[root][self._branch].get("val",""))):
				# print('tnode'+str(tnode))
				if tnode[1]:
					self._children.append(Path(Data, tnode[1], path[i+1]))
			
		elif isinstance(path, str):
			path = readPath(path)
			self._branch = path[0]
			self._children = []
			# Extract root for each child Path
			#for c in path[1:]:
			for i,tnode in enumerate(Formatter().parse(Data[root][self._branch].get("val",""))):
				if tnode[1]:
					self._children.append(Path(Data, tnode[1], path[i+1]))
			
		elif isinstance(path, Path):
			if self._root != path._root:
				raise ValueError
			self._branch = path._branch
			self._children = path._children
		else:
			self._branch = 0
			self._children = []
		# print('Init: '+root+'['+str(self._branch)+'] '+str([c._str() for c in self._children]))

def chooseFrom(Data, list, depth=-16):
	"""Select a random value from the list, recursing on references"""
	list = [{"val": x.get("val",""), "ipa": x.get("ipa",""), "freq":decimal.Decimal(x.get("freq",decimal.Decimal('1')))} for x in list]
	listSum = sum([x["freq"] for x in list])
	a = decimal.Decimal(random.uniform(0,float(listSum)))
	stop = 0
	Rets = []
	# This needs no normalization because values are never directly compared.
	for i,c in enumerate([x["freq"] for x in list]):
		a -= c
		if a <= 0:
			stop = i
			break;
	
	if depth < 0:
		# 1+ elements are strings and 1+ elements are references to arrays
		rets = {"val": "", "ipa": "", "freq": decimal.Decimal(1)/listSum}
		#If val is empty, insert ipa and bail
		if not list[stop]["val"]:
			rets["ipa"] = list[stop]["ipa"]
			return {"val": "", "ipa": list[stop]["ipa"], "path": [stop], "freq":list[stop]["freq"]/listSum}
		#Determine which is a string and which is a reference
		else:
			rets["path"] = [stop]
			for s in Formatter().parse(list[stop]["val"]):
				#Recurse on reference and insert results into string
				if s[1]:
					node = copy.deepcopy(Data[s[1]])
					if s[2]:
						flist = re.split('[^0-9.]+',s[2])
						nstr = s[1]
						for i in range(min(len(flist),len(node))):
							d = decimal.Decimal(flist[i])
							node[i]['freq'] = d
							nstr += ','+str(d)
						Data[nstr] = node
						# flist = [decimal.Decimal(f) for f in re.split('[^0-9.]+',s[2])]
						# for i in range(len(flist)):
							# node[i][freq] = flist[i]
					# Throws a KeyError on invalid reference. Not caught because
						# the Python default error message is good enough and there's
						# nothing for the code to do with an error.
					#Fill reference
					tmp = chooseFrom(Data, node, depth+1)
					
					rets["val"] = rets["val"] + s[0] + tmp["val"]
					rets["freq"] = rets["freq"]*tmp["freq"]
					rets["path"].append(tmp["path"])
					if s[0]:
						#If reference+literal text, insert 
						rets["ipa"] = rets["ipa"] + list[stop]["ipa"] + tmp["ipa"]
					else:
						rets["ipa"] = rets["ipa"] + tmp["ipa"]
				#No reference, only literal text
				else:
					rets["val"] = rets["val"] + s[0]
					rets["ipa"] = rets["ipa"] + list[stop]["ipa"]
			return {"val": rets["val"], "ipa": rets["ipa"], "path": rets["path"], "freq":rets["freq"]}
	else:
		#Recursion depth reached
		print("wordgen.py: recursion depth reached", file=sys.stderr)
		#return list[stop]
		return {"val": list[stop]["val"], "ipa": list[stop]["ipa"], "path": [stop], "freq":list[stop]["freq"]/listSum}

def makeWords(Data, n, root, depth_limit=-16, keepHistory=False, KHSep="→"):
	"""Generate a list of n random descendants of {root}"""
	# For very complex data files, the depth limit may need to be increased.
		# 16 should handle up to ~6-8 syllables for any sensible language
		# If you see {Syllable} or the like in the output, either you have a
		# reference loop or you need to increase this.
	# Assuming a roughly tail-recursive file:
	# num_syllables ~= depth_limit - (1 + syllable_complexity)
		# where syllable_complexity (~4.5 for Cūrórayn) is the average number
		# of recursions between a {Syllable} node and an output string
		# The constant 1 is for the root node,
	# The theoretical maximum is node_width^depth_limit which is obviously much
		# greater, so "wide" data files are able to produce much more output.
			# See recursive.yml for a simple example of this with node_width=2
		# The limit is mostly there to avoid loops, though, so this is actually
		# alright. Increase it or decrease it if needed.
	
	# TL;DR: Computation is dependent on total expansions, which is less
		# than (is bounded by) exponential in depth_limit.
	for x in range(n):
		word = chooseFrom(Data, Data[root], depth_limit)
		yield applyRE(Data, word, keepHistory, KHSep)
	return

def filterRE(RE):
	"""Processes regex from file for use.
	Does not sanitize RE."""
	return RE.translate(str.maketrans({'$':'(?=\15)'}))+'$'
	# return RE.translate({'$':'(?!\15)'})+'$'

def applyRE(Data, word, keepHistory=False, KHSep="→"):
	"""Applies regular expressions in Data to word."""
	if "replacement" in Data:
		for stage in Data["replacement"]:
			for rule in stage:
				# Produces approximately a 40% speedup.
				rule["c"] = re.compile(filterRE(rule["m"]))
			cline = ''
			for c in word["val"]+'\15':
				cline += c
				for rule in stage:
					# Determine if rule-match matches, then replace
					cline = rule["c"].sub(rule["r"], cline)
			if keepHistory:
				if word["val"] != cline:
					word["val"] = word["val"] + KHSep + cline
			else:
				word["val"] = cline
		word["val"] = word["val"].translate(str.maketrans('','','\15'))
	if "replaceIPA" in Data:
		for stage in Data["replaceIPA"]:
			for rule in stage:
				# Produces approximately a 40% speedup.
				rule["c"] = re.compile(filterRE(rule["m"]))
			cline = ''
			for c in word["ipa"]+'\15':
				cline += c
				for rule in stage:
					# Determine if rule-match matches, then replace
					cline = rule["c"].sub(rule["r"], cline)
			if keepHistory:
				if word["ipa"] != cline:
					word["ipa"] = word["ipa"] + KHSep + cline
			else:
				word["ipa"] = cline
		word["ipa"] = word["ipa"].translate(str.maketrans('','','\15'))
	return word

def listAll(Data, node, opts = {
		"ipa":True,
		"HTML":False,
		"path":False,
		"depth":-16,
		"keepHistory":False,
		"keepHistorySep":"→",
		"ignoreZeros":True,
		"freq":True,
	}):
	'''Traverse all descendants of node (base)'''
	def listWords(Data, node, depth, opts, path=[], flist=None):
		pass
	def nextPath(Data, node, path):
		
		return path
	# tmpbuf = []
	ret = DFSPrint(listAllR(
		Data, node, opts["depth"], opts["ignoreZeros"]
	))
	for word in ret:
		yield formatWord(
			applyRE(Data, {
				"val":word[0],
				"ipa":word[1],
				# DFSPrint doesn't work with paths
				"path":[0],
				"freq":word[2]}), opts)
		# newword = applyRE(Data, {"val":word[0], "ipa":word[1]})
		# word = (newword["val"], newword["ipa"], word[2])
		# tmpbuf.append(word[0]+' :\t'+word[1]+'\t'+str(word[2]))
		time.sleep(0.0001)
	#return '\n'.join(tmpbuf)


def listAllR(Data, node, depth, ignoreZeros, path=[], flist=None):
	'''Implementation of listAll. Do not call.'''
	if node in path:
		return {"t": 'V', "node": node}
	elif depth < 0:
		path.append(node)
		list = []
		if not flist:
			flist = [x.get("freq",decimal.Decimal('1')) for x in Data[node]]
		listSum = sum(flist)
		# print(str(node)+": "+str(listSum)+" = "+" + ".join([str(x) for x in flist]))
		if ignoreZeros:
			for i in range(len(Data[node])):
				if i < len(flist):
					# Copy Data[node][i] so that Data is not altered
					N = dict(Data[node][i])
					N["freq"] = flist[i]
					if flist[i]:
						list.append(N)
				else:
					if Data[node][i].get("freq"):
						list.append(Data[node][i])
		else:
			list = Data[node]
			
		matches = []
		for child in list:
			# 1+ elements are strings and 1+ elements are references to arrays
			#Determine which is a string and which is a reference
			matches.append({
				"t": 'A',
				"freq": child.get("freq",decimal.Decimal('1'))/listSum,
				"Acontents": []
			})
			#If no val, insert IPA anyway
			if not child.get("val",""):
				matches[-1]["Acontents"].append({"t": 'L', "val": '', "ipa": child.get("ipa","")})
			else:
				for s in Formatter().parse(child["val"]):
					#Recurse on reference and insert results into string
					if s[1]:
						nstr = s[1]
						node = Data[s[1]]
						if s[2]:
							flist = re.split('[^0-9.]+',s[2])
							for i in range(min(len(flist),len(node))):
								d = decimal.Decimal(flist[i])
								node[i]['freq'] = d
								# Flist.append(d)
								nstr += ','+str(d)
							if not nstr in Data:
								Data[nstr] = node
						else:
							flist = None
						# Throws a KeyError on invalid reference. Not caught because
							# the Python default error message is good enough and
							# there's nothing for the code to do with an error.
						#Fill reference
						tmp = listAllR(Data, nstr, depth+1, ignoreZeros, path, None)
						if s[0]:
							#If reference+literal text, insert 
							matches[-1]["Acontents"].append({"t": 'L', "val": s[0], "ipa": child.get("ipa","")})
							matches[-1]["Acontents"].append(tmp)
						else:
							matches[-1]["Acontents"].append(tmp)
					#No reference, only literal text
					else:
							matches[-1]["Acontents"].append({"t": 'L', "val": s[0], "ipa": child.get("ipa","")})
		#path.pop()
		return {"t": 'N', "node": node, "sum":listSum, "Ncontents": matches}
	else:
		#Recursion depth reached
		print("wordgen.py: recursion depth reached", file=sys.stderr)
		return {"t": 'T', "node": node, "raw": Data[node]}
	

def DFSPrint(Node, freq=1):
	'''Generate list of words suitable for printing from tree structure.'''
	def f_A(Node, freq): # Main case
		buf1 = [("","", 1)]
		for n in Node["Acontents"]:
			tfreq = freq*Node["freq"]
			buf2 = DFSPrint(n, freq)
			# print('n: '+str(Node))
			# print('1: '+str(buf1))
			# print('2: '+str(buf2))
			buf3 = []
			for i in buf1:
				for j in buf2:
					# print('---\ni: '+str(i)+'\nj:'+str(j)+'\n---')
					buf3.append((i[0] + j[0], i[1] + j[1], i[2] * j[2]))
			# print('3: '+str(buf3))
			buf1 = buf3
		return buf1
	def f_N(Node, freq): # Simply iterate and recurse
		ret = []
		# N will always contain As
		for n in Node["Ncontents"]:
			ret.extend(DFSPrint(n, freq*n["freq"]))
		return ret
	def f_L(Node, freq): #Leaf
		#print('L: '+str(path))
		return [(Node["val"], Node["ipa"], freq)]
	def f_V(Node, freq): # Turn into reference
		return [("{"+Node["node"]+"}", "{"+Node["node"]+"}", freq)]
	def f_T(Node, freq): # Truncation -- pretend it's L but different
		#print('T: '+str(freq))
		return [("{"+Node["node"]+"}", "{"+Node["node"]+"}", freq)]
	switch = {
		'A': f_A,
		'N': f_N, 
		'L': f_L,
		'V': f_V,
		'T': f_T
	}
	return switch[Node['t']](Node, freq)

def formatWord(word, opts, formatStr=None):
	'''Print words'''
	if formatStr:
		ipa = ""
		path = ""
		freq = ""
		if opts.get("ipa",False):
			ipa = word["ipa"]
		if opts.get("path",False):
			path = printPath(word["path"])
		if opts.get("freq",False):
			freq = word["freq"]
		return formatStr.format(val=word["val"], ipa=ipa, path=path, freq=freq)
	else:
		opts = { "ipa":  opts.get("ipa",  False),
					"path": opts.get("path", False),
					"HTML": opts.get("HTML", False),
					"freq": opts.get("freq", False),
				}
		return formatWord(word, opts, 
		[	"{val}",																	# -
			"{val}: \t{ipa}",														# -p
			"{val} \t{path}",														# -P
			"{val}: \t{ipa} \t{path}",											# -pP
			"<tr><td>{val}</td></tr>",											# -H
			"<tr><td>{val}</td><td>{ipa}</td></tr>",						# -pH
			"<tr><td>{val}</td><td>{path}</td></tr>",						# -PH
			"<tr><td>{val}</td><td>{ipa}</td><td>{path}</td></tr>",	# -pPH
			"{val} \t{freq}",														# -f
			"{val}: \t{ipa} \t{freq}",											# -pf
			"{val} \t{path} \t{freq}",											# -Pf
			"{val}: \t{ipa} \t{path} \t{freq}",								# -pPf
			"<tr><td>{val}</td><td>{freq}</td></tr>",						# -Hf
			"<tr><td>{val}</td><td>{ipa}</td><td>{freq}</td></tr>",	# -pHf
			"<tr><td>{val}</td><td>{path}</td><td>{freq}</td></tr>",	# -PHf
																						# -pPHf
			"<tr><td>{val}</td><td>{ipa}</td><td>{path}</td><td>{freq}</td></tr>",
		][int(opts.get("ipa",False))
			+int(opts.get("path",False))*2
			+int(opts.get("HTML",False))*4
			+int(opts.get("freq",False))*8
		])

def printPath(path):
	def recurse(path):
		ret = gmpy2.mpz(path[0]).digits(62)
		for a in path[1:]:
			ret = ret + '[' + (recurse(a)) + ']'
		return ret
	return '+' + recurse(path)

def readPath(pathStr):
	# Simple token separator function - recognizes + [ ] and alphanumerics
	def tokensOf(pathStr):
		ret = ""
		inInt = False
		for c in pathStr:
			if c in "[]":
				inInt = False
				if ret:
					yield ret
				ret = c
			elif c in "+":
				pass
			elif c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz":
				# Ints are in base 62
				# Int tokens can be multiple characters
				if inInt:
					ret = ret + c
				else:
					inInt = True
					if ret:
						yield ret
					ret = c
			else:
				# This shouldn't be hit -- raise ValueError?
				pass
		return ret
	def constructPath(tokens, i=0):
		ret = []
		while i < len(tokens):
			if tokens[i] == "[":
				tret,ti = constructPath(tokens, i+1)
				ret.append(tret)
				if (ti == len(tokens)):
					pass
					#raise ValueError("Unterminated subpath", i, str(tokens), str(tokens[i:]))
				i = ti
			elif tokens[i] == "]":
				return ret,i
			else:
				ret.append(int(gmpy2.mpz(tokens[i], 62)))
			i += 1
		return ret,i
	try:
		return constructPath(list(tokensOf(pathStr)))[0]
	except ValueError as err:
		# In case of error, report the full path in addition to the subpath
		err.args = err.args + (pathStr, )
		raise

def followPath(Data, node, path):
	# print(node)
	#root = [{"val": x["val"], "ipa": x.get("ipa",""), "freq":decimal.Decimal(x.get("freq",decimal.Decimal('1')))} for x in Data[node] if x.get("freq",decimal.Decimal('1'))]
	sumFreq = sum([decimal.Decimal(decimal.Decimal(x.get("freq",decimal.Decimal('1')))) for x in Data[node]])
	SNode = {"val": Data[node][path[0]].get("val",""), "ipa": Data[node][path[0]].get("ipa",""), "freq":(decimal.Decimal(Data[node][path[0]].get("freq",decimal.Decimal('1')))/sumFreq)}
	rets = {"val":"", "ipa":"", "freq":decimal.Decimal('1')}
	#print(SNode)
	for i,s in enumerate(Formatter().parse(SNode["val"])):
		#Recurse on reference and insert results into string
		if s[1]:
			# Throws a KeyError on invalid reference. Not caught because
				# the Python default error message is good enough and there's
				# nothing for the code to do with an error.
			#Fill reference
			tmp = followPath(Data, s[1], path[i+1])
			
			rets["val"] = rets["val"] + s[0] + tmp["val"]
			rets["freq"] = rets["freq"]*tmp["freq"]
			if s[0]:
				#If reference+literal text, insert 
				rets["ipa"] = rets["ipa"] + SNode["ipa"] + tmp["ipa"]
			else:
				rets["ipa"] = rets["ipa"] + tmp["ipa"]
		#No reference, only literal text
		else:
			rets["val"] = rets["val"] + s[0]
			rets["ipa"] = rets["ipa"] + SNode["ipa"]
	return rets
		


#if 2 < len(sys.argv) < 7:
	#sys.argv.extend([0]*(7-len(sys.argv)))

# #Print $3 descendants of $2 to depth $6 from datafile $1 with mode flags $4 and $5
# printWords(makeWords(yaml.safe_load(open(sys.argv[1],'r', encoding="utf8")), int(sys.argv[3]), sys.argv[2], int(sys.argv[6])), int(sys.argv[4]), int(sys.argv[5]))
# #if not int(sys.argv[3]):
	# #print(yaml.dump(listAllR(yaml.safe_load(open(sys.argv[1],'r', encoding="utf8")), sys.argv[2], int(sys.argv[6]), True, [])))
	# print(listAll(yaml.safe_load(open(sys.argv[1],'r', encoding="utf8")), sys.argv[2], int(sys.argv[6])))

def main():
	# class Dec(decimal.Decimal, yaml.YAMLObject):
		# yaml_loader = yaml.SafeLoader
		# yaml_dumper = yaml.SafeDumper
		# yaml_tag = u'!d'
		# def __init__(self, data):
			# Decimal(self, value=data[1:])
		# def __repr__(self):
			# return "d"+super().__repr__()
	def dec_repr(dumper, data):
		return dumper.represent_scalar(u'!d', 'd'+str(data))
	yaml.Dumper.add_representer(decimal.Decimal, dec_repr)
	yaml.SafeDumper.add_representer(decimal.Decimal, dec_repr)
	# yaml.Loader.add_implicit_resolver(u'!d', re.compile(r'd\d+'), ['d'])
	yaml.Loader.add_implicit_resolver(u'!d', re.compile(r'd\d*\.?\d+'), ['d'])
	# yaml.SafeLoader.add_implicit_resolver(u'!d', re.compile(r'd\d+'), ['d'])
	yaml.SafeLoader.add_implicit_resolver(u'!d', re.compile(r'd\d*\.?\d+'), ['d'])
	def dec_cons(loader, node):
		return decimal.Decimal(loader.construct_scalar(node)[1:])
	yaml.Loader.add_constructor(u'!d', dec_cons)
	yaml.SafeLoader.add_constructor(u'!d', dec_cons)
	
	parser = optparse.OptionParser(
			usage="usage: %prog [options] <datafile> <root> [command]\n"
			"  [command] may be either 'gen' [default] or 'list'")
	
	genGroup = optparse.OptionGroup(parser, "Options for gen")
	listGroup = optparse.OptionGroup(parser, "Options for list")
	debugGroup = optparse.OptionGroup(parser, "Debugging options")
	
	parser.add_option("-p", "--ipa", dest="IPAmode",
			action="store_true", default=False,
			help="enable phonetic transcriptions")
	parser.add_option("-d", "--depth", dest="depth",
			type="int", default=16,
			help="maximum recursion depth [default: %default]")
	parser.add_option("-H", "--html", dest="HTMLmode",
			action="store_true", default=False,
			help="write output as HTML table")
	
	genGroup.add_option("-n", dest="num",
			type="int", default=1, metavar="numWords",
			help="number of words to generate")
	parser.add_option_group(genGroup)
	
	listGroup.add_option("-0", "--listZeros", dest="ignoreZeros",
			action="store_false", default=True,
			help="include 0-frequency values in list")
	listGroup.add_option("-f", "--showFreqs", dest="showFreqs",
			action="store_true", default=False,
			help="show calculated frequencies in list")
	parser.add_option_group(listGroup)
	
	debugGroup.add_option("-P", "--path", dest="printPaths",
			action="store_true", default=False,
			help="print paths for generated words")
	debugGroup.add_option("-K", "--keepHistory", dest="keepHistory",
			action="store_true", default=False,
			help="save every step of regex application\n"
			"May be hard to read.")
	debugGroup.add_option("--KHSep", dest="KHSep",
			type="string", default="→", metavar="SEP",
			help="what to insert between regex applications")
	debugGroup.add_option("-r", "--seed", dest="seed",
			action="store", default=None, 
			help="random seed")
	parser.add_option_group(debugGroup)
	
	(options, args) = parser.parse_args()
	
	if len(args) < 2:
		parser.error("Not enough arguments")
	if len(args) < 3:
		args.append("gen")
	opts = {
		"ipa":options.IPAmode,
		"HTML":options.HTMLmode,
		"path":options.printPaths,
		"depth":-1*options.depth,
		"keepHistory":options.keepHistory,
		"keepHistorySep":options.KHSep,
		"ignoreZeros":options.ignoreZeros,
		"freq":options.showFreqs,
	}
	random.seed(options.seed)
	Data = yaml.safe_load(open(args[0],'r', encoding="utf8"))
	# Data = 
	if args[2] == "gen":
		if False: # Debug stuff
			try:
				word = list(makeWords(Data, 1, args[1],-1*options.depth, options.keepHistory, options.KHSep))[0]
				# print(word["path"])
				# print(formatWord(word, opts))
				P = Path(Data, args[1], word["path"])
				# print(repr(P))
				# print('P'+P._str())
				# print('O'+printPath(word["path"]))
				# print('L'+printPath(P._list()))
				print(P._list())
				print(P.getWord())
				print(next(P)._list())
				print(P.getWord())
				# print(applyRE(Data, followPath(Data, args[1], word["path"])))
				# print(printPath(word["path"]))
				# print(readPath(printPath(word["path"])))
				# print(followPath(Data, args[1], readPath(printPath(word["path"]))))
				# words = makeWords(Data, options.num, args[1],
					# -1*options.depth, options.keepHistory, options.KHSep)
				# printWords(words, options.IPAmode, options.HTMLmode, options.printPaths)
			except Exception as e:
				print(e)
		if options.HTMLmode:
			print("<table><tr><th>Words</th>")
			if options.IPAmode:
				print("<th>IPA</th>")
			if options.printPaths:
				print("<th>Path</th>")
			print("</tr>")
		for i in range(options.num):
			word = list(makeWords(Data, 1, args[1],-1*options.depth, options.keepHistory, options.KHSep))[0]
			print(formatWord(word, opts))
		if options.HTMLmode:
			print("</table>")
	elif args[2] == "list":
		for word in listAll(Data, args[1], opts):
			print(word)
main()
