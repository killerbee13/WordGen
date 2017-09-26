import sys
import random
import time
import re
import decimal
import optparse
import copy
import collections
import html
from string import Formatter

# import pdb

import gmpy2
import yaml
from tr import tr

sep = re.compile('[^0-9.]*[^0-9.+*=]')
aChar = re.compile('(.)')
expansionCount = 0
one = decimal.Decimal('1')
channels = {}


class Path:
	"""A path to a word."""
	def __next__(self, depth=None):
		if depth:
			self._depth = depth
		# Attempt to increment rightmost-deepest, check for successors
		q = self.asList()
		for n, d in reversed(q):
			if len(n._data[n._root]) > 1:
				j = n._branch + 1
				if j == len(n._data[n._root]):
					continue
				while not n._data[n._root][j].get("freq", one):
					j = n + 1
					if j == len(n._data[n._root]):
						continue
				t = n._data[n._root][j]
				if t.get("freq", one):
					L = [{"freq": 0}] * n._branch
					L.append(t)
					tmp = chooseFrom(n._data, L, self._depth-d + 1)
					print(formatWord(applyRE(n._data, tmp), {
						"ipa": True,
					}))
					n.__init__(n._data, n._root, tmp["path"])
					return self
		raise StopIteration

	def asList(self):
		"""Generates a heap-like (effectively). Used for breadth-first traversal"""
		q = list([(self, 0)])
		i = 0
		while i < len(q):
			# print(q[i])
			n, d = q[i]
			q.extend([(c, d-1) for c in n._children])
			i = i + 1
		return q

	def iter(self):
		return self

	def getWord(self):
		"""return word corresponding to self suitably for formatWord()"""
		Node = self._data[self._root][self._branch]
		sumFreq = sum([
			decimal.Decimal(x.get("freq", one)) for x in self._data[self._root]
		])
		SNode = {
			"val": Node.get("val", ""),
			"ipa": Node.get("ipa", ""),
			"freq": (decimal.Decimal(Node.get("freq", one))/sumFreq)
		}
		rets = {"val": "", "ipa": "", "freq": one}
		for i, s in enumerate(Formatter().parse(SNode["val"])):
			# Reference
			if s[1]:
				# Generate subword from subpath
				tmp = Path(self._data, s[1], self._children[i]).getWord()

				rets["val"] = rets["val"] + s[0] + tmp["val"]
				rets["freq"] = rets["freq"]*tmp["freq"]
				if s[0]:
					# If reference+literal text, insert
					rets["ipa"] = rets["ipa"] + SNode["ipa"] + tmp["ipa"]
				else:
					rets["ipa"] = rets["ipa"] + tmp["ipa"]
			# No reference, only literal text
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
		if isinstance(path, list):
			self._branch = path[0]
			self._children = []
			# Extract root for each child Path
			for i, tnode in enumerate(Formatter().parse(
				Data[root][self._branch].get("val", "")
			)):
				if tnode[1]:
					self._children.append(Path(Data, tnode[1], path[i+1]))

		elif isinstance(path, str):
			path = readPath(path)
			self._branch = path[0]
			self._children = []
			# Extract root for each child Path
			for i, tnode in enumerate(Formatter().parse(
				Data[root][self._branch].get("val", "")
			)):
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


class SwitchingGraph(collections.UserDict):
	class SwitchingNode(collections.UserList):
		class Branch(collections.UserDict):
			class NodeRef(collections.UserString):
				def __repr__(self):
					if self.flist:
						return "n'"+self.data+':'+','.join(self.flist)+"'"
					else:
						return "n'"+self.data+"'"

				def __init__(self, s, l):
					self.flist = l
					# self.data = s
					super().__init__(s)

			def extract(self, val):
				ret = []
				for s in Formatter().parse(val):
					if s[0]:
						ret.append(s[0])
					if s[1]:
						nstr = s[1]
						flist = []
						if s[2]:
							# Trim leading garbage
							_ = re.match(sep, s[2])
							if _:
								_ = _.end()
							# Parse list
							_ = re.split(sep, s[2][_:])
							flist = [str(decimal.Decimal(i)) for i in _]
						ret.append(self.NodeRef(nstr, flist))
				return ret

			def __init__(self, branch):
				# print(repr(branch))
				# print(repr(branch.get("val","")))
				# self.val = self.extract(branch.get("val",""))
				# self.data = branch.copy()
				super().__init__(branch.copy())
				if isinstance(branch.get("val", ""), str):
					self.data["val"] = self.extract(branch.get("val", ""))
				if "freq" not in branch:
					self["freq"] = one

			def _dump(self):
				return "{\t"+",\n\t".join(
					[",\n\t".join(C+":"+self.data[C]) for C in self.data]
				)+"\t}"

		def __init__(self, branches):
			super().__init__([self.Branch(B) for B in branches])

		def _dump(self):
			# return repr(self.data)
			return [B._dump() for B in self.data]

	def __init__(self, data):
		# self.data = {N: self.SwitchingNode(data[N]) for N in data}
		# self.data = dict()
		super().__init__()
		for N in data:
			if N in set(["replace", "replaceIPA", "replacement", "channels"]):
				continue
			self.data[N] = self.SwitchingNode(data[N])
		self.regexes = RegexApplicator(data)

	def __getitem__(self, key):
		if isinstance(key, self.SwitchingNode.Branch.NodeRef) and key.flist:
			node = self.data[key].copy()
			for i in range(min(len(key.flist), len(node))):
				d = decimal.Decimal(key.flist[i])
				node[i]['freq'] = d
			return node
		else:
			return self.data[key]

	def _dump(self):
		return "\n".join([
			N+":\n\t"+'\n\t'.join(self.data[N]._dump()) for N in self.data
		])

	def toYAML(self):
		pass

	def addNode(self, name, branches):
		self.data[name] = self.SwitchingNode(branches)


class RegexApplicator:
	"""This class is NYI"""
	def __init__(self, data):
		self.data = {}
		for N in set(["replace", "replaceIPA", "replacement", "channels"]):
			if N in data:
				self.data[N] = data[N]


def chooseFrom(Data, branches, depth=-16, maxDepth=16):
	"""Select a random value from the branches, recursing
		on references
	"""
	global expansionCount
	specialChannels = set(["val", "freq", "path"])

	if isinstance(branches, dict):
		branches["val"] = branches.get("val", "")
		branches["freq"] = one
		branches["path"] = None
		return branches
	expansionCount += 1
	for x in branches:
		x["val"] = x.get("val", "")
		x["freq"] = decimal.Decimal(x.get("freq", one))
		# Ensure that every channel named exists
		for ch in channels:
			x[ch] = x.get(ch, "")
	branchesSum = sum([x["freq"] for x in branches])
	a = decimal.Decimal(random.uniform(0, float(branchesSum)))
	stop = 0
	# This needs no normalization because values are never directly compared.
	for i, c in enumerate([x["freq"] for x in branches]):
		a -= c
		if a <= 0:
			stop = i
			break

	other_channels = (
		set([_ for _ in branches[stop]]) - specialChannels
	)
	if "path" in branches[stop]:
		pass
	if expansionCount >= maxDepth**2:
		# Expansion limit reached
		print("wordgen.py: expansion limit reached", file=sys.stderr)
		rets = {
			"val": branches[stop]["val"],
			"path": [0],
			"freq": branches[stop]["freq"]/branchesSum
		}
		for ch in other_channels:
			rets[ch] = branches[stop].get(ch, "")
		return rets
	elif depth >= 0:
		# Recursion limit reached
		print("wordgen.py: recursion limit reached", file=sys.stderr)
		rets = {
			"val": branches[stop]["val"],
			"path": [0],
			"freq": branches[stop]["freq"]/branchesSum
		}
		for ch in other_channels:
			rets[ch] = branches[stop].get(ch, "")
		return rets
	else:
		rets = {"val": "", "freq": one/branchesSum, "path": [stop]}
		# If val is empty, simply return the other channels
		if not branches[stop]["val"]:
			for ch in other_channels:
				rets[ch] = branches[stop].get(ch, "")
			return rets
		# Determine which is a string and which is a reference
		else:
			for s in Formatter().parse(branches[stop]["val"]):
				# Recurse on reference and insert results into string

				if s[0]:
					rets["val"] = rets["val"] + s[0]
					for ch in other_channels:
						rets[ch] = rets.get(ch, "") + branches[stop].get(ch, "")
				if s[1]:
					node = copy.deepcopy(Data[s[1]])
					if s[2]:
						_ = re.match(sep, s[2])
						if _:
							_ = _.end()
						flist = re.split(sep, s[2][_:])
						# NYI
						mode = "assign"
						for i in range(min(len(flist), len(node))):
							if flist[i][0] == "*":
								mode = "multiply"
								flist[i] = flist[i][1:]
							elif flist[i][0] == "+":
								mode = "add"
								flist[i] = flist[i][1:]
							elif flist[i][0] == "=":
								mode = "assign"
								flist[i] = flist[i][1:]
							d = decimal.Decimal(flist[i])
							node[i]['freq'] = d
							# nstr += ','+str(d)
						# Data[nstr] = node
					# Throws a KeyError on invalid reference. Not caught
					# because the Python default error message is good
					# enough and there's nothing for the code to do with
					# an error.

					# Fill reference
					tmp = chooseFrom(Data, node, depth+1, maxDepth)
					other_channels.update(
						set([_ for _ in tmp]) - specialChannels
					)

					rets["val"] = rets["val"] + tmp["val"]
					rets["freq"] = rets["freq"]*tmp["freq"]
					rets["path"].append(tmp["path"])

					for ch in other_channels:
							rets[ch] = rets.get(ch, "") + tmp.get(ch, "")
			return rets


def filterRE(RE):
	"""Processes regex from file for use. Currently no-op."""
	return RE


def applyRE(Data, word, keepHistory=False, KHSep=" → "):
	"""Applies regular expressions in Data to word."""
	def doStagedMatchReplace(regexes, word):
		def defaultPlaceholder(defStr, c):
			# return aChar.sub(str, c)
			out = ""
			for t in Formatter().parse(defStr):
				out += t[0]
				if t[1] is not None:
					out += c
			return out

		def doMaps(maps, matches, c):
			def doFSMMatch(map1, map2, c, S):
				if tr(map1, "", c, "cd"):
					return (True, tr(map1, map2, c), S)
				return (False, "", None, None)

			for map in maps:
				m = doFSMMatch(map[0], map[1], c, map[2] if len(map) > 2 else None)
				if m[0]:
					return (m[1], m[2])
			for match in matches:
				m = doFSMMatch(match[0], match[0], c, match[1])
				if m[0]:
					return (m[1], m[2])
			return False

		ret = [word]
		for stage in regexes:
			if isinstance(stage, dict) and "S" in stage:
				# pdb.set_trace()
				state = "S"
				cline = ""
				# print("begin: "+ret[-1])
				if "reversed" in stage and stage["reversed"] & 1:
					ret[-1] = ret[-1][::-1]
				for c in ret[-1]:
					s = stage[state]
					m = doMaps(s.get("map", []), s.get("match", []), c)
					if c in s:
						r = s[c]
						cline += defaultPlaceholder(r[0], c)
						if len(r) > 1:
							state = r[1]
					elif m:
						cline += m[0]
						if m[1]:
							state = m[1]
					elif "default" in s:
						r = s["default"]
						cline += defaultPlaceholder(r[0], c)
						if len(r) > 1:
							state = r[1]
					else:
						cline += c
						if "return" in s:
							state = s["return"]
				if "end" in stage[state]:
					cline += stage[state]["end"]
				if "reversed" in stage and stage["reversed"] & 1:
					ret[-1] = ret[-1][::-1]
				if "reversed" in stage and stage["reversed"] & 2:
					cline = cline[::-1]
				ret.append(cline[:])
			elif isinstance(stage, list) and len(stage) > 0 and "m" in stage[0]:
				for rule in stage:
					if "c" in rule:
						# not continue because rules are never added
						break
					rule["c"] = re.compile(filterRE(rule["m"]))
				cline = ret[-1]
				for rule in stage:
					cline = rule["c"].sub(rule["r"], cline)
				ret.append(cline[:])
			else:
				print("replace stage invalid: {0!r}".format(stage), file=sys.stderr)
		return ret

	ret = {}
	if "replace" in Data:
		assert "path" not in Data["replace"], \
			"path is not a valid channel for replacement rules"
		for channel in Data["replace"]:
			if channel in word:
				ret[channel] = (
					doStagedMatchReplace(
						Data["replace"][channel], word[channel]
					)
				)
	else:  # Compatibility
		if "replacement" in Data:
			ret["val"] = (
				doStagedMatchReplace(
					Data["replacement"], word["val"]
				)
			)
		if "replaceIPA" in Data:
			ret["ipa"] = (
				doStagedMatchReplace(
					Data["replaceIPA"], word["ipa"]
				)
			)
	if keepHistory:
		for channel in ret:
			word[channel] = KHSep.join(ret[channel])
	else:
		for channel in ret:
			word[channel] = ret[channel][-1]
	return word


def listAll(Data, node, opts={
		"ipa": True,
		"HTML": False,
		"path": False,
		"depth": -16,
		"keepHistory": False,
		"keepHistorySep": "→",
		"ignoreZeros": True
	}):
	'''Traverse all descendants of node'''

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
				"val": word[0],
				"ipa": word[1],
				# DFSPrint doesn't work with paths
				"path": [0],
				"freq": word[2]}), opts)
		# newword = applyRE(Data, {"val":word[0], "ipa":word[1]})
		# word = (newword["val"], newword["ipa"], word[2])
		# tmpbuf.append(word[0]+' :\t'+word[1]+'\t'+str(word[2]))
		time.sleep(0.0001)
	# return '\n'.join(tmpbuf)


def listAllR(Data, node, depth, ignoreZeros, path=[], flist=None):
	'''Implementation of listAll. Do not call.'''
	if node in path:
		return {"t": 'V', "node": node}
	elif depth < 0:
		path.append(node)
		list = []
		if not flist:
			flist = [x.get("freq", one) for x in Data[node]]
		listSum = sum(flist)
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
			# Determine which is a string and which is a reference
			matches.append({
				"t": 'A',
				"freq": child.get("freq", one)/listSum,
				"Acontents": []
			})
			# If no val, insert IPA anyway
			if not child.get("val", ""):
				matches[-1]["Acontents"].append(
					{"t": 'L', "val": '', "ipa": child.get("ipa", "")}
				)
			else:
				for s in Formatter().parse(child["val"]):
					# Recurse on reference and insert results into string
					if s[1]:
						nstr = s[1]
						node = Data[s[1]]
						if s[2]:
							_ = re.match('[^0-9.]+', s[2])
							if _:
								_ = _.end()
							flist = re.split('[^0-9.]+', s[2][_:])
							nstr = s[1]
							for i in range(min(len(flist), len(node))):
								d = decimal.Decimal(flist[i])
								node[i]['freq'] = d
								# Flist.append(d)
								nstr += ',' + str(d)
							if nstr not in Data:
								Data[nstr] = node
						else:
							flist = None
						# Throws a KeyError on invalid reference. Not caught because
							# the Python default error message is good enough and
							# there's nothing for the code to do with an error.
						# Fill reference
						tmp = listAllR(Data, nstr, depth+1, ignoreZeros, path, None)
						if s[0]:
							# If reference+literal text, insert
							matches[-1]["Acontents"].append({
								"t": 'L',
								"val": s[0],
								"ipa": child.get("ipa", "")
							})
							matches[-1]["Acontents"].append(tmp)
						else:
							matches[-1]["Acontents"].append(tmp)
					# No reference, only literal text
					else:
						matches[-1]["Acontents"].append({
							"t": 'L',
							"val": s[0],
							"ipa": child.get("ipa", "")
						})
		# path.pop()
		return {"t": 'N', "node": node, "sum": listSum, "Ncontents": matches}
	else:
		# Recursion depth reached
		print("wordgen.py: recursion depth reached", file=sys.stderr)
		return {"t": 'T', "node": node, "raw": Data[node]}


def DFSPrint(Node, freq=1):
	'''Generate list of words suitable for printing from tree structure.'''
	# Main case
	def f_A(Node, freq):
		buf1 = [("", "", 1)]
		for n in Node["Acontents"]:
			# tfreq = freq*Node["freq"]
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

	# Simply iterate and recurse
	def f_N(Node, freq):
		ret = []
		# N will always contain As
		for n in Node["Ncontents"]:
			ret.extend(DFSPrint(n, freq*n["freq"]))
		return ret

	# Leaf
	def f_L(Node, freq):
		# print('L: '+str(path))
		return [(Node["val"], Node["ipa"], freq)]

	# Turn into reference
	def f_V(Node, freq):
		return [("{"+Node["node"]+"}", "{"+Node["node"]+"}", freq)]

	# Truncation -- pretend it's L but different
	def f_T(Node, freq):
		# print('T: '+str(freq))
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
	if formatStr is not None:
		# dbgWord = word.copy()
		# del dbgWord["path"]
		# del dbgWord["freq"]
		# print(dbgWord)
		return formatStr.format(**word)
	else:
		if not opts["HTML"]:
			fstr = ""
			for ch in opts["channels"]:
				fstr += "{"+ch
				if ch == "path":
					word[ch] = printPath(word.get(ch, ""))
				elif ch == "freq":
					fstr += ":.4e"
				else:
					word[ch] = word.get(ch, "")
				fstr += "}\t"
		else:
			fstr = "<tr>"
			for ch in opts["channels"]:
				fstr += "<td>{"+ch+"}</td>"
				if ch == "path":
					word[ch] = printPath(word.get(ch, ""))
				else:
					word[ch] = html.escape(word.get(ch, ""))
			fstr += "</tr>"
		word["val"] = word.get("val", "")
		return formatWord(word, opts, fstr)


def printPath(path):
	def recurse(path):
		ret = gmpy2.mpz(path[0]).digits(62)
		for a in path[1:]:
			if a:
				ret = ret + '[' + (recurse(a)) + ']'
		return ret
	if path:
		return '+' + recurse(path)
	else:
		return '+0'


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
				tret, ti = constructPath(tokens, i+1)
				ret.append(tret)
				if ti == len(tokens):
					pass
					# raise ValueError(
					#	 "Unterminated subpath",
					#	 i,
					#	 str(tokens),
					#	 str(tokens[i:])
					# )
				i = ti
			elif tokens[i] == "]":
				return ret, i
			else:
				ret.append(int(gmpy2.mpz(tokens[i], 62)))
			i += 1
		return ret, i
	try:
		return constructPath(list(tokensOf(pathStr)))[0]
	except ValueError as err:
		# In case of error, report the full path in addition to the subpath
		err.args = err.args + (pathStr, )
		raise


def followPath(Data, node, path):
	# print(node)
	# root = [
	# {
		# "val": x["val"],
		# "ipa": x.get("ipa",""),
		# "freq":decimal.Decimal(x.get("freq",one))
	# }
	# for x in Data[node] if x.get("freq",one)
	# ]
	sumFreq = sum(
		[decimal.Decimal(x.get("freq", one)) for x in Data[node]]
	)
	SNode = {
		"val": Data[node][path[0]].get("val", ""),
		"ipa": Data[node][path[0]].get("ipa", ""),
		"freq": (
			decimal.Decimal(Data[node][path[0]].get("freq", one))
			/ sumFreq
		)
	}
	rets = {"val": "", "ipa": "", "freq": one}
	# print(SNode)
	for i, s in enumerate(Formatter().parse(SNode["val"])):
		# Recurse on reference and insert results into string
		if s[1]:
			# Throws a KeyError on invalid reference. Not caught because
				# the Python default error message is good enough and there's
				# nothing for the code to do with an error.
			# Fill reference
			tmp = followPath(Data, s[1], path[i+1])

			rets["val"] = rets["val"] + s[0] + tmp["val"]
			rets["freq"] = rets["freq"]*tmp["freq"]
			if s[0]:
				# If reference+literal text, insert
				rets["ipa"] = rets["ipa"] + SNode["ipa"] + tmp["ipa"]
			else:
				rets["ipa"] = rets["ipa"] + tmp["ipa"]
		# No reference, only literal text
		else:
			rets["val"] = rets["val"] + s[0]
			rets["ipa"] = rets["ipa"] + SNode["ipa"]
	return rets


def toBNF(Data, StartDef):
	nodes = Data.copy()
	for N in set(["replace", "replaceIPA", "replacement", "channels"]):
		nodes.pop(N)
	pass


def main():
	global expansionCount

	# Enable shorthand for decimal numbers:
	def dec_repr(dumper, data):
		return dumper.represent_scalar(u'!d', 'd'+str(data))

	yaml.Dumper.add_representer(decimal.Decimal, dec_repr)
	yaml.SafeDumper.add_representer(decimal.Decimal, dec_repr)
	yaml.Loader.add_implicit_resolver(u'!d', re.compile(r'd\d*\.?\d+'), ['d'])
	yaml.SafeLoader.add_implicit_resolver(u'!d', re.compile(r'd\d*\.?\d+'), ['d'])

	def dec_cons(loader, node):
		return decimal.Decimal(loader.construct_scalar(node)[1:])

	yaml.Loader.add_constructor(u'!d', dec_cons)
	yaml.SafeLoader.add_constructor(u'!d', dec_cons)

	# Command-line options
	parser = optparse.OptionParser(
			usage="usage: %prog [options] <datafile> <root> [command]\n"
			"  [command] may be either 'gen' [default], 'list', or 'diag'")

	genGroup = optparse.OptionGroup(parser, "Options for gen")
	listGroup = optparse.OptionGroup(parser, "Options for list")
	diagGroup = optparse.OptionGroup(parser, "Options for diag")
	debugGroup = optparse.OptionGroup(parser, "Debugging options")

	parser.add_option("-c", "--channel", dest="channels",
			action="append", metavar="CHANNEL", default=[],
			help="print CHANNEL (can be used multiple times)")
	parser.add_option("-p", "--ipa", dest="channels",
			action="append_const", const="ipa",
			help="print IPA transcriptions (-c ipa)")
	parser.add_option("-d", "--depth", dest="depth",
			type="int", default=16,
			help="maximum recursion depth [default: %default]")
	parser.add_option("-H", "--html", dest="HTMLmode",
			action="store_true", default=False,
			help="write output as HTML table")
	genGroup.add_option("-n", dest="num",
			type="int", default=1, metavar="numWords",
			help="number of words to generate")
	genGroup.add_option("-V", dest="noVal",
			action="store_true", default=False,
			help="Suppress implicit 'val' printing")
	genGroup.add_option("-q", "--quiet", dest="quiet",
			action="store_true", default=False,
			help="Disable printing of the header")
	genGroup.add_option("-F", "--fmt", dest="fstr",
			type="string", metavar="FMT_STR", default=[],
			help="Format string for printing words")
	parser.add_option_group(genGroup)

	listGroup.add_option("-0", "--listZeros", dest="ignoreZeros",
			action="store_false", default=True,
			help="include 0-frequency values in list")
	parser.add_option_group(listGroup)

	diagGroup.add_option("--regex", dest="dbgRE",
			action="store_true", default=False,
			help="Dump regular expressions after filtering.")
	diagGroup.add_option("--nodes", dest="dbgNodes",
			action="store_true", default=False,
			help="Dump switching nodes after filtering.")
	diagGroup.add_option("--retest", dest="dbgRETest",
			action="store_true", default=False,
			help="Apply regexes for CHANNELS to input.")
	diagGroup.add_option("--bnf", dest="dbgBNFExport",
			action="store_true", default=False,
			help="Export to BNF (val only).")
	parser.add_option_group(diagGroup)

	debugGroup.add_option("-P", "--path", dest="channels",
			action="append_const", const="path",
			help="print paths for generated words (-c path)")
	debugGroup.add_option("-K", "--keepHistory", dest="keepHistory",
			action="store_true", default=False,
			help="save every step of regex application\n"
			"May be hard to read.")
	debugGroup.add_option("--KHSep", dest="KHSep",
			type="string", default=" → ", metavar="SEP",
			help="what to insert between regex applications")
	debugGroup.add_option("-r", "--seed", dest="seed",
			action="store", default=None,
			help="random seed")
	debugGroup.add_option("-f", dest="channels",
			action="append_const", const="freq",
			help="show calculated frequencies (-c freq)")
	parser.add_option_group(debugGroup)

	(options, args) = parser.parse_args()

	if len(args) < 2:
		parser.error("Not enough arguments")
	if len(args) < 3:
		args.append("gen")
	if "ipa" in options.channels:
		options.IPAmode = True
	else:
		options.IPAmode = False
	if "path" in options.channels:
		options.path = True
	else:
		options.path = False
	if "freq" in options.channels:
		options.showFreqs = True
	if not options.noVal:
		options.channels = ['val'] + options.channels
	opts = {
		"HTML": options.HTMLmode,
		"path": options.path,
		"depth": -1*options.depth,
		"keepHistory": options.keepHistory,
		"keepHistorySep": options.KHSep,
		"ignoreZeros": options.ignoreZeros,
		"channels": options.channels,
	}
	random.seed(options.seed)
	Data = yaml.safe_load(open(args[0], 'r', encoding="utf8"))
	if args[2] == "gen":
		if False:  # Debug stuff
			try:
				word = applyRE(
					Data,
					chooseFrom(
						Data,
						Data[args[1]],
						-1*options.depth,
						options.depth
					),
					options.keepHistory,
					options.KHSep
				)
				# print(word["path"])
				# print(formatWord(word, opts))
				P = Path(Data, args[1], word["path"])
				# print(repr(P))
				# print('P'+P._str())
				print('O'+printPath(word["path"]))
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
		Header = ""
		# Default some channel names for printing
		channels = {"val": "Words", "ipa": "IPA", "path": "Path"}
		if "channels" in Data:
			for ch, name in Data["channels"].items():
				channels[ch] = name
		if options.HTMLmode:
			Header = "<table><tr>"
			for ch in options.channels:
				Header += "<th>"+html.escape(channels.get(ch, ch))+"</th>"
			Header += "</tr>"
			print(Header)
		else:
			if not options.quiet:
				Header += '\t'.join([
					channels.get(ch, ch) for ch in options.channels
				])
				print(Header)
				print('-'*40)
		for _ in range(options.num):
			word = applyRE(
				Data,
				chooseFrom(
					Data,
					Data[args[1]],
					-1*options.depth,
					options.depth
				),
				options.keepHistory,
				options.KHSep
			)
			expansionCount = 0
			print(formatWord(word, opts))
		if options.HTMLmode:
			print("</table>")
	elif args[2] == "list":
		for word in listAll(Data, args[1], opts):
			print(word)
	elif args[2] == "diag":
		if options.dbgRE:
			if "replace" in Data:
				for channel in Data["replace"]:
					print(channel+':')
					for stage in Data["replace"][channel]:
						print('  [')
						for rule in stage:
							print(
								'    {'
								+ "m: {m}, r: {r}".format(
									m=repr(filterRE(rule['m'])),
									r=repr(rule['r'])
								)
								+ '}'
							)
						print('  ]')
		if options.dbgNodes:
			G = SwitchingGraph(Data)
			G.addNode(":arg", [{"val": args[1]}])
			print(repr(G))
			print(repr(G[":arg"]))
		if options.dbgRETest:
			for ch in options.channels:
				pass
		if options.dbgBNFExport:
			print('-'*40)
			print(toBNF(Data, args[1]))
			print('-'*40)


main()
