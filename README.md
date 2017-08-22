# WordGen
Word generator based on recursive expansion of a context-free grammar.

## Purpose

This program is intended primarily for linguistic use, as exemplified by `syllables.yml`, which is the current complete specification of the phonotactics (and some other things) of my conlang Firen. `sajemtan.yml`, `english.yml`, and `ffb.yml` are similar (though `english.yml` is currently mostly empty, given that I have no good sources to go on and have not put in the effort to write my own description (not that I claim to have the training to be able to write a perfectly accurate one anyway).)

However, this program is much more general than that. In fact, it is capable of expanding any context-free grammar and then applying arbitrary transformations to it (with either regular expressions or finite-state transducers (specifically Mealy machines)), which probably means that it can (in a roundabout way) implement even context-sensitive grammars. (However, the replacement phase is deterministic, so perhaps not. I have not put in the effort of proving this claim one way or the other.)

## Grammar Syntax

The grammar to be expanded is specified in a YAML 1.2 document, rather than an annotated EBNF (or some other standard form) so as to express what I call "channels", which are an essential part of the functionality of the generator. There are two "special" channels: `val` and `freq`; and one internal-use channel: `path`. `val` is the channel that specifies the grammar (its type is a format-string), while `freq` is the channel that assigns weights to each alternative (its type is a decimal floating-point value). `path` is generated automatically by the program as it traverses the nodes and represents the "parse tree" that corresponds to the generated word (its type is a peculiar type of tree for which I do not have a name).

Logically, a grammar (as I use the term in the context of WordGen) is composed: of a sequence of named "switching nodes", each of which is composed of a sequence of "alternatives", each of which is composed of a set of "channels", three of which are required (but defaultable) named `val`, `freq`, and `path` , along with an arbitrary number of channels containing literal-strings; (optionally) a node named `channels` which contains a mapping from channel-names to strings, which controls the presentation; and (optionally) a node named `replace` which contains a mapping of channel-names to sequences of "replacement stages", which are either a) a sequence of regular expressions with replacement-fields (represented by a mapping of `m` to a regex and `r` to a replacement string) or b) a finite state transducer, the nature of which are described in a following paragraph.

[For backwards-compatibility, instead of a node named `replace`, either or both of `replacement` and `replaceIPA` may be given instead, which will function in exactly the same way as `replace[val]` and `replace[ipa]` respectively. These two approaches may not be combined. This is the only part of the code that hardcodes a channel other than the three named above, and for this reason it is deprecated.]

In the actual data file, that looks something like this (excerpted from `ffb.yml`):
```yaml
replace:
  val:
    - - m: 'ss'
        r: 'z'
      - m: 'xx'
        r: 'j'

Consonant:
  - val: p
  - val: b
  - val: k
  - val: g
  - val: f
  - val: v
  - val: t
  - val: d
  - val: s
  - val: x
```

This example uses only literal-strings in `val`, for simplicity. However, `val` is allowed to contain (and in most interesting cases will contain) references to other switching-nodes. These borrow the syntax of Python format-strings, but since the semantics are entirely different, I will specify them here anyway. As used in this program, a format-string consists of a sequence of literal-strings and references, where references are notated `"{as such}"`. References are also allowed to contain frequency-lists (aka 'flists'), which look like this: `"{Syllable:3 1 0}"`. These overwrite the `freq`s of each alternative of the named switching-node in order. If there are more elements in the flist than alternatives in the node, the excess elements are silently ignored. If there are more more alternatives in the node than elements in the flist, then the remainder are simply not overwritten.

Finally, I will describe the syntax through which finite-state transducers are represented in the grammar. From now on, these will be referred to as "FSMs", for brevity. Each FSM is represented by (optionally) a number named `reversed`, and a sequence of states containing sets of rules, where each rule is a mapping from an input character to an output string and a destination state (defaulting to the current state if not provided), *or* a special rule, of which there are currently three types: `default`, `return`, and `end`. If a character read by the input does not match any rule, it will be written to the output tape and the state will not be changed, unless there is a `default` rule, which supercedes this fallthrough, or a `return` rule, which does not supercede the fallthrough but does cause the machine to transition to a named state. (Note: `return` rules do not specify an output string.) The `end` rule matches the end of the input tape, and does not name a state to transition to because there is no further input. The start state is named `S`, and must be present for an FSM to be recognized. If present, `reversed` shall be interpreted as a bitfield, for which the lowest-order bit (`&1`) reverses the input tape, and the next-lowest-order bit (`&2`) reverses the output tape. [Note that `reversed` is therefore reserved as a name for a state.]

[Note that the following FSM will reverse its input:
```yaml
replace:
  val:
    - S: {} # States are required to be mappings, not null, but they can be empty
      reversed: 1
```
Try doing *that* with a regex.]

## Algorithm

The algorithm accepts as input: a datafile, containing a specification in YAML of the grammar; the root node; the mode of operation (we will assume `gen` because it is the most interesting as well as being the default); and a number of options, of which the only two that aren't presentation are "-n", which specifies how many expansions to do, and "-d, --depth", which is used to prevent the program from looping infinitely and/or producing excessively large output for pathological cases.

The algorithm is then called on the named root switching-node *n* times with a maximum recursion depth of *d* and a maximum expansion count of *d\*\*2*.

Abstractly:

 1. Increment the expansion count (which begins at 0 and is global). If we have exceeded the maximum expansion count, or the maximum depth, then abort with an error, returning the literal, unexpanded text of `val` and all other channels to the caller.
 2. First, all frequencies of the current node's alternatives are summed, then a random (decimal floating point) number between 0 and that maximum is generated. Each alternative's frequency is subtracted from that generated number in turn until it is <= 0, at which point the alternative has been selected. (Or, in other words: concatenate all of the `freq` values as ranges on the number line and select a random point within the total range, then select the alternative corresponding to the subrange including the selected point.)
 3. If `val` is empty, simply return the other channels literally
 4. Otherwise, parse `val` into a list of pairs of literal-strings and references. For each element:
    1. If it has a reference:
       1. If it has an flist, extract the flist, then create a copy of the referenced node and overwrite its `freq` values with the extracted flist [Note: This part of the algorithm is currently being updated to support other behavior than simply overwriting the `freq` values, but as of now that is unimplemented]
       2. Recurse on the referenced node, passing an incremented copy of depth
       3. For each channel:
          1. Append the literal-string in that channel in the current alternative and the returned value from the recursive call to the output [Note that the format-string parser parses a string into pairs of literals and references, which does slightly convolute this section]
    2. Otherwise, it is only a literal-string:
       1. For each channel: append the literal-string in that channel to the output
 5. return the output
 6. [After the root node has been fully expanded:]
 7. if `replace` is in the grammar:
    1. For each channel in the output:
       1. For each stage in `replace`:
          1. if the stage is an FSM:
             1. start at state S
             2. If `stage["reversed"]&1`: reverse the input
             3. For each character `c` in the input:
                1. If a rule matching `c` is in the current state (including `default`):
                   1. Append the string in that rule to the output
                   2.If that rule names a state: transition to that state
                2. Otherwise:
                   1. Append `c` to the output
                   2. if `return` is specified: Transition to that state
             4. If `end` is specified: append that string to the output
             5. If `stage["reversed"]&2`: Reverse the output
          2. Otherwise, if the stage is a list of regexes:
             1. For each regex:
                1. Globally substitute that regex in the input with the provided string
 8. Otherwise: [Deprecated]
    1. If `replacement` is in the grammar:
       1. Do 7.1.1 for channel `val` w.r.t. `replacement`
    2. If `replaceIPA` is in the grammar:
       1. Do 7.1.1 for channel `ipa` w.r.t. `replaceIPA`

