# WordGen
Word generator based on recursive expansion of a context-free grammar.

## Purpose

This program is intended primarily for linguistic use, as exemplified by `syllables.yml`, which is the current complete specification of the phonotactics (and some other things) of my conlang Firen. `sajemtan.yml`, and `ffb.yml` are similar. `english.yml` is WIP English sentence generator.

However, this program is much more general than that. In fact, it is capable of expanding any context-free grammar and then applying arbitrary transformations to it (with either regular expressions or finite-state transducers (specifically Mealy machines)), which probably means that it can (in a roundabout way) implement even context-sensitive grammars. (However, the replacement phase is deterministic, so perhaps not. I have not put in the effort of proving this claim one way or the other.)

## Grammar Syntax

The grammar to be expanded is specified in a YAML 1.2 document, rather than an annotated EBNF (or some other standard form) so as to express what I call "channels", which are an essential part of the functionality of the generator. There are two "special" channels: `val` and `freq`; and one internal-use channel: `path`. `val` is the channel that specifies the grammar (its type is a format-string), while `freq` is the channel that assigns weights to each alternative (its type is a decimal floating-point value). `path` is generated automatically by the program as it traverses the nodes and represents the "parse tree" that corresponds to the generated word (its type is a peculiar type of tree for which I do not have a name).

Logically, a grammar (as I use the term in the context of WordGen) is composed: of a sequence of named "switching nodes", each of which is composed of a sequence of "alternatives", each of which is composed of a set of "channels", three of which are required (but defaultable) named `val`, `freq`, and `path` , along with an arbitrary number of channels containing literal-strings; (optionally) a node named `channels` which contains a mapping from channel-names to strings, which controls the presentation; and (optionally) a node named `replace` which contains a mapping of channel-names to sequences of "replacement stages", which are either a) a sequence of regular expressions with replacement-fields (represented by a mapping of `m` to a regex and `r` to a replacement string) or b) a finite state transducer, the nature of which are described in a following paragraph.

(There is currently a work-in-progress feature to allow literal nodes, which contain only a single "alternative" directly, and which do not count toward the depth and expansion limits. This may be removed entirely or replaced with a different syntax, however. Literals only work with "gen", not with "list".)

[For backwards-compatibility, instead of a node named `replace`, either or both of `replacement` and `replaceIPA` may be given instead, which will function in exactly the same way as `replace[val]` and `replace[ipa]` respectively. These two approaches may not be combined. This is the only part of the code (except the "-p" command-line option, equivalent to "-c ipa") that hardcodes a channel other than the three named above, and for this reason it is deprecated.]

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

Finally, I will describe the syntax through which finite-state transducers are represented in the grammar. From now on, these will be referred to as "FSMs", for brevity. Each FSM is represented by (optionally) a number named `reversed`, and a sequence of states containing sets of rules, where each rule is one of the following:
  - A mapping from a single character to a (templated with its match) string (with optional transition).
  - An array of `map` rules, which replace a character in their first set with the corresponding character in their second set (with optional transition).
  - An array of `match` rules, which perform a transition if the character is in their first set (equivalent to a `map` rule with the first and second sets equal and a non-optional transition).
  - A `default` rule, which is a character rule that matches any character (this is why the templates exist for character rules).
  - A `return` rule, which specifies a transition if none of the above types of rules appled.
  - An `end` rule, which is a string to append to the output if the end of the string is reached in that state.
If a character does not match any rule, it will be written to the output tape unchanged and the state will not be changed. The start state is named `S`, and must be present for an FSM to be recognized. If present, `reversed` shall be interpreted as a bitfield, for which the lowest-order bit (`&1`) reverses the input tape, and the next-lowest-order bit (`&2`) reverses the output tape. [Note that `reversed` is therefore not a valid state name, though this is not specifically checked.]

For an example of all of the above, the following FSMs from `english.yml` will replace any @ in its input with either "a" or "an" depending on whether the following letter is a vowel or a consonant, and then uppercase the first letter of the input:
```yaml
    - reversed: 3
      S: 
        match:
          - ["aeiou", V]
          - ["bcdfghjklmnpqrstvwxyz", C]
      V:
        '@': ["na", S]
        match:
          - ["bcdfghjklmnpqrstvwxyz", C]
      C:
        '@': ["a", S]
        match:
          - ["aeiou", V]
    - S:
        map:
          - ["a-z","A-Z", E]
        return: E
      # States are required to be mappings, not null, but they can be empty
      E: {}
```

[Note that the following FSM will reverse its input:
```yaml
replace:
  val:
    - S: {}
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
                1. If a rule for `c` is in the current state:
                   1. Append the string in that rule to the output (with any "{}" replaced by `c`)
                   2. If that rule names a state: transition to that state
                2. If a `map` or `match` rule matches:
                   1. Perform the given mapping (note that a `match` rule is exactly equivalent to a `map` rule with identical match and replace strings, but that it must name a transition state)
                   2. If that rule names a state: transition to that state
                3. If a `default` rule exists:
                   1. Append the string in that rule to the output (with any "{}" replaced by `c`)
                   2. If that rule names a state: transition to that state
                4. Otherwise:
                   1. Append `c` to the output
                   2. if `return` is specified: Transition to that state
             4. If `end` is specified: append that string to the output
             5. If `stage["reversed"]&2`: Reverse the output
          2. Otherwise, if the stage is a list of regexes:
             1. For each regex:
                1. Globally substitute that regex in the input with the provided string
 8. Otherwise: [Deprecated]
    1. If `replacement` is in the grammar:
       1. Do 7.i.a for channel `val` w.r.t. channel `replacement`
    2. If `replaceIPA` is in the grammar:
       1. Do 7.i.a for channel `ipa` w.r.t. channel `replaceIPA`

