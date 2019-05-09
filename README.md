# pbcplus2pomdp

Python scripts that translate probabilistic action descriptions in pBC+ to POMDP models. The outputs are .pomdp files that can be used as input to POMDP solver APPL (https://github.com/AdaCompNUS/sarsop).

## Usage

Noncompositional generation:

$ python pbcplus2pomdp.py path/to/lpmln/files discount_factor(0-1)

Compositional generation:

$ python pbcplus2pomdp_compositional.py path/to/lpmln/file/with/no/action path/to/lpmln/action/description/files discount_factor(0-1)

For example,

$ python pbcplus2pomdp.py examples/tiger.lpmln  0.95

$ python pbcplus2pomdp_compositional.py examples/dialog/dialog-comp/dialog_noaction.lpmln examples/dialog/dialog-comp/actions 0.95

## System Dependencies

This Python script requires the following system to be installed
- Python 2.7
- clingo python library: https://github.com/potassco/clingo/blob/master/INSTALL.md
- lpmln2asp system: http://reasoning.eas.asu.edu/lpmln/index.html

## Examples

The folder "examples" contains lpmln encodings for several example pBC+ action description.

- tiger.lpmln: the well-known tiger example in POMDP literature

- two_tiger.lpmln: a variation of the tiger example where there are two tigers and three doors

- dialog: an example from [1](http://www.cs.binghamton.edu/~szhang/papers/2017_CONF_AAAI_Zhang.pdf)

## References

[1] Zhang, Shiqi, Piyush Khandelwal, and Peter Stone. "Dynamically constructed (po) mdps for adaptive robot planning." Thirty-First AAAI Conference on Artificial Intelligence. 2017.
