Here are the comprehensive and well-structured markdown notes:

**Welcome to Haskell for Imperative Programmers
=============================

### Prerequisites

* Knowledge of imperative programming (e.g., C, C++, Python, Java)
* Basic theory of programming languages:
	+ Types and how they work
	+ Evaluation of expressions
	+ Difference between compiler and interpreter
* Haskell toolchain installed:
	+ GHC
	+ GHCI
	+ Cabal

### What is Functional Programming?
---------------------------------

### Key Concepts

* **Pure functions**: mathematical definition, input, result, no side effects
* **Immutable data**: data types cannot be changed in place
* **No or less side effects**: absence of side effects, hard to verify
* **Declarative vs. Imperative Programming
------------------------------------

### Imperative Approach

* Example: computing the sum of an array or list of numbers
* Step-by-step instructions:
	+ Set sum to zero
	+ Set i to zero
	+ Iterate until condition is met
	+ Add each value to the sum
	+ Return the sum
* **Algorithm-oriented**: describe an algorithm to produce a sum

### Declarative Approach

* Example: computing the sum of an array or list of numbers
* Define what it means to have a sum:
	+ Sum of an empty list is zero
	+ Sum of a list with at least one element x is x plus the rest of the sum
* **Definition-oriented**: define what it means to have a sum
* Using partial function application and folding function

### Lazy Evaluation
----------------

### Key Concept

* **Lazy evaluation**: evaluate only when necessary
* Example: func1, func2, and func3 take one year each
* **Strict evaluation** (e.g., C, Java): evaluate all functions sequentially
* **Lazy evaluation** (Haskell): evaluate only what is needed
	+ Evaluate z first (1 year)
	+ Evaluate x or y (1 year)
	+ Total time: 2 years

### Importance of Lazy Evaluation

* Evaluate only what is needed
* Performance improvement
* Understand the difference between lazy and strict evaluation

### Next Steps
--------------

* Learn function definitions
* Learn about lists
* Learn to define your own data types