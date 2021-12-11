# API.Cultor development guidelines

* MIR development
  * Define a criteria for algorithms evaluation
  * Test results and make conclusions

* Branch politics
  * Every new feature or bug fix starts in a new branch forked from master
  * Except from major refactorings or migrations use atomic branch (only one use case per each)

* Python programming
  * Respect PEP recomendations, in particular PEP8
  * Every method should have its test
  * Deprecate unused code

* Documentation in .md files using Markdown
  * Convention: Names starts with a caps and spaces are replaced by '-'. Example: 'Dev-Guidelines.md'
