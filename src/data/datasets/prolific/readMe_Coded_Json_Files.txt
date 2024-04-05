* Each file consists of 4 parts
* 1. codes: [...] where each entry is a TORE-Code
	* tokens: lists the index of the tokens to which this code is applied
	* name: lists the lemmatized form of the token
	* index: the ID of the TORE-Code
	* relationship_membership: the index of the TORE-Relationship that this code is part of if any
* 2. docs: [...] lists the documents that make up this part of the dataset
	* name: name of the document
	* begin_index: the index of the first token in that document
	* end:index: the index of the last token in that document
* 3. tokens: [...] lists all tokens of the data set. Tokenization is done via NLTK Tokenizer
	* index: the ID of that token
	* name: the actual characters that make up the token
	* lemma: the lemmatized version of the name
	* pos: The Part-of-Speech Tag of that Token (either Noun (n), Verb (v), Adjective(a), or None ())
	* num_name_codes: Number of arbritrary text codes applied to token (not applicable to this data set)
	* num_tore_codes: Number of TORE-Codes applied to this token
* 4. tore_relationships: [...] where each entry is a TORE-Relationship (TORE relationships were not used for the purpose of this paper)
	* TOREEntity: index of the TORE-Code that this relationship refines
	* target_tokens: index of the tokens that this relationship points at
	* relationship:name: name of the relationship
	* index: the ID of the TORE-Relationship
