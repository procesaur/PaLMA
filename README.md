# Parallel language models and applications

<div>

## About

Application is developed in support of the University of Belgrade doctoral dissertation _Composite pseudogrammars based on parallel language models of Serbian_ by Mihailo Škorić under the supervision of Ranka Stanković.

Usage instructions are shown bellow for each of three usage scenarios:

</div>

<div>

## Perplexity visualisation

requirements:

*   Inputted text
*   Selection of language model
*   Slider size definition

Any kind of text is accepted as input. Inputted text will be truncated if too long!

Easiest way to select language model is to paste its huggingface id, eg. _procesaur/gpt2-srlat_. The model will be downloaded to a local computer if it is not already!  
For visualisation language models like GPT and BERT (and perhaps other) should work.  
Selected model should ideally be applicable for the inputted text, eg. the language of the model and text should match.

Slider size should be set between 2 and 10, and it will define the size of the token-n-grams being evaluated using the model, in order to produce the final values.  
The shorter the text, the higher impact of this variable will be.

In so, the parameters ![](/static/help1.png) will output: ![](/static/help2.png)

</div>

<div>

## Generation

requirements:

*   Inputted text
*   Selection of (GPT) language model
*   Outputted text length definition
*   Temperature definition

and optionally:

*   Number of outputs definition
*   Selection of the second model that will evaluate all of the outputs

Again, any kind of text is accepted as input. Inputted text will be truncated if too long!

Easiest way to select language model is to paste its huggingface id, eg. _procesaur/gpt2-srlat_. The model will be downloaded to a local computer if it is not already!  
Any GPT2 (and just maybe another generative) model should work.

Outputted text length can be set to a 100 tokens max (if you do not edit source code), and it represents the length of the newly generated text, not regarding the input

Temperature parameter set between 0 and 1 using a slider defines weather, and how much will the generated text be unique.  
Setting the temperature to 0 will result in most conservative generation, and setting it to 1 will result in most far-fetched text.

Number of outputs simply defines the number of outputted samples the model will generate.  
If nothing is selected, default is 1.

Second language model is selected just like the first eg. _procesaur/gpt2-srlat_, and any language models like GPT and BERT (and perhaps other) should work.  
This model will be used to evaluate generated samples.  
If it is applied, perplexities will be shown on the right side of each generated sample.

In so, the parameters ![](/static/help3.png) will output: ![](/static/help4.png)

</div>

<div>

## Full Evaluation (Serbian only)

requirements:

*   Inputted text
*   Specialized resources and tools on the local computer

Any kind of text is accepted as input, but it will be truncated if too long! Of course, you should input text on Serbian or similar language.

If you are running this on a local instance, you probably will not be able to use this feature, as certain tools and files, such as TreeTagger and Serbian Morphological Dictionary is needed for preprocessing.  

If you input a sentence:  
_Tokom naredna tri meseca, Džejk i Nejtiri se zaljubljuju dok on počinje da se povezuje sa domorocima._ You will receive an output: ![](/static/help5.png)

The graph represents different disturbances in perplexity (eg. semantical > orange and syntactical > brown) Above the graph nominal perplexity for each model is presented, and bellow the graph there are four different measures of the correctness.  
Each is tagged with either **OK** or **NOT OK**, followed by the probability of that choice. the four measures are:

*   General -- overall correctness
*   Semantics -- semantical correctness
*   Syntax (forms) -- syntactical correctness regarding word forms
*   Syntax (word order) -- syntactical correctness regarding word order in the sentences

</div>