The folder contains Analysis codes using CatE and WeSTClass.

The codebase is based on the original codes of [CaTE](https://github.com/yumeng5/CatE) and [WeSTClass](https://github.com/yumeng5/WeSTClass)

### CaTE

You need to provide input seed words in topics.txt and get the new class specific keywords in res_topics.txt as shown in the ipython notebook.

### WeSTClass

WeSTClass needs input keywords (class specific) written in keywords.txt and list of sentences (each per line). It then outputs the results in out.txt whose format is 
```
class_id \t probability (per line)
```
The training of WeSTClass can be seen in the above ipython notebook
