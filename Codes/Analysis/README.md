The folder contains Analysis codes using CatE and WeSTClass.

The codebase is based on the original codes of [CaTE](https://github.com/yumeng5/CatE) and [WeSTClass](https://github.com/yumeng5/WeSTClass)

### CaTE

You need to provide input seed words in topics.txt (under ``CateE/datasets/tweet/``) and get the new class specific keywords in res_topics.txt (under ``CateE/datasets/tweet/``) as shown in the ipython notebook ``Cate_Analysis.ipynb``.

### WeSTClass

WeSTClass needs input keywords (class specific) written in keywords.txt (under ``WestClass/tweet/``) and the list of sentences to be classified (each per line). It then outputs the results in out.txt (under ``WestClass/tweet/``) whose format is 
```
class_id \t probability (per line)
```
The training of WeSTClass can be found in the ipython notebook ``WestClass_Training.ipynb``.
