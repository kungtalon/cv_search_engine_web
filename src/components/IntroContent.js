module.exports = `<!DOCTYPE html>

<html>
<head>
<meta charset='UTF-8'><meta name='viewport' content='width=device-width initial-scale=1'>
<title>introduction</title>
</head>
<body>
<h2 id='' align='center'>A Search Engine for Academic Computer Vision Papers</h2>
<hr/>
<h3 id='1-introduction'>1. Introduction</h3>
<p>With the development of artificial intelligence, thousands of papers are published every year. How to manage all these papers becomes a challenge. Besides, to better utilize these precious research results, how can we build a search engine that can help us find related work quickly and precisely?</p>
<p>One big success is the well-known ArXiv, which is a free distribution service and an open-access archive for 1,968,218 scholarly articles in the fields of physics, computer science, statistics, electrical engineering and so on. Undoubtedly, it is a great thing that people can find most of the papers they want on such a paper search engine, but it can be improved in many aspects.
First, ArXiv is sensitive to typo. Even if you capitalize or forget to capitalize a certain letter, you might fail to find desired results.</p>
<p>Second, the searching algorithm in ArXiv is not intelligent enough to understand your query. For instance, if you have just started your journey in computer vision and want to know more about the famous &quot;ResNet&quot;, you cannot find the paper if you search for &quot;Residual networks&quot;. Instead, the title for that paper is &quot;Deep Residual Learning for Image Recognition&quot;.</p>
<p>Third, during paper review, researchers need to search for a specific topic or method. For experienced researchers, these might not be problems, but beginners need a more friendly platform.</p>
<p>Based on the reasons above, we aim to design a search engine for searching academic papers.</p>
<ul>
<li>It should be friendly to users who are not very experienced in computer vision, such as students.</li>
<li>Users can search for specific topic or method, such as &#39;Transformer&#39;, &#39;Bert&#39;, &#39;MLP-Mixer&#39;, and obtain relevant papers which use these methods.</li>
<li>It is robust to small typo problems. For instance, if a user types &quot;RexNet&quot;, the search engine will still find papers related to &quot;ResNet&quot;.</li>

</ul>
<p>We built the search engine through a learn-to-rank model trained by lightGBM (also LGBM), which is a powerful machine learning model frequently used for classification and ranking. Our model takes multiple features as input and outputs a list of papers with high relevance scores.</p>
<hr />
<h3 id='2-data'>2. Data</h3>
<p>Before comming to the step of model building, we have to obtain a well-labeled dataset for this specific task.</p>
<p>We choose papers that are published on Conference on Computer Vision and Pattern Recognition (CVPR) and International Conference on Computer Vision (ICCV) as our data. CVPR is an annual conference on computer vision and pattern recognition, which is regarded as one of the most important conferences in its field. ICCV is another top conference in computer vision and is held every two years. We choose CVPR and ICCV as our domain because many outstanding papers are published on them.</p>
<p>Selenium is a convenient autonomous crawler in Python. With Selenium, we obtained approximately 15,000 papers published in CVPR and ICCV from 2015 to 2021. Each paper can be treated as a document, which consists of title, authors, abstract, LaTex version, publised year, whether it is in the workshop and whether it has supplementary material. And we use regular expressions to extract the subsections from the latex. The obtained data are stored in Pandas DataFrame.</p>
<div align="center">
<img src="https://cdn-images-1.medium.com/max/800/1*Jhf_20CBYAm-mwkstSiBlA.png" width="65%"/>
<p className='img-caption'>An example of our crawled papers</p>
</div>
<p>The next step is to select several queries and annotate the relevance between the queries and some candidate documents. The candidates are first retrieved by classical methods like BM25. We scored the relevance from 5 (very high relevance) to 1 (very low relevance) and seperated the dataset into training data (40 queries), validation data (10 queries) and test data (10 queries).</p>
<hr />
<h3 id='3-methods'>3. Methods</h3>
<p>Initially, we use <a href='https://pyterrier.readthedocs.io/en/latest/'>PyTerrier</a> to index the documents and retrieve 50 papers with top relevance. And then learn-to-rank model will be applied to reranking these papers.</p>
<h4 id='31-word2vec'>3.1 Word2Vec</h4>
<p><a href='https://github.com/RaRe-Technologies/gensim'>Gensim</a> is a useful library when it comes to mapping each word into a semantic vector, which is called the embedding vector. After turning words into embeddings, words with similar meanings with have similar embeddings.</p>
<div align="center">
<img src="https://cdn-images-1.medium.com/max/800/0*Ri2A787ZiMR2LvF0.png" width="60%"/>
<p className='img-caption'>The concept of embedding</p>
</div>
<p>To put all the text data into the Word2vec model, we have to parse the texts into lists of strings. <a href='https://www.nltk.org/'>NLTK</a> can help transform a bunch of long texts into short lists of tokens.</p>
<pre><code>from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
tokenizer = RegexpTokenizer(r&#39;\w+&#39;)
ps = PorterStemmer()
stops = set(stopwords.words(&#39;english&#39;))
data[&#39;abstract_tokens&#39;] = data[&#39;abstract&#39;].apply(lambda s: [ps.stem(t) for t in tokenizer.tokenize(s) if t not in stops])
</code></pre>
<p>After applying preprocessing functions to all of our text fields, we can merge all fields to get a training corpus, with which Word2vec model can learn the relations between words. Here we choose the embedding size to be 256.</p>
<pre><code>corpus = list(data[&#39;abstract_token&#39;]) + list(data[&#39;subsections_token&#39;]) + list(data[&#39;title_token&#39;])
model = Word2Vec(sentences=corpus, vector_size=256, window=5, min_count=1, workers=4, epochs=10)
</code></pre>
<p>Then, this model can be used to generate embeddings for all words in the vocabulary. But to get an embedding for an abstract, we need to sum up all the embeddings of words in this abstract. For queries, it is the same thing. Embeddings of documents are pre-calculated while the embedding for queries are generated on-the-fly.</p>
<h4 id='32-preparing-the-features'>3.2 Preparing the features</h4>
<p>For each pair of query and document, we join the metadata with the pairs and start extract the features which can measure the relations between the query and the documents or some characteristics about the documents.</p>
<p>We defined four types of features for this task:</p>
<ol start='' >
<li>Similarity between the query embedding and the documentem beddings. We included three types of &quot;distances&quot;: inner product, cosine similarity and Euclidean distance</li>

</ol>
<pre><code>def _cal_embedding_dists(self, doc_data):
    series = []
    distances = {
        &#39;dot&#39;: lambda x, y : x.dot(y),
        &#39;cos&#39;: lambda x, y : x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9),
        &#39;euclidean&#39;: lambda x, y : np.linalg.norm(x - y)
    }
    for _, row in doc_data.iterrows():
        cur_dists = []
        query_embedding = self.query_embeddings_dict[row[&#39;qid&#39;]]
        for field in [&#39;title&#39;, &#39;abstract&#39;, &#39;subsections&#39;]:
            dists = [dist_func(query_embedding, row[field + &#39;_embedding&#39;]) for dist_func in distances.values()]
            cur_dists.extend(dists)
        series.append(np.array(cur_dists))
    return pd.Series(series)
</code></pre>
<ol start='2' >
<li>Relevance score given by PyTerrier. We have adopted the methods of TF-IDF, BM25, CoordinatMatch to score the texts.</li>
<li>Some basic attributes of the documents, such as the year, the conference, whether it comes from a workshop and whether it has supplementary materials.</li>

</ol>
<pre><code>import pyterrier as pt
def _get_doc_property(self):
    # publish time, conference, is_workshop, has_supp ... 
    pipeline = (
        (pt.apply.doc_score(lambda row: int(row[&quot;year&quot;])))
        **
        (pt.apply.doc_score(lambda row: int(row[&quot;conference&quot;]==&quot;CVPR&quot;)))  #CVPR:1,ICCV:0
        **
        (pt.apply.doc_score(lambda row: int(row[&quot;workshop&quot;]!=&#39;&#39;)))
        **
        (pt.apply.doc_score(lambda row: int(row[&quot;supp_link&quot;]!=&#39;&#39;)))
        **
        (pt.apply.doc_score(lambda row: row[&#39;score&#39;]))
    )
    res = pipeline.transform(self.data)[&#39;features&#39;]
    return res
</code></pre>
<ol start='4' >
<li>Whether part of the query matches the author names.</li>

</ol>
<h4 id='33-learn-to-rank-with-lightgbm'>3.3 Learn-to-rank with LightGBM</h4>
<p>The library of <a href='https://lightgbm.readthedocs.io/en/latest/index.html'>lightgbm</a> is a well-written, open-source library in Python based on <a href='https://scikit-learn.org/stable/'>Scikit-Learn</a>. So we can easily use this library to train our model with sklearn APIs. Also, this class has implemented a version for listwise ranking, which is most suited for our task of learning to rank. For training, we directly use the annotated relevance from 1 to 5 as the target label and fit the model using the features extracted and the corresponding labels.</p>
<pre><code>self.model = lgb.LGBMRanker(
    task=&quot;train&quot;,
    silent=True,
    min_child_samples=1,
    num_leaves=24,
    max_depth=5,
    objective=&quot;lambdarank&quot;,
    metric=&quot;ndcg&quot;,
    learning_rate= 0.064,
    importance_type=&quot;gain&quot;,
    num_iterations=150,
    subsample=0.8
)
self.model.fit(
    train_feats,
    train_labels,
    group=train_features.groupby(&#39;qid&#39;)[&#39;qid&#39;].count().to_numpy(),
    eval_set=[(val_feats,val_labels)],
    eval_group=[val_features.groupby(&#39;qid&#39;)[&#39;qid&#39;].count().to_numpy()],
    eval_at=[5, 10, 20],
    eval_metric=&#39;ndcg&#39;
)
</code></pre>
<hr />
<h3 id='4-results-and-discussions'>4. Results and Discussions</h3>
<p>First, let&#39;s introduce the metric used in our project. nDCG is a type of weighted sum of relevance scores of a sorted list of retrieved documents.</p>
<div align="center">
<img src="https://cdn-images-1.medium.com/max/800/1*tQolkMu38lgpy--GVPEFIQ.png" width="30%">
</div>
<p>where relᵢ is the relevance score of the i-th document, and iDCG is the maximum possible DCG score when the order becomes the ideal order. This metric attaches more importance to the correct ranking of top-k results.</p>
<p>Then let&#39;s compare our list-wise LGBM with other baselines we have mentioned above.</p>
<div align="center">
<img src="https://cdn-images-1.medium.com/max/1200/1*bhjPuRzyHRkWF8F9Fwcwbg.png" width="88%">
</div>
<p>The list-wise LGBM outperforms other models.From the figure, we can see that listwise LGBM model works best, while pointwise LGBM and logistic regression cannot exceed BM25. The result is close to what we expect.</p>
<p>LGBM is a boosting algorithm, but logistic regression is a single model. Therefore, it is reasonable that aggregation of different models work better than a single model. Besides, LGBM is able to fit non-linear features, while, as a generalized linear model, logistic regression cannot fit non-linear features well.</p>
<p>The reason why listwise model works better than pointwise model is that listwise model can simultaneously learn both the relations between queries and documents and the relations among different documents given the same query. The essence of reranking is to re-permute the documents. Thus, the score is not the key, but the ordering is. Pointwise model focuses on predicting a score for each sample, but neglect the relation among different samples.</p>
<p>To make more sense of how our model works, we are wondering what features play important role in our model, and the tool to find that out is feature importance. Usual methods include linear regression and tree-based methods. Considering our model LightGBM is based on decision trees, we can easily acquire the importance of each feature.</p>
<pre><code>def plot_importance(model, feat_names):
    values = model.booster_.feature_importance()
    fi = sorted(zip(values, feat_names), reverse=True)
    values, features = zip(*fi)
    
    plt.bar(values[:-1], features[:-1])
    plt.xticks(rotation=90)
    plt.show()
</code></pre>
<p>The number shows how many times that the feature has been used to split a node in decision trees of LGBM. The more times it is used, the more crucial the feature becomes.</p>
<div align="center">
<img src="https://cdn-images-1.medium.com/max/1200/1*uP4vgpLqG-9H9pKA-joCfA.png" width="80%"/>
<p>The ranking of feature importance of our LGBM model.</p>
</div>
<p>In the picture, &quot;score&quot; is the weighted sum of BM25 scores of different parts of a document. The suffix &quot;s&quot; stands for subtitles, &quot;a&quot; for &quot;abstract&quot; and &quot;t&quot; for title. We can get that the most important features are BM25 scores and innerproducts of embeddings, which agrees with our expectation.</p>
<hr />
<h3 id='5-whats-next'>5. What&#39;s Next</h3>
<p>There are some possible improvements in our future work.</p>
<ol start='' >
<li><strong>More Powerful Models</strong>. Tree-based algorithm divides data into bins, which may lose accuracy when dealing with contiguous variables. Large language models based on deep neural networks like BERT probably work better.</li>
<li><strong>Query Design</strong>. Our queries might be subjective. We cannot design queries that are related to unfamiliar computer vision topics. An objective way is to refer to some overview papers that divide computer vision into different fields. Then, we can design queries that cover all the fields.</li>
<li><strong>Annotation</strong>. The distributions of scores labeled by two members are different, which might be a potential problem. We may need to design a rigorous rule of labeling.</li>
<li><strong>More Data Sources</strong>. We can include more computer vision conferences, such as ECCV, into our database.</li>
<li><strong>Sampling</strong>. Negative sampling is a good way of data augmentation. And it is also a good way to balance the labels if low-relevance samples are under-represented.</li>

</ol>
<hr />
</body>
</html>`;