# Репозиторий содержит прорешанные домашние задания по различным курсам. 

## В папке ``DLS and ML`` содержатся прорешанные домашние задания по курсу Deep Learning от DLS МФТИ, а также задания, выполненные в качестве тренировки. 
### Простые работы: 
1. ``1. hw_knn``. Обучен классификатор K-Nearest Neighbors с применением Grid Search и средств библиотек ``pandas``, ``sklearn``. 
2. ``2. hw_linear_models``. Реализована функция градиентного спуска, генератор батчей, класс для логистической регрессии и логистической регрессии с регуляризацией ($l1$ и $l2$). Класс для логистической регрессии с регуляризацией был протестирован обучением на сгенерированных данных и обучением на датасете MNIST с применением кросс-валидации.
3. ``3. hw_dls_ml_kaggle``. Проанализированы табличные данные с помощью ``pandas`` и ``matplotlib``. Реализован pipeline для предобработки данных (нормализация числовых признаков и кодирование категориальных). Обучена логистическая регрессия из ``sklearn``, а также обучен классификатор CatBoostClassifier и сделана посылка на ``Kaggle``.
4. ``3. hw_(extended)``. Расширенная версия предыдущей работы, выполненная в качестве тренировки по использованию инструментов библиотеки ``sklearn``. Проведён более детальный анализ набора данных, исследованы корреляции признаков между собой и с целевой переменной, сделаны выводы. Реализованы три базовых решения: классификатор на основе некоторых условий, случайный и наивный байесовский классификаторы. Реализован подбор параметров для решающего дерева, SVM и логистической регрессии, последние два алгоритма дополнительно обучались на сжатых с помощью PCA данных. Реализован подбор параметров для случайного леса, выявлены наиболее важные признаки и на их основе обучена дополнительная модель. Реализован подбор параметров для различных алгоритмов бустинга над деревьями: LightGBM, XGBoost и CatBoost.
5. ``4. hw_conv_and_fullyconn_networks``. Реализованы класс для логистической регрессии и цикл обучения на ``PyTorch``. Сравнивались функции активации обучением трёхслойной нейронной сети на датасете MNIST. Произведены эксперименты с ядрами свёрток и обучена LeNet на MNIST.
6. ``Video_game_sales_analysis``. Работы представляет собой анализ продаж видеоигр и выполнена в качестве тренировки по использованию инструментов библиотек ``pandas`` и ``matplotlib``. С помощью данных библиотек визуализированы различные графики распределений, корреляционная матрица, а также каждый график сопровождается небольшими выводами. 

### Более объёмные и сложные работы:
7. ``5. hw_dls_classification_simps_kaggle``. Реализованы класс датасета для загрузки изображений, цикл обучения, визуализация, аугментация данных, несколько архитектур свёрточных нейронных сетей с применением MaxPool, BatchNorm, Dropout, а также Fine tuning предобученной ResNet18. Сделан сабмит на ``Kaggle``.
8. ``6. hw_dls_sem_segm``. Реализованы архитектуры SegNet и UNet, а также функции потерь Binary Cross Entropy, Dice loss, Focal loss, а также Correlation Maximized Structural Similarity loss по описанию из статьи. На перечисленных лоссах обучены SegNet и UNet, проведено сравнение качества сегментации по метрике IoU, что отражено в отчёте в конце ноутбука. 
9. ``7. hw_dls_autoencoders``. Реализованы архитектуры Vanilla Autoencoder, Variational Autoencoder (VAE), Conditional VAE. Первая модель обучалась на фотографиях людей, осуществлена генерация фотографий с применением модели на основе случайного тензора, приведённого к латентному распределению. "Пририсованы" улыбки путём выделения "вектора улыбки" на основе латентного вектора для улыбающихся людей. Остальные две модели обучались на датасете MNIST, для них реализована функция loss vae, осуществлена латентная репрезентация в виде точек в двумерном пространстве с помощью sklearn и matplotlib. Прорешана бонусная часть. 
10. ``8. hw_dls_gan_s``. Реализованы архитектуры дискриминатора (классификатора) и генератора, обучена генеративно-состязательная модель, осуществлён сэмплинг, реализован подход к оценке качества генерации на основе KNN, визуализированы распределения реальных и сгенерированных изображений с помощью метода TSNE в sklearn, сделаны выводы.
11. ``9. DLS_project_face_recognition``. Реализован пайплайн распознавания лиц, осуществлён Fine tuning предобученной Inception ResNet v1, реализована метрика TPR@FPR, обучена модель на Cross Entropy loss, Tripletloss и на имплементированном ArcFace.
12. ``10. hw_dls_simple_embeddings``. Реализованы функция расчёта метрик HITS и DCG, функция ранжирования текстов на основе векторного представления слов. Обучена модель Word2Vec, проведено сравнение качества ранжирования на основе различных способов обработки, токенизаторов и эмбеддингов. 
13. ``11. hw_dls_text_classification``. Реализованы архитектуры с обычными RNN- и LSTM-блоками, обучены модели классификация текстов с двумя подходами к аггрегации эмбеддингов - осреднение и максимизация. Проведено несколько экспериментов, результаты по которым отражены в выводах в самом ноутбуке.  
14. ``12. hw_language_modelling``. Реализованы архитектуры с LSTM- и GRU-блоками, проведены эксперименты с различными подходами к оптимизации, а также с добавлением в архитектуру LayerNorm, дополнительных линейных слоёв и повышением размерности скрытого состояния. 
15. ``Seism_img_seg``. Работа не относится к домашним заданиям с курсов и выполнялась самостоятельно для научно-исследовательской работы. Реализованы архитектуры UNet, UNet with Hypercolumns и UNet with Spatial and Channel ‘Squeeze & Excitation’ blocks. Также реализована функция потерь на основе IoU и упрощённой Binary Cross Entropy, метрика mIoU. Обучены модели сегментации геологических слоёв на сейсмических изображениях, результаты всех моделей сравнивались на инференсе. 
16. ``Multi-label_classification``. Работа по мульти-лейбл классификации различной одежды, выполненная в качестве тренировки. Реализована предобработка данных и аугментация на основе случайного весового семплера, функция потерь для мульти-лейбл классификации, метрика и архитектура модели. Интегрирована библиотека ``wandb`` для логирования результатов и с помощью инструмента данной библиотеки реализован подбор оптимальных параметров с помощью sweeps. Также с помощью библиотеки ``shap`` проанализированы shap values для некоторых изображений и сделаны выводы.
17. ``Slot_attention_and_custom_module``. Работа по реализации классической модели объектно-ориентированного обучения -- Slot Attention из оригинальной [статьи](https://arxiv.org/pdf/2006.15055.pdf). Также реализован собственный модуль, основанный на обучаемых среднем и стандартном отклонении. 

## В папке ``Graphs and networks`` содержатся прорешанные домашние задания по курсу "Теория графов и сетей".
### Простые работы: 
1. ``Simple_graphs_and_graph_coloring``. Работа по реализации несложных графов, их визуализации и оценке некоторых параметров с использованием средств библиотеки ``networkx``. В работе также реализован жадный алгоритм вершинной раскраски графа. 
2. ``Flows``. Работа по решению задач о максимальном потоке в сетях с помощью библиотеки ``networkx`` и некоторых приложениях данной задачи, в частности нахождение непересекающихся по внутренним вершинам путей и оптимизации прибыли. 
3. ``Random_graphs_and_simple_hidden_markov_model``. Работа по реализации случайных графов с применением некоторых инструментов библиотеки ``networkx`` и несложной марковской модели с использованием библиотеки ``pomegranate``. 

## В папке ``NLP`` содержатся прорешанные задания по одноимённому курсу. 
### Простые работы: 
1. ``AdaM``. Работа по реализации упрощённого оптимизатора AdaM. 
2. ``Elasticsearch``. Работа по реализации лексического поиска с помощью ``rank-bm25`` и ``Elasticsearch``. 
3. ``FNN_and_CNN``. Работа по исследованию применимости полносвязных и свёрточных нейронных сетей для задач NLP, в частности классификации.
4. ``Seq2seq_and_attention``. Работа по реализации простенького переводчика с английского языка на русский с использованием модели seq2seq на основе механизма внимания.
5. ``BERT_and_BART``. Работа по реализации модели языкового моделирования на основе BERT, а также моделей BERT и BART из библиотеки ``transformers`` для задачи классификации. 
6. ``Simple_GPT``. Реализация языковой модели на основе упрощённой GPT.

### Более объёмные и сложные работы: 
7. ``Sentiment_analysis``. Работа по реализации модели анализа настроений по рецензиям на платформе IMDB. Реализован RNN классификатор, проанализированы предсказания модели, предложены пути повышения качества в виде аналогов Max Pooling и Mean Pooling, проанализировано качество классификации с применением GRU и LSTM, предпринята попытка улучшения предобработки данных с помощью GPT2 Tokenizer. 

## В папке ``Тренировки по ML (Яндекс)`` содержатся прорешанные домашние задания по классическим алгоритмам машинного обучения. 
### Реализованы: алгоритм KNN для классификации, распределение Лапласа, функции ошибки для линейной регрессии и их производные по параметрам, Power method, Bagging для задачи регрессии и out of bag оценивание, простейший бустинг для задачи регрессии, простенькая модель нейронных сетей для классификации, оценка важности признаков для логистической регрессии и бустинга с использованием встроенных инструментов и shap values.
