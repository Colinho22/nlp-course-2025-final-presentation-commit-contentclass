---
layout: gallery
title: "Embeddings Chart Gallery"
description: "33 visualizations covering word embedding concepts, Word2Vec architecture, training dynamics, and applications"
stats:
  total: 33
  categories: 5
categories:
  - id: fundamentals
    name: Fundamentals
  - id: word2vec
    name: Word2Vec
  - id: training
    name: Training
  - id: visualization
    name: Visualization
  - id: applications
    name: Applications
---

<!-- Fundamentals -->
{% include chart-card.html
   title="Word as Vector Concept"
   image="/assets/charts/embeddings/word_as_vector_concept_bsc.png"
   pdf="/assets/charts/embeddings/word_as_vector_concept_bsc.pdf"
   category="fundamentals"
   caption="Converting words to numerical vectors - the foundation of NLP"
%}

{% include chart-card.html
   title="Semantic Space 2D"
   image="/assets/charts/embeddings/semantic_space_2d_bsc.png"
   pdf="/assets/charts/embeddings/semantic_space_2d_bsc.pdf"
   category="fundamentals"
   caption="2D visualization showing how similar words cluster together"
%}

{% include chart-card.html
   title="Similarity Clustering"
   image="/assets/charts/embeddings/similarity_clustering_2d_bsc.png"
   pdf="/assets/charts/embeddings/similarity_clustering_2d_bsc.pdf"
   category="fundamentals"
   caption="Word clusters based on semantic similarity"
%}

{% include chart-card.html
   title="Embedding Space 3D"
   image="/assets/charts/embeddings/embedding_space_3d.png"
   pdf="/assets/charts/embeddings/embedding_space_3d.pdf"
   category="fundamentals"
   caption="3D visualization of word embeddings with semantic clustering"
%}

{% include chart-card.html
   title="Distributed Features"
   image="/assets/charts/embeddings/distributed_features_bsc.png"
   pdf="/assets/charts/embeddings/distributed_features_bsc.pdf"
   category="fundamentals"
   caption="How meaning is distributed across embedding dimensions"
%}

{% include chart-card.html
   title="Dimensionality Comparison"
   image="/assets/charts/embeddings/dimensionality_comparison_bsc.png"
   pdf="/assets/charts/embeddings/dimensionality_comparison_bsc.pdf"
   category="fundamentals"
   caption="Impact of embedding dimension on representation quality"
%}

<!-- Word2Vec Architecture -->
{% include chart-card.html
   title="Skip-gram Architecture"
   image="/assets/charts/embeddings/skipgram_architecture_bsc.png"
   pdf="/assets/charts/embeddings/skipgram_architecture_bsc.pdf"
   category="word2vec"
   caption="Neural network architecture for Skip-gram model"
%}

{% include chart-card.html
   title="CBOW Architecture"
   image="/assets/charts/embeddings/cbow_architecture_bsc.png"
   pdf="/assets/charts/embeddings/cbow_architecture_bsc.pdf"
   category="word2vec"
   caption="Continuous Bag of Words (CBOW) model structure"
%}

{% include chart-card.html
   title="Word2Vec Architectures Compared"
   image="/assets/charts/embeddings/word2vec_architectures.png"
   pdf="/assets/charts/embeddings/word2vec_architectures.pdf"
   category="word2vec"
   caption="Side-by-side comparison of Skip-gram and CBOW"
%}

{% include chart-card.html
   title="Skip-gram Training Steps"
   image="/assets/charts/embeddings/skipgram_training_steps_bsc.png"
   pdf="/assets/charts/embeddings/skipgram_training_steps_bsc.pdf"
   category="word2vec"
   caption="Step-by-step training process for Skip-gram"
%}

{% include chart-card.html
   title="Word2Vec Objectives"
   image="/assets/charts/embeddings/word2vec_objectives_comparison_bsc.png"
   pdf="/assets/charts/embeddings/word2vec_objectives_comparison_bsc.pdf"
   category="word2vec"
   caption="Comparison of training objectives for Word2Vec variants"
%}

<!-- Training -->
{% include chart-card.html
   title="Negative Sampling Process"
   image="/assets/charts/embeddings/negative_sampling_process_bsc.png"
   pdf="/assets/charts/embeddings/negative_sampling_process_bsc.pdf"
   category="training"
   caption="How negative sampling makes training efficient"
%}

{% include chart-card.html
   title="Negative Sampling Visualization"
   image="/assets/charts/embeddings/negative_sampling.png"
   pdf="/assets/charts/embeddings/negative_sampling.pdf"
   category="training"
   caption="Visual representation of positive vs negative samples"
%}

{% include chart-card.html
   title="Softmax Problem"
   image="/assets/charts/embeddings/softmax_problem_bsc.png"
   pdf="/assets/charts/embeddings/softmax_problem_bsc.pdf"
   category="training"
   caption="Computational challenge of full softmax over vocabulary"
%}

{% include chart-card.html
   title="Softmax Visualization"
   image="/assets/charts/embeddings/softmax_visualization.png"
   pdf="/assets/charts/embeddings/softmax_visualization.pdf"
   category="training"
   caption="How softmax converts scores to probabilities"
%}

{% include chart-card.html
   title="Training Dynamics"
   image="/assets/charts/embeddings/training_dynamics.png"
   pdf="/assets/charts/embeddings/training_dynamics.pdf"
   category="training"
   caption="Loss and accuracy curves during embedding training"
%}

{% include chart-card.html
   title="Training Evolution"
   image="/assets/charts/embeddings/training_evolution_bsc.png"
   pdf="/assets/charts/embeddings/training_evolution_bsc.pdf"
   category="training"
   caption="How embeddings change over training epochs"
%}

{% include chart-card.html
   title="GloVe Co-occurrence Heatmap"
   image="/assets/charts/embeddings/glove_cooccurrence_heatmap_bsc.png"
   pdf="/assets/charts/embeddings/glove_cooccurrence_heatmap_bsc.pdf"
   category="training"
   caption="Word co-occurrence matrix used by GloVe"
%}

{% include chart-card.html
   title="FastText Subword"
   image="/assets/charts/embeddings/fasttext_subword_bsc.png"
   pdf="/assets/charts/embeddings/fasttext_subword_bsc.pdf"
   category="training"
   caption="Character n-gram approach in FastText"
%}

<!-- Visualization -->
{% include chart-card.html
   title="Word Arithmetic 3D"
   image="/assets/charts/embeddings/word_arithmetic_3d_bsc.png"
   pdf="/assets/charts/embeddings/word_arithmetic_3d_bsc.pdf"
   category="visualization"
   caption="King - Man + Woman = Queen in 3D vector space"
%}

{% include chart-card.html
   title="Landmark Analogies 3D"
   image="/assets/charts/embeddings/landmark_analogies_3d.png"
   pdf="/assets/charts/embeddings/landmark_analogies_3d.pdf"
   category="visualization"
   caption="3D visualization of geographic and cultural analogies"
%}

{% include chart-card.html
   title="Cultural Analogies 3D"
   image="/assets/charts/embeddings/cultural_analogies_3d.png"
   pdf="/assets/charts/embeddings/cultural_analogies_3d.pdf"
   category="visualization"
   caption="Cultural relationships in embedding space"
%}

{% include chart-card.html
   title="Semantic Clusters 3D"
   image="/assets/charts/embeddings/semantic_clusters_3d.png"
   pdf="/assets/charts/embeddings/semantic_clusters_3d.pdf"
   category="visualization"
   caption="Word clusters visualized in 3D space"
%}

{% include chart-card.html
   title="Context-Dependent Bank"
   image="/assets/charts/embeddings/context_dependent_bank_bsc.png"
   pdf="/assets/charts/embeddings/context_dependent_bank_bsc.pdf"
   category="visualization"
   caption="How 'bank' gets different vectors in different contexts"
%}

{% include chart-card.html
   title="Contextual Bank"
   image="/assets/charts/embeddings/contextual_bank_bsc.png"
   pdf="/assets/charts/embeddings/contextual_bank_bsc.pdf"
   category="visualization"
   caption="Contextual embeddings for polysemous words"
%}

<!-- Applications -->
{% include chart-card.html
   title="Analogy Results"
   image="/assets/charts/embeddings/analogy_results_bsc.png"
   pdf="/assets/charts/embeddings/analogy_results_bsc.pdf"
   category="applications"
   caption="Performance on word analogy tasks"
%}

{% include chart-card.html
   title="Semantic Arithmetic"
   image="/assets/charts/embeddings/semantic_arithmetic.png"
   pdf="/assets/charts/embeddings/semantic_arithmetic.pdf"
   category="applications"
   caption="Vector operations that capture semantic relationships"
%}

{% include chart-card.html
   title="Pre-trained Comparison"
   image="/assets/charts/embeddings/pretrained_comparison_bsc.png"
   pdf="/assets/charts/embeddings/pretrained_comparison_bsc.pdf"
   category="applications"
   caption="Comparison of pre-trained embedding models"
%}

{% include chart-card.html
   title="Applications Dashboard"
   image="/assets/charts/embeddings/applications_dashboard.png"
   pdf="/assets/charts/embeddings/applications_dashboard.pdf"
   category="applications"
   caption="Real-world applications of word embeddings"
%}

{% include chart-card.html
   title="Applications Impact"
   image="/assets/charts/embeddings/week2_applications_impact.png"
   pdf="/assets/charts/embeddings/week2_applications_impact.pdf"
   category="applications"
   caption="Impact of embeddings on NLP applications"
%}

{% include chart-card.html
   title="Neural Evolution Timeline"
   image="/assets/charts/embeddings/week2_neural_evolution.png"
   pdf="/assets/charts/embeddings/week2_neural_evolution.pdf"
   category="applications"
   caption="Evolution from Word2Vec to transformer embeddings"
%}

{% include chart-card.html
   title="Evolution Timeline"
   image="/assets/charts/embeddings/evolution_timeline.png"
   pdf="/assets/charts/embeddings/evolution_timeline.pdf"
   category="applications"
   caption="Historical progression of embedding techniques"
%}

{% include chart-card.html
   title="Embedding Space Overview"
   image="/assets/charts/embeddings/week2_embedding_space.png"
   pdf="/assets/charts/embeddings/week2_embedding_space.pdf"
   category="applications"
   caption="Overview of embedding space properties"
%}
