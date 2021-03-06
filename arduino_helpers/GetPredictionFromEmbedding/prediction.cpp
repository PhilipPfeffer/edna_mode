#include <math.h>
#include "mean_embeddings.h"
#include "prediction.h"

// Compute the Euclidian distance.
float compute_dist(float sample_embedding[], int mean_embedding_idx) {
  float sum = 0;
  for (int i = 0; i < embedding_size; i++) {
     sum += pow(sample_embedding[i] - mean_embeddings[mean_embedding_idx][i], 2);
  }
  return sqrt(sum);
}

// Get the index of the prediction in the labels array.
// Use `const char *prediction = mean_embeddings_labels[prediction_idx];`
// to extract the prediction string.
int get_prediction_idx(float sample_embedding[]) {
  int prediction_idx = -1;
  float min_dist = MAX_FLOAT;
  for (int i = 0; i < num_labels; i++) {
    float dist = compute_dist(sample_embedding, i);
    
    if (dist < min_dist) {
      min_dist = dist;
      prediction_idx = i;
    }
  }
  
  return prediction_idx;
}


const char *get_prediction_label(int prediction_idx) {
  return mean_embeddings_labels[prediction_idx];
}
