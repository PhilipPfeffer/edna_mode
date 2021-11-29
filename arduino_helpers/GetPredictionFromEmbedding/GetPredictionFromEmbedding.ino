/*
  Get a prediction from mean embeddings.
*/

#include "mean_embeddings.h"

float MAX_FLOAT = 3.4028235E+38;

void setup() {
  Serial.begin(9600);
}

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

void loop() {
  float sample_embedding[embedding_size] = { 0.6, 0, 0, 0 };
  char *ground_truth_label = "arden";
  int prediction_idx = get_prediction_idx(sample_embedding);
  const char *prediction = mean_embeddings_labels[prediction_idx];

  // Write prediction to serial port.
  Serial.write("Prediction: ");
  Serial.write(prediction);
  Serial.write(" ");
  Serial.write(prediction == ground_truth_label ? "1\n" : "0\n");
  delay(100);
}
