/*
  Get a prediction from mean embeddings.
*/

#include "mean_embeddings.h"
#include "prediction.h"

void setup() {
  Serial.begin(9600);
}


void loop() {
  float sample_embedding[embedding_size] = { 0.6, 0, 0, 0 };
  char *ground_truth_label = "arden";
  int prediction_idx = get_prediction_idx(sample_embedding);
  const char *prediction = get_prediction_label(prediction_idx);

  // Write prediction to serial port.
  Serial.write("Prediction: ");
  Serial.write(prediction);
  Serial.write(" ");
  Serial.write(prediction == ground_truth_label ? "1\n" : "0\n");
  delay(100);
}
