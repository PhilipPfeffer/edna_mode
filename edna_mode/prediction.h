#include <stdint.h>
#define MAX_FLOAT 3.4028235E+38

// Compute the Euclidian distance.
float compute_dist(int8_t *sample_embedding, int mean_embedding_idx);

// Get the index of the prediction in the labels array.
// Use `get_prediction_label(prediction_idx);`
// to extract the prediction string.
int get_prediction_idx(int8_t *sample_embedding);

// Get the prediction label string from the index.
const char *get_prediction_label(int prediction_idx);
