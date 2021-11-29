#include "mean_embeddings.h"

const char *mean_embeddings_labels[num_labels] = {"phil", "arden", "greg"};
const float mean_embeddings[num_labels][embedding_size] = {
  { 0, 0, 0, 0 },
  { 1, 0, 0, 0 },
  { 0, 0, 0, 1 } };
